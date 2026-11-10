#include "pipeline_list.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "../../platform.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace nvn::plist {
namespace {

constexpr u32 Magic = 0x4c505454; // "TTPL"
constexpr u32 FormatVersion = 1;
constexpr size_t MaxRecords = 20000;
constexpr u64 ForgetAfterSeconds = 90ull * 86400;

struct Header { u32 magic, format, shader_cache_version, reserved; };

std::mutex mutex;
bool loaded = false;
u32 cache_version = 0;
std::vector<Record> records;
std::unordered_map<u64, size_t> index;
std::unordered_set<u64> used_this_run;
bool dirty = false, incompatible = false;
std::chrono::steady_clock::time_point last_save = std::chrono::steady_clock::now();

std::string Path() { return rt::DataDir() + "/pipeline_list.bin"; }
u64 Now() { return (u64)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
double Score(const Record& r, u64 now) { const double days = now > r.last_used ? (now - r.last_used) / 86400.0 : 0.0; return r.uses / (1.0 + days / 14.0); }

struct Writer {
    std::vector<u8> bytes;
    template <typename T> void put(const T& v) { const u8* b = reinterpret_cast<const u8*>(&v); bytes.insert(bytes.end(), b, b + sizeof v); }
};
void Encode(Writer& w, const Record& r) {
    Writer body;
    body.put(r.id); body.put(r.uses); body.put(r.last_used); body.put((u8)r.kind); body.put((u8)r.shaders.size());
    for (const auto& s : r.shaders) { body.put(s.stage); body.put(s.cache_key); body.put(s.identity); }
    body.put((u32)r.state.size());
    body.bytes.insert(body.bytes.end(), r.state.begin(), r.state.end());
    w.put((u32)body.bytes.size());
    w.bytes.insert(w.bytes.end(), body.bytes.begin(), body.bytes.end());
}
bool Decode(const u8* p, size_t n, Record& r) {
    size_t at = 0;
    const auto get = [&](auto& v) { if (at + sizeof v > n) return false; std::memcpy(&v, p + at, sizeof v); at += sizeof v; return true; };
    u8 kind = 0, count = 0; u32 state = 0;
    if (!get(r.id) || !get(r.uses) || !get(r.last_used) || !get(kind) || !get(count) || kind > 2 || count > 8) return false;
    r.kind = (Kind)kind; r.shaders.resize(count);
    for (auto& s : r.shaders) if (!get(s.stage) || !get(s.cache_key) || !get(s.identity)) return false;
    if (!get(state) || at + state != n) return false;
    r.state.assign(p + at, p + at + state);
    return MakeId(r.kind, r.shaders, r.state) == r.id; // (a damaged record is dropped)
}

} // namespace

bool Enabled() { static const bool on = !std::getenv("OOT_NO_PREWARM") || std::atoi(std::getenv("OOT_NO_PREWARM")) == 0; return on; }
bool Trace() { static const bool on = std::getenv("OOT_PREWARM_TRACE") != nullptr; return on; }

u64 MakeId(Kind kind, const std::vector<ShaderRef>& shaders, const std::vector<u8>& state) {
    u64 h = 0xcbf29ce484222325ull;
    const auto mix = [&](const void* p, size_t n) { const u8* b = static_cast<const u8*>(p); for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull; };
    const u8 k = (u8)kind; mix(&k, 1);
    for (const auto& s : shaders) { mix(&s.stage, 4); mix(&s.cache_key, 8); mix(&s.identity, 8); }
    mix(state.data(), state.size());
    return h;
}

void Load(u32 shader_cache_version) {
    std::lock_guard lock{mutex};
    if (loaded) return;
    loaded = true; cache_version = shader_cache_version;
    FILE* f = std::fopen(Path().c_str(), "rb");
    if (!f) return;
    std::vector<u8> data;
    { u8 buf[65536]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) data.insert(data.end(), buf, buf + n); }
    std::fclose(f);
    Header h{};
    if (data.size() < sizeof h) { incompatible = true; return; }
    std::memcpy(&h, data.data(), sizeof h);
    if (h.magic != Magic || h.format != FormatVersion || h.shader_cache_version != shader_cache_version) {
        if (Trace()) RT_LOG("pipeline list: format %u / shader cache %u does not match, starting over", h.format, h.shader_cache_version);
        incompatible = dirty = true; // (replaced at the next save; appending to it would mix formats)
        return;
    }
    size_t at = sizeof h;
    while (at + 4 <= data.size()) {
        u32 size; std::memcpy(&size, data.data() + at, 4);
        if (size > (1u << 20) || at + 4 + size > data.size()) break; // (a record cut short by a crash)
        Record r;
        if (Decode(data.data() + at + 4, size, r) && !index.count(r.id)) { index[r.id] = records.size(); records.push_back(std::move(r)); }
        at += 4 + size;
    }
}

size_t Count() { std::lock_guard lock{mutex}; return records.size(); }

std::vector<Record> Snapshot() {
    std::vector<Record> out;
    { std::lock_guard lock{mutex}; out = records; }
    const u64 now = Now();
    std::stable_sort(out.begin(), out.end(), [&](const Record& a, const Record& b) { const double x = Score(a, now), y = Score(b, now); return x != y ? x > y : a.last_used > b.last_used; });
    return out;
}

bool Add(Record r) {
    std::lock_guard lock{mutex};
    if (!loaded) return false;
    r.id = MakeId(r.kind, r.shaders, r.state);
    if (index.count(r.id)) return false;
    r.uses = 1; r.last_used = Now();
    used_this_run.insert(r.id);
    if (incompatible) dirty = true; // (the next save writes the whole list)
    else {
        const std::string path = Path();
        bool fresh = true;
        if (FILE* probe = std::fopen(path.c_str(), "rb")) { std::fclose(probe); fresh = false; }
        if (FILE* f = std::fopen(path.c_str(), "ab")) {
            Writer w;
            if (fresh) { Header h{Magic, FormatVersion, cache_version, 0}; w.put(h); }
            Encode(w, r);
            std::fwrite(w.bytes.data(), 1, w.bytes.size(), f);
            std::fclose(f);
        }
    }
    index[r.id] = records.size();
    records.push_back(std::move(r));
    if (records.size() > MaxRecords) dirty = true;
    return true;
}

void Use(u64 id) {
    std::lock_guard lock{mutex};
    auto it = index.find(id);
    if (it == index.end() || !used_this_run.insert(id).second) return;
    auto& r = records[it->second];
    ++r.uses; r.last_used = Now();
    dirty = true;
}

void Save(bool force) {
    std::lock_guard lock{mutex};
    if (!loaded || !dirty) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - last_save < std::chrono::seconds(60)) return;
    last_save = now;
    const u64 t = Now();
    std::vector<Record> keep;
    for (const auto& r : records) if (t < r.last_used || t - r.last_used < ForgetAfterSeconds) keep.push_back(r);
    if (keep.size() > MaxRecords) { // (least valuable out)
        std::stable_sort(keep.begin(), keep.end(), [&](const Record& a, const Record& b) { return Score(a, t) > Score(b, t); });
        keep.resize(MaxRecords);
    }
    Writer w;
    Header h{Magic, FormatVersion, cache_version, 0};
    w.put(h);
    for (const auto& r : keep) Encode(w, r);
    const std::string path = Path(), temporary = path + ".tmp";
    if (FILE* f = std::fopen(temporary.c_str(), "wb")) {
        const bool ok = std::fwrite(w.bytes.data(), 1, w.bytes.size(), f) == w.bytes.size();
        std::fclose(f);
        std::error_code ec;
        if (ok) std::filesystem::rename(temporary, path, ec);
        if (ok && !ec) {
            if (keep.size() != records.size()) { records = std::move(keep); index.clear(); for (size_t i = 0; i < records.size(); ++i) index[records[i].id] = i; }
            dirty = incompatible = false;
        }
    }
}

u64 AvailableMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX status{}; status.dwLength = sizeof status;
    return GlobalMemoryStatusEx(&status) ? (u64)status.ullAvailPhys : 0;
#else
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256]; unsigned long long kb = 0;
    while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    std::fclose(f);
    return (u64)kb << 10;
#endif
}

u64 ProcessResidentBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    return GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters) ? (u64)counters.WorkingSetSize : 0;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long size = 0, resident = 0;
    if (std::fscanf(f, "%llu %llu", &size, &resident) != 2) resident = 0;
    std::fclose(f);
    return resident * 4096;
#endif
}

} // namespace nvn::plist
