#include "prewarm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#include "overlay.h"
#include "present.h"
#include "vk.h"

namespace nvn {
namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

// ---- the shader cache directory: what is stale goes ---------------------------------------------------------------------------
// A translation file is named by the hash of everything that decides its content, ShaderCacheVersion included, so a file of an older
// version can never be found again. A stamp in the directory records the version that wrote the files: when it differs, the files are
// of the old version and removed. Files of the current version that nothing has used for a long time (use renews a file's time) and
// that no pipeline in the list needs are removed by a slow pass once a week, in the background. Only files named like the cache's own are
// ever touched.
bool CacheFileName(const fs::path& p) {
    const std::string name = p.filename().string();
    if (name.size() != 20 || name.compare(16, 4, ".bin") != 0) return false;
    return std::all_of(name.begin(), name.begin() + 16, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void RemoveOldVersionFiles(const std::string& dir) {
    const fs::path stamp = fs::path(dir) / ".version";
    const std::string current = std::to_string(ShaderCacheVersionValue());
    std::string previous;
    if (FILE* f = std::fopen(stamp.string().c_str(), "rb")) { char buf[32] = {}; if (std::fgets(buf, sizeof buf, f)) previous = buf; std::fclose(f); }
    if (previous == current) return;
    u32 removed = 0; u64 bytes = 0;
    if (!previous.empty()) { // (no stamp: first start with this feature, the files' ages decide)
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec) || !CacheFileName(it->path())) continue;
            const auto size = it->file_size(ec);
            if (fs::remove(it->path(), ec)) { ++removed; bytes += ec ? 0 : size; }
        }
        RT_LOG("shader cache: version %s -> %s, %u files (%.1f MiB) of the old version removed", previous.c_str(), current.c_str(), removed, bytes / 1048576.0);
    }
    if (FILE* f = std::fopen(stamp.string().c_str(), "wb")) { std::fputs(current.c_str(), f); std::fclose(f); }
}
void RemoveUnusedFiles(const std::string& dir, std::unordered_set<u64> needed) {
    const fs::path marker = fs::path(dir) / ".cleaned";
    std::error_code ec;
    const auto now = fs::file_time_type::clock::now();
    if (fs::exists(marker, ec)) { const auto t = fs::last_write_time(marker, ec); if (!ec && now - t < std::chrono::hours(24 * 7)) return; }
    if (FILE* f = std::fopen(marker.string().c_str(), "wb")) std::fclose(f);
    std::thread([dir, needed = std::move(needed), now] {
#ifdef __linux__
        setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 15);
#endif
#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#endif
        u32 removed = 0, kept = 0; u64 bytes = 0;
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec) || !CacheFileName(it->path())) continue;
            const auto written = it->last_write_time(ec);
            if (ec || now - written < std::chrono::hours(24 * 90)) { ++kept; continue; }
            u64 key = std::strtoull(it->path().filename().string().substr(0, 16).c_str(), nullptr, 16);
            if (needed.count(key)) { ++kept; continue; }
            const auto size = it->file_size(ec);
            if (fs::remove(it->path(), ec)) { ++removed; bytes += size; }
            std::this_thread::sleep_for(std::chrono::microseconds(200)); // (unhurried)
        }
        if (removed) RT_LOG("shader cache: %u translations unused for 90 days removed (%.1f MiB), %u kept", removed, bytes / 1048576.0, kept);
    }).detach();
}

// ---- budget ---------------------------------------------------------------------------------------------------------------
constexpr u64 MiB = 1ull << 20;
struct Budget { u64 available = 0, cap = 0; bool shared = false; };
Budget MemoryBudget() {
    Budget b;
    b.shared = Vk().properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU; // (integrated GPUs share the RAM with the game: the Steam Deck)
    b.available = plist::AvailableMemory();
    const double fraction = b.shared ? 0.03 : 0.08;
    b.cap = b.available ? (u64)(b.available * fraction) : 192 * MiB;
    b.cap = std::clamp<u64>(b.cap, 48 * MiB, b.shared ? 512 * MiB : 2048 * MiB);
    if (const char* mb = std::getenv("OOT_PREWARM_MB")) b.cap = std::strtoull(mb, nullptr, 10) * MiB; // (diagnostic)
    return b;
}

void LowerPriority() {
#ifdef __linux__
    setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);
#endif
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

// ---- the work -------------------------------------------------------------------------------------------------------------
struct Work {
    std::vector<plist::Record> records;
    std::atomic<size_t> next{0};
    std::atomic<u32> done{0}, built{0}, failed{0};
    std::atomic<u32> allowed{0};   // threads with a lower index than this keep working
    std::atomic<bool> background{false}, finished{false};
    std::atomic<u32> running{0};
    Clock::time_point start = Clock::now();
};

void Worker(std::shared_ptr<Work> work, u32 index) {
    bool lowered = false;
    for (;;) {
        if (index >= work->allowed.load(std::memory_order_relaxed)) break;
        if (work->background.load(std::memory_order_relaxed) && !lowered) { LowerPriority(); lowered = true; }
        const size_t i = work->next.fetch_add(1);
        if (i >= work->records.size()) break;
        if (PrewarmBuild(work->records[i])) ++work->built; else ++work->failed;
        ++work->done;
        if (work->background.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::milliseconds(2)); // (leaves the game its share)
    }
    if (--work->running == 0) work->finished = true;
}

} // namespace

void PrewarmMain() {
    if (!plist::Enabled()) return;
    const bool trace = plist::Trace();
    const auto t_begin = Clock::now();
    plist::Load(ShaderCacheVersionValue());
    const std::string dir = ShaderCacheDir();
    RemoveOldVersionFiles(dir);
    std::vector<plist::Record> all = plist::Snapshot();
    {
        std::unordered_set<u64> needed;
        for (const auto& r : all) for (const auto& s : r.shaders) needed.insert(s.cache_key);
        RemoveUnusedFiles(dir, std::move(needed));
    }
    if (all.empty()) { if (trace) RT_LOG("prewarm: no pipeline list yet (first run), nothing to do"); return; }

    Vk(); // the window and the device
    const Budget budget = MemoryBudget();
    const u64 rss_before = plist::ProcessResidentBytes();

    // Most valuable first, as many as the memory budget holds: a pipeline is estimated at 128 KiB (compute 256 KiB; measured on NVIDIA: about 175 KiB each besides the translations) plus the translations it
    // needs (the size of their cache files, counted once).
    auto work = std::make_shared<Work>();
    {
        u64 cost = 0; std::unordered_map<u64, u64> counted; size_t skipped = 0;
        for (auto& r : all) {
            u64 add = r.kind == plist::Kind::Compute ? 256 * 1024 : 128 * 1024, files = 0;
            std::vector<std::pair<u64, u64>> fresh;
            for (const auto& s : r.shaders) if (!counted.count(s.cache_key)) {
                const u64 bytes = ShaderCacheBytes(s.cache_key);
                if (!bytes) { add = ~0ull; break; } // (its translation is gone: nothing to make this from)
                counted[s.cache_key] = bytes; fresh.emplace_back(s.cache_key, bytes); files += bytes;
            }
            if (add == ~0ull || cost + add + files > budget.cap) { for (const auto& f : fresh) counted.erase(f.first); ++skipped; continue; }
            cost += add + files;
            work->records.push_back(std::move(r));
        }
        if (trace || skipped) RT_LOG("prewarm: %zu pipelines listed, %zu selected (about %.0f MiB of a %.0f MiB budget; %.0f MiB of memory available, %s), %zu left for the game to make",
            all.size(), work->records.size(), cost / (double)MiB, budget.cap / (double)MiB, budget.available / (double)MiB, budget.shared ? "shared-memory GPU" : "discrete GPU", skipped);
    }
    if (work->records.empty()) return;
    PipelineCache(); // (created here: on a thread of the pool it would race)

    const u32 hardware = std::max(1u, std::thread::hardware_concurrency());
    const u32 threads = budget.shared ? std::max(1u, hardware / 2) : std::max(1u, std::min(8u, hardware > 3 ? hardware - 2 : 1u));
    work->allowed = threads; work->running = threads;
    for (u32 i = 0; i < threads; ++i) std::thread(Worker, work, i).detach();

    // The screen: not for a short wait.
    std::vector<SDL_Gamepad*> pads;
    {
        int count = 0; SDL_JoystickID* ids = SDL_GetGamepads(&count);
        for (int i = 0; ids && i < count; ++i) if (SDL_Gamepad* p = SDL_OpenGamepad(ids[i])) pads.push_back(p);
        SDL_free(ids);
    }
    const auto shown_after = work->start + std::chrono::milliseconds(300);
    bool skip = false, shown = false;
    auto next_frame = Clock::now();
    const u32 total = (u32)work->records.size();
    while (!work->finished && !skip) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) { plist::Save(true); std::_Exit(0); }
            if ((e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat) || e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) skip = true;
            overlay::ProgressEvent(e);
        }
        if (!shown && Clock::now() >= shown_after) shown = true;
        if (shown) {
            const u32 done = work->done.load();
            const double elapsed = std::chrono::duration<double>(Clock::now() - work->start).count();
            overlay::Progress progress{done, total, done >= 6 && elapsed > 0.4 ? elapsed / done * (total - done) : -1.0};
            if (PresentProgress(progress)) skip = true;
        }
        next_frame += std::chrono::microseconds(16667);
        if (next_frame < Clock::now()) next_frame = Clock::now();
        std::this_thread::sleep_until(next_frame);
    }
    for (SDL_Gamepad* p : pads) SDL_CloseGamepad(p);
    const double seconds = std::chrono::duration<double>(Clock::now() - work->start).count();
    if (skip && !work->finished) {
        // The rest goes on at low priority, on fewer threads; what the game needs sooner it makes itself.
        work->allowed = std::max(1u, threads / 3);
        work->background = true;
        RT_LOG("prewarm: skipped after %.1f s with %u of %u made; the rest continues in the background", seconds, work->done.load(), total);
        return;
    }
    const u64 rss_after = plist::ProcessResidentBytes();
    RT_LOG("prewarm: %u pipelines made (%u could not be: translation or driver) in %.2f s on %u threads; memory +%.0f MiB; total start-up work %.2f s",
           work->built.load(), work->failed.load(), seconds, threads, rss_after > rss_before ? (rss_after - rss_before) / (double)MiB : 0.0,
           std::chrono::duration<double>(Clock::now() - t_begin).count());
}

} // namespace nvn
