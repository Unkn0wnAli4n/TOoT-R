#include "addresses.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "platform.h"
#include "rt.h"

namespace rt {
namespace {
constexpr uint64_t bases[] = {0x800000000ull, 0x810000000ull, 0x820000000ull};
const char* const names[] = {"main", "subsdk0", "sdk"};
std::atomic<unsigned> missing{0};
struct Table {
    bool loaded = false; // (no file: a development tree of the reference version - every place is where it was)
    std::unordered_map<uint64_t, uint64_t> at[3];
};
const Table& Addresses() {
    static const Table table = [] {
        Table t;
        const char* env = std::getenv("OOT_ADDRESSES");
        const std::string path = env && *env ? env : DataDir() + "/addresses.txt";
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) { RT_LOG("addresses: no %s: the reference version's own are used", path.c_str()); return t; }
        char module[16]; unsigned long long ref, now; size_t count = 0, moved = 0;
        while (std::fscanf(f, "%15s %llx %llx", module, &ref, &now) == 3)
            for (int m = 0; m < 3; ++m) if (std::string(module) == names[m]) { t.at[m][ref] = now; ++count; moved += ref != now; }
        std::fclose(f);
        t.loaded = true;
        RT_LOG("addresses: %zu places of this game version (%zu at another address than in the reference version)", count, moved);
        return t; }();
    return table;
}
} // namespace

uint64_t AddressOf(uint64_t reference_offset, GameModule module) {
    const Table& t = Addresses();
    const int m = (int)module;
    if (!t.loaded) return bases[m] + reference_offset;
    const auto it = t.at[m].find(reference_offset);
    if (it == t.at[m].end()) {
        static std::mutex mutex; static std::unordered_map<uint64_t, bool> told;
        std::lock_guard lock{mutex};
        if (!told[reference_offset * 4 + m]) { told[reference_offset * 4 + m] = true; ++missing; RT_LOG("addresses: %s+%#" PRIx64 " of the reference version is not in this game version: what uses it is off", names[m], reference_offset); }
        return 0;
    }
    return bases[m] + it->second;
}
unsigned AddressesMissing() { return missing.load(); }
} // namespace rt
