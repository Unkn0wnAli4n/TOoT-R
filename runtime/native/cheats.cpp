// Cheats (the settings window's Cheats tab): values of the game's own data manager held where the player wants them.
// The game keeps its named values (the save's flags and counters) in GameDataMgr, in one container per type: the
// count at +8, the entries at +16, a table of {hash, entry} pairs at +24 probed from hash % size (+36); a value is
// named by the MurmurHash3 (seed 0) of its name. The manager's holder is at main+0x4c7e180. (The layout and the
// names are Longshot's findings, host/oot_state_tap.c and hos_main.c; this code is this runtime's.)
// Tick() runs on the game's thread once a frame (present pacing). Nothing is written while every cheat is off.
// OOT_GAMEDATA_PROBE="Name,Name,..." logs whether each name exists (as an Int, Float, Bool or UInt) and its value.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../addresses.h"
#include "../rt.h"
#include "graphics_settings.h"

namespace cheats {
namespace {
constexpr u64 Main = 0x800000000ull;
u64 HolderAddress() { return OOT_AT(0x4c7e180); }
#define Holder HolderAddress() // (Longshot's 0x84c9e180: its addresses are main + 0x80020000)
enum Kind { Bool, Int, Float, Enum, Text, UInt, Kinds };
constexpr struct { u32 at, stride, value; } Layout[Kinds] = {{0x008, 0x18, 0x16}, {0x0c8, 0x28, 0x1c}, {0x188, 0x28, 0x1c}, {0x248, 0x38, 0x1c}, {0x608, 0x108, 0x68}, {0x788, 0x28, 0x1c}};
bool Readable(u64 a) { return a >= Main && a < (1ull << 39) - 0x200; }
template <typename T> T Load(u64 a) { T v{}; if (Readable(a)) std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof v); return v; }
template <typename T> void Store(u64 a, T v) { if (Readable(a)) std::memcpy(reinterpret_cast<void*>(a), &v, sizeof v); }
u32 Murmur(const char* s) {
    const size_t n = std::strlen(s); u32 h = 0, k; size_t i = 0;
    const auto b = [&](size_t j) { return (u32)(unsigned char)s[j]; };
    for (; i + 4 <= n; i += 4) { k = b(i) | b(i + 1) << 8 | b(i + 2) << 16 | b(i + 3) << 24; k *= 0xcc9e2d51u; k = k << 15 | k >> 17; k *= 0x1b873593u; h ^= k; h = h << 13 | h >> 19; h = h * 5 + 0xe6546b64u; }
    k = 0;
    switch (n & 3) { case 3: k ^= b(i + 2) << 16; [[fallthrough]]; case 2: k ^= b(i + 1) << 8; [[fallthrough]]; case 1: k ^= b(i); k *= 0xcc9e2d51u; k = k << 15 | k >> 17; k *= 0x1b873593u; h ^= k; }
    h ^= (u32)n; h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h;
}
u64 Manager() { if (!Holder) return 0; const u64 holder = Load<u64>(Holder); return Readable(holder) ? Load<u64>(holder) : 0; }
// The address of the value named by `hash` in the container `kind` (0: it has none).
u64 Value(u64 manager, Kind kind, u32 hash) {
    if (!Readable(manager) || !hash) return 0;
    const u64 box = manager + Layout[kind].at, entries = Load<u64>(box + 16), table = Load<u64>(box + 24);
    const u32 count = Load<u32>(box + 8), size = Load<u32>(box + 36);
    if (!Readable(entries) || !Readable(table) || !size || size > (1u << 22)) return 0;
    for (u32 probe = 0, slot = hash % size; probe < size && probe < 4096; ++probe, slot = slot + 1 == size ? 0 : slot + 1) {
        const u32 found = Load<u32>(table + slot * 8ull), entry = Load<u32>(table + slot * 8ull + 4);
        if (!found) return 0;
        if (found == hash) return entry < count ? entries + (u64)entry * Layout[kind].stride + Layout[kind].value : 0;
    }
    return 0;
}
void Probe(u64 manager) {
    static const char* names = std::getenv("OOT_GAMEDATA_PROBE"); static bool done = false;
    if (!names || done) return;
    if (!Value(manager, Int, 0x31ab5580u)) return; // (the save is not loaded yet)
    done = true;
    std::string all = names; size_t at = 0;
    while (at <= all.size()) {
        const size_t end = std::min(all.find(',', at), all.size()); const std::string name = all.substr(at, end - at); at = end + 1;
        if (name.empty()) continue;
        const u32 hash = Murmur(name.c_str()); bool any = false;
        static const char* const kinds[Kinds] = {"Bool", "Int", "Float", "Enum", "Text", "UInt"};
        for (int k = 0; k < Kinds; ++k) if (k != Text) if (const u64 v = Value(manager, (Kind)k, hash)) { any = true;
            if (k == Float) RT_LOG("game data: %s (%#x) is a %s: %g", name.c_str(), hash, kinds[k], Load<float>(v));
            else if (k == Bool) RT_LOG("game data: %s (%#x) is a %s: %d", name.c_str(), hash, kinds[k], Load<u8>(v));
            else RT_LOG("game data: %s (%#x) is a %s: %d", name.c_str(), hash, kinds[k], Load<s32>(v)); }
        if (!any) RT_LOG("game data: no value named %s (%#x)", name.c_str(), hash);
    }
}
} // namespace

void Tick() {
    const gfx::Settings& g = gfx::Current();
    static const bool probing = std::getenv("OOT_GAMEDATA_PROBE") != nullptr;
    const bool no_damage = g.damage_taken == 4 || g.cheat_health; // (Damage Taken at None: the life write is held where its hook is, and the hearts are filled here too)
    if (!no_damage && !g.cheat_rupees && !g.cheat_magic && !probing) return;
    const u64 manager = Manager();
    if (!manager) { static int n = 0; if (probing && n++ % 600 == 0 && n < 3000) RT_LOG("game data: no manager yet (holder %#llx)", (unsigned long long)(Holder ? Load<u64>(Holder) : 0)); return; }
    if (probing) Probe(manager);
    if (no_damage) { // (quarter hearts)
        const u64 life = Value(manager, Int, 0x31ab5580u), most = Value(manager, Int, 0xfbe01da1u); // PlayerStatus.Life, .MaxLife
        if (life && most) { const s32 m = Load<s32>(most); if (m >= 4 && m <= 400 && Load<s32>(life) < m && Load<s32>(life) > 0) Store<s32>(life, m); }
    }
    if (g.cheat_rupees) { // (never fewer than the most held since the cheat went on)
        static s32 kept = -1; static u64 of = 0;
        const u64 rupees = Value(manager, Int, 0xa77921d7u); // PlayerStatus.CurrentRupee
        if (rupees) { const s32 now = Load<s32>(rupees); if (of != manager) { of = manager; kept = now; } if (now > kept) kept = now; else if (now < kept && now >= 0 && kept <= 9999) Store<s32>(rupees, kept); }
    }
    if (g.cheat_magic) {
        static const u32 magic = Murmur("PlayerStatus.MP"), most = Murmur("PlayerStatus.MPMax"); // (both 0 until Link has magic: nothing is given)
        const u64 m = Value(manager, Int, magic), top = Value(manager, Int, most);
        if (m && top) { const s32 t = Load<s32>(top); if (t > 0 && t <= 1000 && Load<s32>(m) < t) Store<s32>(m, t); }
    }
}
} // namespace cheats
