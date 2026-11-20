// The Cheats tab's tools that act inside the game: teleport, the time of day, Link's age, the inventory, flight, a free
// camera, damage taken, movement speed and the autosave switch. What the game's own functions and data are (the event
// request a map change is made of, the pouch's add / remove event, the game data manager's tables, the controller a
// flight moves) was worked out by Longshot for this same game version (host/hos_main.c, host/oot_damage.c,
// gpu/vk/oot_cheat_items.h: GPL-3.0-or-later); this file is written for this runtime, where guest addresses are host
// addresses and the game's functions are called with rt_call.
//
// Everything runs on the game's own thread, at Link's movement update (site 95 of tools/oot_patches.py): the settings
// window only leaves a request.
#include "debug_tools.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "../ops.h"
#include "../rt.h"
#include "../addresses.h"
#include "graphics_settings.h"
#include "input_settings.h"

void oot_note_link(unsigned long long components); // (engine_hooks.cpp: Link's body, for the ledge fix above 60 fps)

namespace tools {
namespace {
constexpr u64 Main = 0x800000000ull;
// G(0x84c9e180): a place of the reference version by the address Longshot's notes give it (main + 0x80020000), in this
// game (rt::AddressOf: 0 when this game version has no such place; tools/signatures.py reads these too).
#define G(theirs) OOT_AT((theirs) - 0x80020000ull)
bool Readable(u64 a) { return a >= Main && a < (1ull << 39) - 0x100; }
template <typename T> T Load(u64 a) { T v; std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof v); return v; }
template <typename T> void Store(u64 a, T v) { std::memcpy(reinterpret_cast<void*>(a), &v, sizeof v); }
s64 NowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
float LowFloat(const V128& v) { float f; std::memcpy(&f, &v, 4); return f; }
void SetLowFloat(V128& v, float f) { u64 lo = 0; std::memcpy(&lo, &f, 4); v = Q(lo, 0); }

// A call of one of the game's functions from a hook: the registers as they were afterwards (the caller reads its
// result from the Cpu before End).
struct Call {
    Cpu* c; Cpu saved;
    explicit Call(Cpu* cpu) : c(cpu), saved(*cpu) {}
    u64 Stack(u64 bytes) { c->sp = (c->sp - bytes) & ~0xfull; std::memset(reinterpret_cast<void*>(c->sp), 0, bytes); return c->sp; }
    void Run(u64 fn) { if (!fn) return; c->x[30] = 0; (rt_call)(c, fn); }
    ~Call() { const bool excl = false; *c = saved; c->excl_valid = excl; }
};

u64 Singleton(u64 slot) { if (!slot) return 0; const u64 holder = Load<u64>(slot); return Readable(holder) ? Load<u64>(holder) : 0; }
#define GameDataSlot G(0x84c9e180)
#define EventMgrSlot G(0x84c9e090)

// GameDataMgr's getters by MurmurHash3 of the name: f_8022d9c0 (bool) and f_80cd0ea0 (a string).
bool DataBool(Cpu* c, u32 hash, bool* value) {
    const u64 mgr = Singleton(GameDataSlot);
    if (!Readable(mgr)) return false;
    Call call{c};
    const u64 out = call.Stack(0x40);
    c->x[0] = mgr; c->x[1] = out; c->x[2] = hash;
    call.Run(G(0x8022d9c0));
    *value = Load<u8>(out) & 1;
    return c->x[0] & 1;
}
bool DataString(Cpu* c, u32 hash, std::string* text) {
    text->clear();
    const u64 mgr = Singleton(GameDataSlot);
    if (!Readable(mgr)) return false;
    u64 at; bool found;
    {
        Call call{c};
        const u64 out = call.Stack(0x40);
        c->x[0] = mgr; c->x[1] = out; c->x[2] = hash;
        call.Run(G(0x80cd0ea0));
        found = c->x[0] & 1; at = Load<u64>(out);
    }
    if (!found || !Readable(at)) return false;
    for (int i = 0; i < 79; ++i) { const char ch = Load<char>(at + i); if (!ch) break; text->push_back(ch); }
    return true;
}
// A bool written in place: GameDataMgr+32 is the (hash, index) table (+44 its size), the bools 24-byte entries at +24
// (+16 their count), the value at +22.
bool DataSetBool(u32 hash, bool value) {
    const u64 mgr = Singleton(GameDataSlot);
    if (!Readable(mgr)) return false;
    const u64 table = Load<u64>(mgr + 32); const u32 size = Load<u32>(mgr + 44);
    if (!Readable(table) || !size) return false;
    for (u32 probe = 0, slot = hash % size; probe < size; ++probe, slot = (slot + 1) % size) {
        const u32 key = Load<u32>(table + slot * 8ull);
        if (!key) return false;
        if (key != hash) continue;
        const u32 index = Load<u32>(table + slot * 8ull + 4), count = Load<u32>(mgr + 16);
        if (index >= count) return false;
        Store<u8>(Load<u64>(mgr + 24) + index * 24ull + 22, value ? 1 : 0);
        return true;
    }
    return false;
}
constexpr u32 IsChildhood = 0xcea220d3u, CurrentBanc = 0x1d6189dau, IsLoadingBanc = 0xef2b60c2u, MapIconsOff = 0x9a4dd779u;

// Every place the tools use is in this game version (rt::AddressOf): one missing, and none of them runs - a half of a
// teleport or of an inventory change is worse than none.
bool Ready() {
    static const bool ready = [] {
        const u64 all[] = {G(0x8022b6f0), G(0x8022d9c0), G(0x802dc440), G(0x805b63e0), G(0x806a2b20), G(0x806d2530), G(0x80bc7710), G(0x80bc7764), G(0x80bc7864), G(0x80cd0ea0), G(0x80d99f10), G(0x80d9a820), G(0x828d4a70), G(0x832a9764), G(0x834722d0), G(0x84994b20), G(0x84b6c460), G(0x84c18a88), G(0x84c9e090), G(0x84c9e0d0), G(0x84c9e180), G(0x84c9e2a8), G(0x84c9e898), G(0x84ca6bc0), G(0x84e8c060), G(0x84e98e28), G(0x84eb6888)};
        unsigned missing = 0; for (const u64 a : all) missing += !a;
        if (missing) RT_LOG("tools: %u of the game's functions and data the Cheats tab's tools use are not in this game version: the tools are off", missing);
        return !missing; }();
    return ready;
}
bool EventPlaying() { const u64 events = Singleton(EventMgrSlot); return Readable(events) && Load<u64>(events + 104); }

// ---- the inventory ----------------------------------------------------------------------------------------------------------
const std::vector<Item> items = {
#define S(section)
    {"Kokiri Sword", "Weapon_Sword_Kokiri", 0, "Equipment"}, {"Master Sword", "Weapon_Sword_Master", 0, "Equipment"},
    {"Giant's Knife", "Weapon_Sword_Giant", 0, "Equipment"}, {"Biggoron's Sword", "Weapon_Sword_Biggoron", 0, "Equipment"},
    {"Deku Shield", "Weapon_Shield_Deku", 0, "Equipment"}, {"Hylian Shield", "Weapon_Shield_Hylian", 0, "Equipment"},
    {"Mirror Shield", "Weapon_Shield_Mirror", 0, "Equipment"}, {"Goron Tunic", "Armor_Adult_Goron", 0, "Equipment"},
    {"Zora Tunic", "Armor_Adult_Zora", 0, "Equipment"}, {"Iron Boots", "Armor_Adult_HeavyBoot", 0, "Equipment"},
    {"Hover Boots", "Armor_Adult_HoverBoot", 0, "Equipment"},
    {"Deku Sticks", "Item_DekuStick", 1, "Items"}, {"Deku Nuts", "Item_DekuNut", 1, "Items"}, {"Bombs", "Item_Bomb", 1, "Items"},
    {"Bombchus", "Item_Bombchu", 1, "Items"}, {"Fairy Slingshot", "Weapon_Range_SlingShot", 0, "Items"},
    {"Deku Seeds", "Item_DekuSeed", 1, "Items"}, {"Fairy Bow", "Weapon_Range_Bow", 0, "Items"},
    {"Arrows", "Item_Arrow_Normal", 1, "Items"}, {"Fire Arrow", "Item_Arrow_Fire", 0, "Items"},
    {"Ice Arrow", "Item_Arrow_Ice", 0, "Items"}, {"Light Arrow", "Item_Arrow_Light", 0, "Items"},
    {"Din's Fire", "Item_DinFire", 0, "Items"}, {"Farore's Wind", "Item_FaroreWind", 0, "Items"},
    {"Nayru's Love", "Item_NayruLove", 0, "Items"}, {"Fairy Ocarina", "Item_Ocarina_Fairy", 0, "Items"},
    {"Ocarina of Time", "Item_Ocarina_Time", 0, "Items"}, {"Hookshot", "Item_Hookshot", 0, "Items"},
    {"Longshot", "Item_Longshot", 0, "Items"}, {"Boomerang", "Item_Boomerang", 0, "Items"},
    {"Lens of Truth", "Item_GlassTruth", 0, "Items"}, {"Magic Beans", "Item_MagicBean", 1, "Items"},
    {"Megaton Hammer", "Item_Hammer", 0, "Items"}, {"Empty Bottle", "Item_Bottle", 1, "Items"},
    {"Keaton Mask", "Armor_Child_Mask_Keaton", 0, "Masks"}, {"Skull Mask", "Armor_Child_Mask_Skull", 0, "Masks"},
    {"Spooky Mask", "Armor_Child_Mask_Spooky", 0, "Masks"}, {"Bunny Hood", "Armor_Child_Mask_Bunny", 0, "Masks"},
    {"Goron Mask", "Armor_Child_Mask_Goron", 0, "Masks"}, {"Zora Mask", "Armor_Child_Mask_Zora", 0, "Masks"},
    {"Gerudo Mask", "Armor_Child_Mask_Gerudo", 0, "Masks"}, {"Mask of Truth", "Armor_Child_Mask_Truth", 0, "Masks"},
    {"Goron's Bracelet", "Item_Gauntlet_Goron", 0, "Upgrades"}, {"Silver Gauntlets", "Armor_Adult_Gauntlet_Silver", 0, "Upgrades"},
    {"Golden Gauntlets", "Armor_Adult_Gauntlet_Gold", 0, "Upgrades"}, {"Silver Scale", "Item_Scale_Silver", 0, "Upgrades"},
    {"Golden Scale", "Item_Scale_Gold", 0, "Upgrades"}, {"Bomb Bag", "Item_BombBag_Small", 0, "Upgrades"},
    {"Big Bomb Bag", "Item_BombBag_Medium", 0, "Upgrades"}, {"Biggest Bomb Bag", "Item_BombBag_Large", 0, "Upgrades"},
    {"Quiver", "Item_Quiver_Small", 0, "Upgrades"}, {"Big Quiver", "Item_Quiver_Medium", 0, "Upgrades"},
    {"Biggest Quiver", "Item_Quiver_Large", 0, "Upgrades"}, {"Bullet Bag", "Item_BulletBag_Small", 0, "Upgrades"},
    {"Big Bullet Bag", "Item_BulletBag_Medium", 0, "Upgrades"}, {"Biggest Bullet Bag", "Item_BulletBag_Large", 0, "Upgrades"},
    {"Adult's Wallet", "Item_Wallet_Medium", 0, "Upgrades"}, {"Giant's Wallet", "Item_Wallet_Large", 0, "Upgrades"},
    {"Magic Meter", "Item_FairyBless_MP", 0, "Upgrades"}, {"Double Magic", "Item_FairyBless_MPDouble", 0, "Upgrades"},
    {"Double Defense", "Item_FairyBless_DefenseDouble", 0, "Upgrades"},
    {"Kokiri's Emerald", "Item_Stone_Forest", 0, "Quest"}, {"Goron's Ruby", "Item_Stone_Fire", 0, "Quest"},
    {"Zora's Sapphire", "Item_Stone_Water", 0, "Quest"}, {"Forest Medallion", "Item_Medal_Forest", 0, "Quest"},
    {"Fire Medallion", "Item_Medal_Fire", 0, "Quest"}, {"Water Medallion", "Item_Medal_Water", 0, "Quest"},
    {"Shadow Medallion", "Item_Medal_Shadow", 0, "Quest"}, {"Spirit Medallion", "Item_Medal_Spirit", 0, "Quest"},
    {"Light Medallion", "Item_Medal_Light", 0, "Quest"},
    {"Zelda's Lullaby", "Item_Melody_ZeldaLullaby", 0, "Songs"}, {"Epona's Song", "Item_Melody_EponaSong", 0, "Songs"},
    {"Saria's Song", "Item_Melody_SariaSong", 0, "Songs"}, {"Sun's Song", "Item_Melody_SunSong", 0, "Songs"},
    {"Song of Time", "Item_Melody_TimeSong", 0, "Songs"}, {"Song of Storms", "Item_Melody_StormSong", 0, "Songs"},
    {"Minuet of Forest", "Item_Melody_ForestMinuet", 0, "Songs"}, {"Bolero of Fire", "Item_Melody_FireBolero", 0, "Songs"},
    {"Serenade of Water", "Item_Melody_WaterSerenade", 0, "Songs"}, {"Nocturne of Shadow", "Item_Melody_ShadowNocturne", 0, "Songs"},
    {"Requiem of Spirit", "Item_Melody_SpiritRequiem", 0, "Songs"}, {"Prelude of Light", "Item_Melody_LightPrelude", 0, "Songs"},
    {"Stone of Agony", "Item_AgonyStone", 0, "Quest"}, {"Gerudo's Card", "Item_Token_Gerudo", 0, "Quest"},
    {"Gold Skulltula Tokens", "Item_Token_Gold", 1, "Quest"},
};
std::atomic<int> counts[128];
std::atomic<int> age{-1};
std::mutex text_mutex; std::string map_now, message;
void Say(const char* format, ...) __attribute__((format(printf, 1, 2)));
void Say(const char* format, ...) {
    char line[256]; va_list args; va_start(args, format); std::vsnprintf(line, sizeof line, format, args); va_end(args);
    RT_LOG("tools: %s", line);
    std::lock_guard lock{text_mutex}; message = line;
}

// Learned songs are saved IsGet.<actor> bools, not pouch quantities (their names' hashes).
u32 SongFlag(const char* name) {
    static const struct { const char* name; u32 hash; } songs[] = {
        {"Item_Melody_ZeldaLullaby", 0x01c20002u}, {"Item_Melody_EponaSong", 0xb492ac33u}, {"Item_Melody_SariaSong", 0xd3f8bfccu},
        {"Item_Melody_SunSong", 0x165dd1adu}, {"Item_Melody_TimeSong", 0x03b3450eu}, {"Item_Melody_StormSong", 0x2b39dd4fu},
        {"Item_Melody_ForestMinuet", 0xdc0a0b54u}, {"Item_Melody_FireBolero", 0x49041ab9u}, {"Item_Melody_WaterSerenade", 0x3fa63986u},
        {"Item_Melody_ShadowNocturne", 0xc2fb95f4u}, {"Item_Melody_SpiritRequiem", 0x802a36a8u}, {"Item_Melody_LightPrelude", 0xd6320ba4u},
    };
    for (const auto& s : songs) if (!std::strcmp(name, s.name)) return s.hash;
    return 0;
}
// QueryGetPouchContentNumByActorName: f_805b63e0(pouch manager, &name, 0, 0) counts one content.
int PouchCount(Cpu* c, const char* name) {
    const u64 manager = Load<u64>(G(0x84e98e28));
    if (!Readable(manager) || std::strlen(name) >= 0x60) return -1;
    Call call{c};
    const u64 frame = call.Stack(0x80);
    std::memcpy(reinterpret_cast<void*>(frame + 0x10), name, std::strlen(name) + 1);
    Store<u64>(frame, frame + 0x10);
    c->x[0] = manager; c->x[1] = frame; c->x[2] = 0; c->x[3] = 0;
    call.Run(G(0x805b63e0));
    return (int)(s32)c->x[0];
}
// OneShotAddOrRemovePouchContent (f_828d4a70) for one content: the event's status, or -1 when the pouch is not ready.
int PouchChange(Cpu* c, const char* name, int delta) {
    const u64 pouch = Load<u64>(G(0x84c9e0d0)), state = Load<u64>(G(0x84c9e2a8));
    if (!Readable(pouch) || !Load<u64>(pouch) || !Readable(state) || !Load<u64>(state) || !Load<u64>(G(0x84eb6888)) ||
        !Load<u64>(G(0x84e8c060)) || !delta || std::strlen(name) >= 0x60) return -1;
    Call call{c};
    const u64 event = call.Stack(0xe0);
    std::memcpy(reinterpret_cast<void*>(event + 0x80), name, std::strlen(name) + 1);
    Store<u64>(event + 0x20, event + 0x60); Store<u64>(event + 0x30, event + 0x70); Store<u64>(event + 0x38, event + 0x68);
    Store<u64>(event + 0x48, event + 0x74); Store<u64>(event + 0x50, event + 0x74); Store<u64>(event + 0x60, event + 0x80);
    static const char script[] = "Script"; Store<u64>(event + 0x68, reinterpret_cast<u64>(script)); // (the game's own string there; any copy of it does)
    Store<u32>(event + 0x70, (u32)delta);
    c->x[0] = event;
    call.Run(G(0x828d4a70));
    return (int)(Load<u32>(event + 0x18) >> 30);
}
// Masks share one slot: the game takes "Item_Mask_X" when adding one and may keep it as "Armor_Child_Mask_X".
bool MaskAlias(const char* name, std::string* alias) {
    if (std::strncmp(name, "Armor_Child_Mask_", 17)) return false;
    *alias = std::string("Item_Mask_") + (name + 17); return true;
}
int CountOf(Cpu* c, const char* name) {
    if (const u32 song = SongFlag(name)) { bool learned = false; return DataBool(c, song, &learned) ? learned : -1; }
    const int n = PouchCount(c, name);
    std::string alias;
    if (!MaskAlias(name, &alias)) return n;
    return std::max(n, PouchCount(c, alias.c_str()));
}
void EditItem(Cpu* c, int index, int asked) {
    if (index < 0 || index >= (int)items.size()) return;
    const char* name = items[index].name;
    const int have = CountOf(c, name);
    if (have < 0) { Say("%s: no save is loaded", items[index].label); return; }
    if (const u32 song = SongFlag(name)) {
        if (DataSetBool(song, !have)) Say("%s %s", items[index].label, have ? "forgotten" : "learned");
        return;
    }
    std::string alias; const bool mask = MaskAlias(name, &alias);
    int delta = asked ? asked : have > 0 ? -have : 1;
    if (have + delta < 0) delta = -have;
    if (!delta) return;
    int status = -1;
    if (mask && delta > 0) { // (the mask in the slot goes back first)
        for (const Item& other : items) {
            std::string other_alias;
            if (&other == &items[index] || !MaskAlias(other.name, &other_alias)) continue;
            for (const char* spelling : {other.name, other_alias.c_str()}) if (const int held = PouchCount(c, spelling); held > 0) PouchChange(c, spelling, -held);
        }
        delta = 1;
    }
    if (mask && delta < 0) { for (const char* spelling : {name, alias.c_str()}) if (const int held = PouchCount(c, spelling); held > 0) status = PouchChange(c, spelling, -held); }
    else status = PouchChange(c, mask ? alias.c_str() : name, delta);
    // (the game takes the change at its own pace: the window's count follows within half a second. Its limits apply:
    // a full bag takes no more, bombs need a bomb bag)
    if (status < 0) Say("%s: the inventory is not ready", items[index].label);
    else if (items[index].kind) Say("%s: %+d asked for (had %d)", items[index].label, delta, have);
    else Say("%s %s", items[index].label, delta > 0 ? "given" : "taken away");
}

// ---- the time of day --------------------------------------------------------------------------------------------------------
void SetTime(Cpu* c, int hour) { // (ExecuteWmSetTime: seconds since midnight to the world manager)
    const u64 world = Singleton(G(0x84c9e898));
    if (!Readable(world) || !Load<u64>(world + 0x510)) { Say("the world's clock is not there (no save loaded?)"); return; }
    { Call call{c}; c->x[0] = world; c->x[1] = (u64)hour * 3600; call.Run(G(0x834722d0)); }
    Say("the time is %02d:00", hour);
}

// ---- teleport -----------------------------------------------------------------------------------------------------------------
// Map changes run through an event: an entrance requests Dm_SY_BancChangeGo with NextBanc / NextStartPos / BootEventName,
// the event reserves the map change and the sequence loads the map when the event ends. This builds the same request.
struct Destination { const char* label; const char* child_banc, *child_start, *child_boot; const char* adult_banc, *adult_start, *adult_boot; };
const Destination destinations[] = {
    {"Kokiri Forest", "KokiriForest_Child", "LostWoodBridge_FrontOfHyruleField", "Dm_SY_BancChangeArrive", "KokiriForest_Adult", "LostWoodBridge_FrontOfHyruleField", "Dm_SY_BancChangeArrive"},
    {"Hyrule Field", "HyruleField_Child", "HyruleField_FrontOfLostWoodBridge", "Dm_SY_BancChangeArrive", "HyruleField_Adult", "HyruleField_FrontOfLostWoodBridge", "Dm_SY_BancChangeArrive"},
    {"Kakariko Village", "KakarikoVillage_Child", "KakarikoVillage_Entrance", "Dm_SY_BancChangeArrive_KakarikoVillage_01", "KakarikoVillage_Adult", "KakarikoVillage_Entrance", "Dm_SY_BancChangeArrive_KakarikoVillage_01"},
    {"Market", "Market_Child", "Market_Entrance", "Dm_SY_BancChangeArrive_Market_02", "Market_Adult", "Market_Entrance", "Dm_SY_BancChangeArrive_Market_02"},
    {"Hyrule Castle / Ganon's Castle (outside)", "HyruleCastle", "HyruleCastle_Entrance", "Dm_SY_BancChangeArrive_HyruleCastle_01", "GanonCastle", "GanonCastleEntrance", "Dm_SY_BancChangeArrive_GanonCastle_01"},
    {"Lost Woods", "LostWood_Child", "LostWood_FrontOfSacredForestMeadow", "Dm_SY_BancChangeArrive_LostWood", "LostWood_Adult", "LostWood_FrontOfSacredForestMeadow", "Dm_SY_BancChangeArrive_LostWood"},
    {"Sacred Forest Meadow", "SacredForestMeadow_Child", "OutsideOfForestTemple", "Dm_SY_BancChangeArrive_SacredForestMeadow_01", "SacredForestMeadow_Adult", "OutsideOfForestTemple", "Dm_SY_BancChangeArrive_SacredForestMeadow_01"},
    {"Lake Hylia", "HyruleField_Child", "LakeHylia_FromGerudoValley", "Dm_SY_BancChangeArrive_LakeHylia_02", "HyruleField_Adult", "LakeHylia_FromGerudoValley", "Dm_SY_BancChangeArrive_LakeHylia_02"},
    {"Graveyard", "Graveyard_Child", "GraveYard_KakarikoSide", "Dm_SY_BancChangeArrive_Graveyard_01", "Graveyard_Adult", "GraveYard_KakarikoSide", "Dm_SY_BancChangeArrive_Graveyard_01"},
    {"Death Mountain Trail", "DeathMountainTrail_Child", "MountainTrail_KakarikoSide", "Dm_SY_BancChangeArrive_DeathMountainTrail_01", "DeathMountainTrail_Adult", "MountainTrail_KakarikoSide", "Dm_SY_BancChangeArrive_DeathMountainTrail_01"},
    {"Death Mountain Crater", "DeathMountainCrater_Child", "Crater_GoronCitySide", "DeathMountainCrater_Arrive", "DeathMountainCrater_Adult", "Crater_GoronCitySide", "DeathMountainCrater_Arrive"},
    {"Goron City", "GoronCity_Child", "GoronCity_MountainTrailSide", "Dm_SY_BancChangeArrive_GoronCity_01", "GoronCity_Adult", "GoronCity_MountainTrailSide", "Dm_SY_BancChangeArrive_GoronCity_01"},
    {"Zora's River", "ZoraRiver_Child", "ZoraRiver_Entrance", "Dm_SY_BancChangeArrive", "ZoraRiver_Adult", "ZoraRiver_Entrance", "Dm_SY_BancChangeArrive"},
    {"Zora's Domain", "ZoraDomain_Child", "ZoraDomain_FromZoraRiver", "Dm_SY_BancChangeArrive", "ZoraDomain_Adult", "ZoraDomain_FromZoraRiver", "Dm_SY_BancChangeArrive"},
    {"Zora's Fountain", "ZoraFountain_Child", "ZoraFountain_FromZoraDomain", "Dm_SY_BancChangeArrive", "ZoraFountain_Adult", "ZoraFountain_FromZoraDomain", "Dm_SY_BancChangeArrive"},
    {"Gerudo Valley", "GerudoValley_Child", "GerudoValley_Entrance", "FromHyruleField_ArriveGerudoValley", "GerudoValley_Adult", "GerudoValley_Entrance", "FromHyruleField_ArriveGerudoValley"},
    {"Gerudo Fortress (adult)", nullptr, nullptr, nullptr, "GerudoFortress_Adult", "GerudoFortress_FromHountedWasteland", "Dm_SY_BancChangeArrive_GerudoFortress_02"},
    {"Haunted Wasteland", "HountedWasteland_Child", "HountedWasteland_FromGerudoFortress", "FromGerudoFortress_ArriveWateland", "HountedWasteland_Adult", "HountedWasteland_FromGerudoFortress", "FromGerudoFortress_ArriveWateland"},
    {"Desert Colossus", "DesertColossus_Child", "DesertColossus_FromHountedWasteland", "Dm_SY_BancChangeArrive_DesertColossus_01", "DesertColossus_Adult", "DesertColossus_FromHountedWasteland", "Dm_SY_BancChangeArrive_DesertColossus_01"},
    {"Inside the Deku Tree", "InsideTheDekuTree", "Entrance", "Dm_SY_BancChangeArrive_InsideTheDekuTree_01", "InsideTheDekuTree", "Entrance", "Dm_SY_BancChangeArrive_InsideTheDekuTree_01"},
    {"Dodongo's Cavern", "DodongoCavern", "DodongoCavern_MountainTrailSide", "Dm_SY_BancChangeArrive", "DodongoCavern", "DodongoCavern_MountainTrailSide", "Dm_SY_BancChangeArrive"},
    {"Jabu-Jabu's Belly", "InsideJabuJabuBelly", "InsideJabuJabuBelly_FromZoraFountain", "Dm_SY_BancChangeArrive", "InsideJabuJabuBelly", "InsideJabuJabuBelly_FromZoraFountain", "Dm_SY_BancChangeArrive"},
    {"Forest Temple", "ForestTemple", "ForestTemple", "Dm_SY_BancChangeArrive", "ForestTemple", "ForestTemple", "Dm_SY_BancChangeArrive"},
    {"Fire Temple", "FireTemple", "FireTemple", "Dm_SY_BancChangeArrive", "FireTemple", "FireTemple", "Dm_SY_BancChangeArrive"},
    {"Water Temple", "WaterTemple", "WaterTemple_FromLakeHylia", "SageOfWater_Arrival_WaterTemple", "WaterTemple", "WaterTemple_FromLakeHylia", "SageOfWater_Arrival_WaterTemple"},
    {"Shadow Temple", "ShadowTemple", "Entrance", "ShadowTemple_Arrive", "ShadowTemple", "Entrance", "ShadowTemple_Arrive"},
    {"Spirit Temple", "SpiritTemple", "SpiritTemple_FromRightHand", "Dm_SY_BancChangeArrive", "SpiritTemple", "SpiritTemple_FromRightHand", "Dm_SY_BancChangeArrive"},
    {"Bottom of the Well", "BottomOfTheWell", "Well_KakarikoSide", "Dm_SY_BancChangeArrive", "BottomOfTheWell", "Well_KakarikoSide", "Dm_SY_BancChangeArrive"},
    {"Ice Cavern", "IceCavern", "IceCavern_FromZoraFountain", "SageOfWater_Arrival_IceCavern", "IceCavern", "IceCavern_FromZoraFountain", "SageOfWater_Arrival_IceCavern"},
    {"Gerudo Training Ground", "GerudoTrainingGround", "GerudoTrainingGround_FromGerudoFortress", "FromGerudoFortress_ArriveTrainingGround", "GerudoTrainingGround", "GerudoTrainingGround_FromGerudoFortress", "FromGerudoFortress_ArriveTrainingGround"},
    {"Inside Ganon's Castle", "InsideGanonCastle", "Entrance", "Dm_SY_BancChangeArrive_InsideGanonCastle_01", "InsideGanonCastle", "Entrance", "Dm_SY_BancChangeArrive_InsideGanonCastle_01"},
};
constexpr int DestinationCount = sizeof destinations / sizeof *destinations;

// One argument of an EventRequestArg: 200-byte entries at arg+8 (capacity arg+0, count arg+16); an entry: +0 its type
// (1 a string pointer, 4 a bool), +8 its name buffer, +16 that buffer's capacity, +152 the value.
bool EventArg(u64 arg, const char* name, u32 type, u64 value) {
    const u32 capacity = Load<u32>(arg), count = Load<u32>(arg + 16);
    if (count >= capacity) return false;
    const u64 entry = Load<u64>(arg + 8) + (u64)count * 200;
    const u64 name_buf = Load<u64>(entry + 8); const u32 name_cap = Load<u32>(entry + 16);
    const size_t length = std::strlen(name);
    if (!Readable(name_buf) || length + 1 > name_cap) return false;
    std::memcpy(reinterpret_cast<void*>(name_buf), name, length + 1);
    if (type == 4) Store<u8>(entry + 152, value & 1); else Store<u64>(entry + 152, value);
    Store<u32>(entry, type); Store<u32>(arg + 16, count + 1);
    return true;
}
s64 teleported_at = 0;
bool Teleport(Cpu* c, u64 actor, int index) {
    if (index < 0 || index >= DestinationCount) return false;
    const s64 now = NowNs();
    if (teleported_at && now - teleported_at < 8'000'000'000ll) { Say("another map change is still on its way (one every 8 seconds)"); return false; }
    const u64 events = Singleton(EventMgrSlot);
    if (!Readable(events)) { Say("the game's event manager is not there yet"); return false; }
    if (Load<u64>(events + 104)) { Say("an event is playing (a cutscene, a map change): try again when it has ended"); return false; }
    bool loading = false, child = true;
    if (DataBool(c, IsLoadingBanc, &loading) && loading) { Say("a map is loading"); return false; }
    DataBool(c, IsChildhood, &child);
    const Destination& d = destinations[index];
    const char* banc = child ? d.child_banc : d.adult_banc; const char* start = child ? d.child_start : d.adult_start; const char* boot = child ? d.child_boot : d.adult_boot;
    if (!banc) { Say("%s has no map for %s", d.label, child ? "the child" : "the adult"); return false; }
    // (the event reads its string arguments after the request returns: they stay here)
    alignas(16) static char strings[512];
    std::memset(strings, 0, sizeof strings);
    std::snprintf(strings, 128, "%s", banc); std::snprintf(strings + 128, 128, "%s", start); std::snprintf(strings + 256, 128, "%s", boot);
    const u64 text = reinterpret_cast<u64>(strings);
    bool ok = true; u64 result = 0;
    {
        Call call{c};
        const u64 arg = call.Stack(0x1b20 + 0x100);
        c->x[0] = arg; call.Run(G(0x80d99f10)); // (the request's constructor)
        {   // the event's name: arg+6800 points at its buffer, arg+6808 its capacity
            const char* event = "Dm_SY_BancChangeGo";
            const u64 buf = Load<u64>(arg + 6800); const u32 cap = Load<u32>(arg + 6808);
            if (!Readable(buf) || std::strlen(event) + 1 > cap) ok = false; else std::memcpy(reinterpret_cast<void*>(buf), event, std::strlen(event) + 1);
        }
        ok = ok && EventArg(arg, "NextBanc", 1, text) && EventArg(arg, "NextStartPos", 1, text + 128) && EventArg(arg, "BootEventName", 1, text + 256) && EventArg(arg, "IsCheckAir", 4, 0);
        if (ok) { c->sp = arg; c->x[0] = arg + 0x1950; c->x[1] = actor; c->x[2] = 0; call.Run(G(0x802dc440)); } // (the starter actor's link)
        if (ok) { c->sp = arg; c->x[0] = events; c->x[1] = arg; call.Run(G(0x80d9a820)); result = c->x[0]; }
        for (const u64 link : {arg + 0x1968, arg + 0x1950}) { // (the request's own cleanup of its two actor links)
            Store<u64>(link, G(0x84994b20));
            c->sp = arg; c->x[0] = link; call.Run(G(0x8022b6f0));
        }
    }
    // (an area that hides the minimap leaves its saved flag set when a teleport skips the area's exit: put back
    // through the game's own setter, which changes the live value and the save's)
    bool hidden = false;
    if (ok && result && DataBool(c, MapIconsOff, &hidden) && hidden) {
        const u64 mgr = Singleton(GameDataSlot);
        if (Readable(mgr)) { Call call{c}; c->x[0] = mgr; c->x[1] = 0; c->x[2] = MapIconsOff; call.Run(G(0x806a2b20)); }
    }
    if (!ok) { Say("the map change could not be asked for"); return false; }
    teleported_at = now;
    Say("going to %s%s", d.label, result ? "" : " (the game has not taken the request)");
    return true;
}
void SwitchAge(Cpu* c, u64 actor) {
    bool child = true;
    if (!DataBool(c, IsChildhood, &child)) { Say("Link's age is not in the game's data (no save loaded?)"); return; }
    std::string current; DataString(c, CurrentBanc, &current);
    int target = 3; // (the Market: the Temple of Time is in it)
    for (int i = 0; i < DestinationCount; ++i) {
        const Destination& d = destinations[i];
        const char* other = child ? d.adult_banc : d.child_banc;
        if (other && ((d.child_banc && current == d.child_banc) || (d.adult_banc && current == d.adult_banc))) { target = i; break; }
    }
    if (!DataSetBool(IsChildhood, !child)) { Say("Link's age could not be written"); return; }
    if (!Teleport(c, actor, target)) { DataSetBool(IsChildhood, child); return; }
    Say("Link becomes %s: loading %s", child ? "the adult" : "the child", destinations[target].label);
}

// ---- requests from the settings window ------------------------------------------------------------------------------------
enum { None, DoTeleport, DoTime, DoAge, DoItem, DoMark, DoReturn };
std::mutex ask_mutex; struct Ask { int what = None, a = 0, b = 0; }; std::vector<Ask> asks;
void Push(int what, int a = 0, int b = 0) { std::lock_guard lock{ask_mutex}; if (asks.size() < 16) asks.push_back({what, a, b}); }
std::atomic<s64> player_seen{0};

// ---- Link's controller, flight -------------------------------------------------------------------------------------------
// (vtable 0x84c54f10, the one OneShotPhysicsCctChangePosition moves, from his components)
u64 PlayerController(u64 components) {
    u64 at = components;
    for (const unsigned offset : {80u, 32u, 16u, 8u}) { if (!Readable(at)) return 0; at = Load<u64>(at + offset); }
    // (its class by what its table of functions holds at +0xe8, the change of position: the table's own address is in
    // no instruction, so tools/signatures.py could not find it in another game version)
    const u64 table = Readable(at) ? Load<u64>(at) : 0;
    return Readable(table) && G(0x806d2530) && Load<u64>(table + 0xe8) == G(0x806d2530) ? at : 0;
}
std::atomic<u64> speed_controller{0};
thread_local u64 flight_controller = 0, flight_actor = 0; thread_local float flight_yaw = 0;
u64 GameplayView() {
    const u64 manager = Singleton(G(0x84ca6bc0));
    const u64 cameras = Readable(manager) && Load<u32>(manager + 32) ? Load<u64>(manager + 40) : 0;
    const u64 camera = Readable(cameras) ? Load<u64>(cameras) : 0;
    return Readable(camera) ? Load<u64>(camera + 120) : 0;
}
#define LookAtCamera G(0x84c18a88)
void WriteRotation(u64 matrix, float yaw) {
    const float s = std::sin(yaw), co = std::cos(yaw);
    const float rotation[3][3] = {{co, 0, s}, {0, 1, 0}, {-s, 0, co}};
    for (unsigned row = 0; row < 3; ++row) for (unsigned col = 0; col < 3; ++col) Store<float>(matrix + row * 16 + col * 4, rotation[row][col]);
}

void Update(Cpu* c, u64 actor, u64 components) {
    static u64 flying_actor = 0, marked_actor = 0, marked_controller = 0;
    static s64 last_tick = 0, last_player_tick = 0, counted_at = 0;
    static float target[3], velocity[3], mark[3], yaw;
    const s64 now = NowNs();
    player_seen.store(now, std::memory_order_relaxed);
    if (marked_actor != actor || now - last_player_tick > 2'000'000'000ll) marked_controller = 0;
    last_player_tick = now;
    bool flying = flight && !EventPlaying(); // (an event that takes Link waits for him to stand: he is let go meanwhile)
    if (!flying) flying_actor = 0;

    // (what the window shows, twice a second)
    if (now - counted_at > 500'000'000ll) {
        counted_at = now;
        for (size_t i = 0; i < items.size() && i < 128; ++i) counts[i].store(CountOf(c, items[i].name), std::memory_order_relaxed);
        if (static const bool log = std::getenv("OOT_TOOLS_LOG") != nullptr; log) {
            std::string line; for (size_t i = 0; i < items.size(); ++i) if (const int n = counts[i].load(); n) line += std::string(" ") + items[i].label + "=" + std::to_string(n);
            static std::string last; if (line != last) { RT_LOG("tools: inventory:%s", line.c_str()); last = line; }
        }
        bool child = true; const bool loaded = counts[0].load() >= 0;
        age.store(loaded && DataBool(c, IsChildhood, &child) ? (child ? 0 : 1) : -1, std::memory_order_relaxed);
        std::string map; DataString(c, CurrentBanc, &map);
        { std::lock_guard lock{text_mutex}; if (map != map_now) { RT_LOG("map: %s (%s)", map.c_str(), child ? "child" : "adult"); map_now = map; } }
    }
    speed_controller.store(gfx::Current().move_speed > 100 ? PlayerController(components) : 0, std::memory_order_relaxed);

    {   // OOT_TOOLS_ASK="<seconds>:<what>=<a>[,<b>];..." (development: requests without the window; what: teleport, time,
        // age, item, mark, return, flight, camera)
        static const char* script = std::getenv("OOT_TOOLS_ASK"); static s64 began = 0; static size_t at = 0;
        if (script && script[at]) {
            if (!began) began = now;
            int seconds = 0, a = 0, b = 0, used = 0; char what[16] = "";
            if (std::sscanf(script + at, "%d:%15[a-z]=%d%n", &seconds, what, &a, &used) >= 3) {
                if (now - began >= seconds * 1'000'000'000ll) {
                    at += used; if (script[at] == ',') { b = std::atoi(script + at + 1); while (script[at] && script[at] != ';') ++at; }
                    if (script[at] == ';') ++at;
                    const std::string w = what;
                    if (w == "teleport") Push(DoTeleport, a); else if (w == "time") Push(DoTime, a); else if (w == "age") Push(DoAge);
                    else if (w == "item") Push(DoItem, a, b); else if (w == "mark") Push(DoMark); else if (w == "return") Push(DoReturn);
                    else if (w == "flight") flight = a; else if (w == "camera") free_camera = a;
                }
            } else at = std::strlen(script);
        }
    }
    std::vector<Ask> todo; { std::lock_guard lock{ask_mutex}; todo.swap(asks); }
    bool do_mark = false, do_return = false;
    for (const Ask& ask : todo) switch (ask.what) {
        case DoTeleport: Teleport(c, actor, ask.a); break;
        case DoTime: SetTime(c, ask.a); break;
        case DoAge: SwitchAge(c, actor); break;
        case DoItem: EditItem(c, ask.a, ask.b); counted_at = 0; break;
        case DoMark: do_mark = true; break;
        case DoReturn: do_return = true; break;
    }
    if (!flying && !do_mark && !do_return) return;

    const u64 controller = PlayerController(components);
    if (!controller) return;
    if (do_mark) {
        for (unsigned i = 0; i < 3; ++i) mark[i] = Load<float>(actor + 796 + i * 4);
        marked_actor = actor; marked_controller = controller;
        Say("place marked (%.1f, %.1f, %.1f)", mark[0], mark[1], mark[2]);
    }
    const bool returning = do_return && marked_controller == controller;
    if (do_return && !returning) Say("no place is marked in this map");
    if (!flying && !returning) return;
    const s64 elapsed = now - last_tick;
    if (flying_actor != actor) yaw = std::atan2(Load<float>(actor + 816), Load<float>(actor + 840));
    const auto moved_away = [&] { // (the game moved Link since the flight last placed him: a respawn, a warp)
        float apart = 0; for (unsigned i = 0; i < 3; ++i) { const float d = Load<float>(actor + 796 + i * 4) - target[i]; apart += d * d; }
        return apart > 4.0f;
    };
    if (returning) {
        std::memcpy(target, mark, sizeof target); std::memset(velocity, 0, sizeof velocity);
        flying_actor = flying ? actor : 0;
    } else if (flying_actor != actor || elapsed < 0 || elapsed > 250'000'000 || moved_away()) {
        for (unsigned i = 0; i < 3; ++i) target[i] = Load<float>(actor + 796 + i * 4);
        std::memset(velocity, 0, sizeof velocity);
        flying_actor = actor; last_tick = now;
        return;
    } else if (elapsed == 0) return;
    last_tick = now;

    if (!returning) {
        const float dt = std::min((float)elapsed * 1e-9f, 0.05f);
        float desired[3] = {0, 0, 0};
        if (!free_camera) { // (the free camera has the sticks while it is on)
            const float right = input::pad_now[0].load() / 32767.0f, forward = input::pad_now[1].load() / 32767.0f;
            const u64 buttons = input::pad_buttons.load();
            const int vertical = ((buttons >> input::Up) & 1) - ((buttons >> input::Down) & 1);
            const u64 view = GameplayView();
            float rx = 1, rz = 0;
            if (Readable(view) && Load<u64>(view) == LookAtCamera) {
                const float x = Load<float>(view + 8), z = Load<float>(view + 16), length = std::hypot(x, z);
                if (length > 0.01f) { rx = x / length; rz = z / length; }
            }
            desired[0] = rx * right + rz * forward; desired[1] = (float)vertical; desired[2] = rz * right - rx * forward;
        }
        const float length = std::sqrt(desired[0] * desired[0] + desired[1] * desired[1] + desired[2] * desired[2]);
        const bool fast = (input::pad_buttons.load() >> input::B) & 1;
        const float speed = (fast ? 12.0f : 4.0f) / std::max(1.0f, length), blend = 1.0f - std::exp(-12.0f * dt);
        for (unsigned i = 0; i < 3; ++i) { velocity[i] += (desired[i] * speed - velocity[i]) * blend; target[i] += velocity[i] * dt; }
        if (std::hypot(desired[0], desired[2]) > 0.08f) {
            const float delta = std::remainder(std::atan2(desired[0], desired[2]) - yaw, 6.28318531f);
            yaw += std::max(-8.0f * dt, std::min(delta, 8.0f * dt));
        }
    }
    {
        Call call{c};
        const u64 arg = call.Stack(16);
        for (unsigned i = 0; i < 3; ++i) Store<float>(arg + i * 4, target[i]);
        c->x[0] = controller; c->x[1] = arg;
        flight_yaw = yaw;
        flight_actor = flying && !returning ? actor : 0;
        flight_controller = flying && !returning ? controller : 0;
        call.Run(G(0x806d2530)); // (the controller's change of position)
        flight_controller = 0;
    }
    if (returning) Say("back at the marked place");
}

bool IsPlayer(u64 actor) {
    if (!G(0x84b6c460)) return false;
    if (!Readable(actor) || Load<u64>(actor) != G(0x84b6c460)) return false;
    // (other actors share this vtable: the actor's resource record names Link)
    const u64 resource = Load<u64>(actor + 16);
    return Readable(resource) && Load<u32>(resource + 88) == 0x79616c50 && Load<u16>(resource + 92) == 0x7265 && Load<u8>(resource + 94) == 0;
}
} // namespace

void PlayerUpdate(Cpu* c, unsigned long long actor) {
    flight_actor = 0;
    if (!Ready()) return;
    if (!Readable(actor)) return; // (the site is in the movement object's own update: x19 is one; that it is Link's is checked below)
    const u64 components = Load<u64>(actor + 624);
    const u64 movement = Readable(components) ? Load<u64>(components + 800) : 0;
    if (!Readable(movement) || Load<u64>(movement + 24) != actor || !IsPlayer(movement - 0x1d50)) return;
    ::oot_note_link(components);
    // (Link can be updated by different worker threads on successive frames)
    static std::mutex one;
    std::unique_lock lock{one, std::try_to_lock};
    if (!lock.owns_lock()) return;
    static const bool off = std::getenv("OOT_NO_TOOLS") != nullptr;
    if (!off && Ready()) Update(c, actor, components);
}
void FlightPose(Cpu*, unsigned long long actor, unsigned long long matrix) {
    if (flight_actor && flight_actor == actor && Readable(matrix)) { flight_actor = 0; WriteRotation(matrix, flight_yaw); }
}
void FlightRotation(Cpu*, unsigned long long controller, unsigned long long matrix) {
    if (flight_controller && flight_controller == controller && Readable(matrix)) WriteRotation(matrix, flight_yaw);
}

// The free camera: the gameplay view's matrix (view + 8, three rows of four) built from a position and angles of its
// own, moved with the sticks; Link stays where he is.
void FreeView(Cpu*, unsigned long long view) {
    if (!Ready()) return;
    if (!Readable(view) || Load<u64>(view) != LookAtCamera || GameplayView() != view) return;
    static std::mutex mutex; std::lock_guard lock{mutex};
    static u64 active_view = 0; static s64 last_tick = 0; static double eye[3], yaw, pitch;
    input::swallow_sticks.store(free_camera, std::memory_order_relaxed);
    if (!free_camera) { active_view = 0; return; }
    const s64 now = NowNs(), elapsed = now - last_tick;
    if (active_view != view || elapsed > 500'000'000) {
        double target[3];
        std::memcpy(eye, reinterpret_cast<const void*>(view + 96), sizeof eye); std::memcpy(target, reinterpret_cast<const void*>(view + 120), sizeof target);
        const double x = eye[0] - target[0], y = eye[1] - target[1], z = eye[2] - target[2];
        yaw = std::atan2(x, z); pitch = std::atan2(y, std::hypot(x, z));
        active_view = view;
    }
    last_tick = now;
    const double dt = elapsed > 0 && elapsed < 100'000'000 ? (double)elapsed * 1e-9 : 0;
    const u64 buttons = input::pad_buttons.load();
    yaw -= 1.6 * dt * input::pad_now[2].load() / 32767.0;
    pitch = std::max(-1.55, std::min(1.55, pitch - 1.2 * dt * input::pad_now[3].load() / 32767.0));
    const double sy = std::sin(yaw), cy = std::cos(yaw), sp = std::sin(pitch), cp = std::cos(pitch);
    const double axes[3][3] = {{cy, 0, -sy}, {-sp * sy, cp, -sp * cy}, {cp * sy, sp, cp * cy}};
    const double step = dt * (((buttons >> input::B) & 1) ? 30.0 : 8.0);
    const double right = input::pad_now[0].load() / 32767.0, forward = input::pad_now[1].load() / 32767.0;
    eye[1] += step * (double)((int)((buttons >> input::Up) & 1) - (int)((buttons >> input::Down) & 1));
    for (int i = 0; i < 3; ++i) eye[i] += step * (right * axes[0][i] - forward * axes[2][i]);
    for (int row = 0; row < 3; ++row) {
        const double* axis = axes[row];
        const float values[4] = {(float)axis[0], (float)axis[1], (float)axis[2], (float)-(axis[0] * eye[0] + axis[1] * eye[1] + axis[2] * eye[2])};
        for (int col = 0; col < 4; ++col) Store<float>(view + 8 + (row * 4 + col) * 4, values[col]);
    }
}

// A character controller's forward speed being set (s0: metres a second, or a frame): Link's is scaled by the Movement
// Speed row while the player steers him. Only speeds up to 15 m/s (his run is 4.5): a ledge climb sets some 55 for a
// frame, and scaled it threw him past the ledge.
void Speed(Cpu* c, unsigned long long controller, bool per_second) {
    if (!controller || controller != speed_controller.load(std::memory_order_relaxed)) return;
    const int percent = gfx::Current().move_speed;
    if (percent <= 100 || (!input::pad_now[0].load() && !input::pad_now[1].load())) return;
    const float speed = LowFloat(c->v[0]);
    if (!(std::fabs(speed) * (per_second ? 1.0f : 60.0f) <= 15.0f)) return;
    SetLowFloat(c->v[0], speed * (percent / 100.0f));
}

// Link's life being written (quarter hearts; w11 before, w12 after, at x13; x19 the actor). The Damage Taken row scales
// what a hit takes (half rounds up: a quarter-heart hit still hurts); healing is never scaled. At the first site w9 is
// the whole hit, before the game clamped the result to zero.
void Life(Cpu* c, bool damage_site) {
    const u32 old = (u32)c->x[11], next = (u32)c->x[12];
    if (next >= old) return;
    const int setting = gfx::Current().damage_taken; const bool keep = setting == 4 || gfx::Current().cheat_health; // (Damage Taken at None, or Infinite Health)
    {static const bool log = std::getenv("OOT_LIFE_LOG") != nullptr; static int shown = 0; // (diagnostic: every drop of a life value written here)
     if (log && shown++ < 40) RT_LOG("life write: %u -> %u (hit %u, player %d, setting %d)", old, next, (u32)c->x[9], (int)(IsPlayer(c->x[19]) && c->x[13] == Load<u64>(c->x[19] + 7184) + 8), setting);}
    if (setting == 1 && !keep) return;
    if (!IsPlayer(c->x[19]) || c->x[13] != Load<u64>(c->x[19] + 7184) + 8) return;
    if (keep) { c->x[12] = old; return; }
    u64 damage = damage_site ? (u32)c->x[9] : old - next;
    if (setting == 0) damage = (damage + 1) / 2; else if (setting == 2) damage *= 2; else damage *= 4;
    c->x[12] = damage >= old ? 0 : old - (u32)damage;
}

// A save request reaching the save manager (the reason in w1, the mode in w2). With Autosave off the ones a map
// change makes are dropped (the request made at the crossing and the two an arrival can make): the caller is the
// autosave's own function, and its caller says which. Saving from the menu is another path.
bool SaveRequest(Cpu* c) {
    if (gfx::Current().autosave || !G(0x80bc7764)) return false;
    if (c->x[30] != G(0x80bc7764)) return false;
    const u64 parent = Readable(c->sp) ? Load<u64>(c->sp + 24) : 0;
    const u32 reason = (u32)c->x[1];
    const bool transition = (parent == G(0x80bc7710) && reason == 6) || (parent == G(0x80bc7864) && reason == 2) || (parent == G(0x832a9764) && reason == 1);
    if (!transition || (u32)c->x[2] != 0) return false;
    static int told = 0; if (told++ < 3) RT_LOG("autosave: a map change's save is left out (the Autosave row is off)");
    c->x[0] = 0;
    return true;
}

const std::vector<Item>& Items() { return items; }
int ItemCount(int index) { return index >= 0 && index < 128 ? counts[index].load(std::memory_order_relaxed) : -1; }
int Age() { return InGame() ? age.load(std::memory_order_relaxed) : -1; }
std::string Map() { std::lock_guard lock{text_mutex}; return map_now; }
std::string LastMessage() { std::lock_guard lock{text_mutex}; return message; }
bool Available() { return Ready(); }
bool InGame() { const s64 seen = player_seen.load(std::memory_order_relaxed); return seen && NowNs() - seen < 250'000'000ll; }
const std::vector<const char*>& Destinations() {
    static const std::vector<const char*> names = [] { std::vector<const char*> v; for (const Destination& d : destinations) v.push_back(d.label); return v; }();
    return names;
}
void AskTeleport(int destination) { Push(DoTeleport, destination); }
void AskTime(int hour) { Push(DoTime, hour); }
void AskAge() { Push(DoAge); }
void AskItem(int index, int delta) { Push(DoItem, index, delta); }
void AskMark() { Push(DoMark); }
void AskReturn() { Push(DoReturn); }

} // namespace tools
