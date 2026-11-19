// Achievements, read from the game's own save files.
//
// The game keeps its progress as flags named by text and stored by the 32-bit MurmurHash3 of that name
// (progress.sav: a table of (hash, value) pairs after a 0x20-byte header). Most of what a player would call a
// milestone has a flag of its own: "HavePlayedGetEvent.<item>" is set once the item's get-scene has played (every
// item, song, medallion, stone, mask and sword has one), "Gohma_DefeatBoss", "IsGameClear". caption.sav, the
// save's summary, holds the Gold Skulltula tokens. When the game commits its save data the slots are read again
// and whatever is newly reached is unlocked, with a pop-up. Unlocked achievements are kept in
// <data>/achievements.ini (one line each: id=unix time), across all save slots.
#include "achievements.h"
#include "../native.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

std::string SettingsDirectory(); // fs.cpp: the data root

namespace achievements {
namespace {
namespace sfs = std::filesystem;

u32 Murmur3(const char* text) {
    const u8* data = reinterpret_cast<const u8*>(text);
    const size_t n = std::strlen(text);
    u32 h = 0;
    const u32 c1 = 0xcc9e2d51, c2 = 0x1b873593;
    auto rotl = [](u32 x, int r) { return (x << r) | (x >> (32 - r)); };
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        u32 k; std::memcpy(&k, data + i, 4);
        k *= c1; k = rotl(k, 15); k *= c2;
        h ^= k; h = rotl(h, 13); h = h * 5 + 0xe6546b64;
    }
    u32 k = 0;
    switch (n & 3) {
    case 3: k ^= (u32)data[i + 2] << 16; [[fallthrough]];
    case 2: k ^= (u32)data[i + 1] << 8; [[fallthrough]];
    case 1: k ^= data[i]; k *= c1; k = rotl(k, 15); k *= c2; h ^= k;
    }
    h ^= (u32)n;
    h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;
    return h;
}

// What reaches an achievement: all of `flags` set (a name starting with '+' is "HavePlayedGetEvent." and the
// rest), or at least `tokens` Gold Skulltula tokens.
struct Def {
    const char* id;
    const char* title;
    const char* description;
    std::vector<const char*> flags;
    int tokens = 0;
};
const std::vector<Def>& Defs() {
    static const std::vector<Def> defs = {
        // the story
        {"kokiri_sword", "Ready for Adventure", "Find the Kokiri Sword.", {"+Weapon_Sword_Kokiri"}},
        {"fairy_ocarina", "A Gift from Saria", "Receive the Fairy Ocarina.", {"+Item_Ocarina_Fairy"}},
        {"gohma", "The Great Deku Tree's Curse", "Defeat the monster inside the Great Deku Tree.", {"Gohma_DefeatBoss"}},
        {"stone_forest", "Kokiri's Emerald", "Obtain the Spiritual Stone of the Forest.", {"+Item_Stone_Forest"}},
        {"zelda_letter", "Royal Correspondence", "Receive Zelda's Letter.", {"+Item_ZeldaLetter"}},
        {"stone_fire", "Goron's Ruby", "Obtain the Spiritual Stone of Fire.", {"+Item_Stone_Fire"}},
        {"stone_water", "Zora's Sapphire", "Obtain the Spiritual Stone of Water.", {"+Item_Stone_Water"}},
        {"ocarina_time", "The Ocarina of Time", "Receive the Ocarina of Time.", {"+Item_Ocarina_Time"}},
        {"master_sword", "The Blade of Evil's Bane", "Draw the Master Sword.", {"+Weapon_Sword_Master"}},
        {"medal_light", "Sage of Light", "Receive the Light Medallion.", {"+Item_Medal_Light"}},
        {"medal_forest", "Sage of Forest", "Receive the Forest Medallion.", {"+Item_Medal_Forest"}},
        {"medal_fire", "Sage of Fire", "Receive the Fire Medallion.", {"+Item_Medal_Fire"}},
        {"medal_water", "Sage of Water", "Receive the Water Medallion.", {"+Item_Medal_Water"}},
        {"medal_shadow", "Sage of Shadow", "Receive the Shadow Medallion.", {"+Item_Medal_Shadow"}},
        {"medal_spirit", "Sage of Spirit", "Receive the Spirit Medallion.", {"+Item_Medal_Spirit"}},
        {"all_medals", "The Six Sages", "Hold all six medallions.", {"+Item_Medal_Light", "+Item_Medal_Forest", "+Item_Medal_Fire", "+Item_Medal_Water", "+Item_Medal_Shadow", "+Item_Medal_Spirit"}},
        {"game_clear", "Hero of Time", "Finish the game.", {"IsGameClear"}},
        // equipment
        {"slingshot", "Fairy Slingshot", "Get the Fairy Slingshot.", {"+Weapon_Range_SlingShot"}},
        {"boomerang", "What Goes Around", "Get the Boomerang.", {"+Item_Boomerang"}},
        {"bombs", "Handle with Care", "Get your first Bombs.", {"+Item_Bomb"}},
        {"bottle", "Bottled Up", "Get an empty Bottle.", {"+Item_Bottle"}},
        {"hookshot", "Pulled Along", "Get the Hookshot.", {"+Item_Hookshot"}},
        {"longshot", "Long Reach", "Get the Longshot.", {"+Item_Longshot"}},
        {"bow", "Fairy Bow", "Get the Fairy Bow.", {"+Weapon_Range_Bow"}},
        {"hammer", "Megaton Hammer", "Get the Megaton Hammer.", {"+Item_Hammer"}},
        {"lens", "Seeing the Unseen", "Get the Lens of Truth.", {"+Armor_GlassTruth"}},
        {"mirror_shield", "Mirror Shield", "Get the Mirror Shield.", {"+Weapon_Shield_Mirror"}},
        {"tunics", "Dressed for the Occasion", "Get the Goron Tunic and the Zora Tunic.", {"+Armor_Adult_Goron", "+Armor_Adult_Zora"}},
        {"boots", "Well Heeled", "Get the Iron Boots and the Hover Boots.", {"+Armor_Adult_HeavyBoot", "+Armor_Adult_HoverBoot"}},
        {"golden_gauntlets", "Golden Gauntlets", "Get the Golden Gauntlets.", {"+Armor_Adult_Gauntlet_Gold"}},
        {"golden_scale", "Golden Scale", "Get the Golden Scale.", {"+Item_Scale_Gold"}},
        {"magic_arrows", "Elemental Archer", "Get the Fire, Ice and Light Arrows.", {"+Item_Arrow_Fire", "+Item_Arrow_Ice", "+Item_Arrow_Light"}},
        {"great_fairy_spells", "Gifts of the Goddesses", "Learn Din's Fire, Farore's Wind and Nayru's Love.", {"+Item_DinFire", "+Item_FaroreWind", "+Item_NayruLove"}},
        {"biggoron_sword", "Biggoron's Sword", "Get Biggoron's Sword.", {"+Weapon_Sword_Biggoron"}},
        {"giant_wallet", "Big Spender", "Get the Giant's Wallet.", {"+Item_Wallet_Large"}},
        {"gerudo_card", "One of Us", "Get the Gerudo Membership Card.", {"+Item_Token_Gerudo"}},
        {"agony", "Stone of Agony", "Get the Stone of Agony.", {"+Item_AgonyStone"}},
        {"mask_truth", "The Mask of Truth", "Get the Mask of Truth.", {"+Armor_Child_Mask_Truth"}},
        {"heart_piece", "Piece of Heart", "Find your first Piece of Heart.", {"+Obj_Heart_Piece"}},
        // songs
        {"zelda_lullaby", "Royal Melody", "Learn Zelda's Lullaby.", {"+Item_Melody_ZeldaLullaby"}},
        {"epona_song", "Epona's Song", "Learn Epona's Song.", {"+Item_Melody_EponaSong"}},
        {"saria_song", "Saria's Song", "Learn Saria's Song.", {"+Item_Melody_SariaSong"}},
        {"sun_song", "Sun's Song", "Learn the Sun's Song.", {"+Item_Melody_SunSong"}},
        {"time_song", "Song of Time", "Learn the Song of Time.", {"+Item_Melody_TimeSong"}},
        {"storm_song", "Song of Storms", "Learn the Song of Storms.", {"+Item_Melody_StormSong"}},
        {"warp_songs", "Maestro", "Learn all six warp songs.", {"+Item_Melody_ForestMinuet", "+Item_Melody_FireBolero", "+Item_Melody_WaterSerenade", "+Item_Melody_ShadowNocturne", "+Item_Melody_SpiritRequiem", "+Item_Melody_LightPrelude"}},
        // collecting
        {"token_1", "Curse of the Skulltula", "Collect a Gold Skulltula token.", {}, 1},
        {"token_10", "Token Collector", "Collect 10 Gold Skulltula tokens.", {}, 10},
        {"token_50", "Skulltula Hunter", "Collect 50 Gold Skulltula tokens.", {}, 50},
        {"token_100", "Lifting the Curse", "Collect all 100 Gold Skulltula tokens.", {}, 100},
    };
    return defs;
}

std::mutex mutex;
std::map<std::string, int64_t> unlocked; // id -> unix seconds
bool popups = true;
std::deque<int> queue;                   // achievements to pop up
int showing = -1;
std::chrono::steady_clock::time_point shown_at;
std::condition_variable wake;
bool pending = false;

std::string File() { return SettingsDirectory() + "/achievements.ini"; }
void Load() {
    FILE* f = std::fopen(File().c_str(), "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        std::string l = line;
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
        const size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = l.substr(0, eq), value = l.substr(eq + 1);
        if (key == "popups") popups = value != "0";
        else unlocked[key] = std::atoll(value.c_str());
    }
    std::fclose(f);
}
void Save() { // (under `mutex`)
    const std::string path = File(), temp = path + ".new";
    FILE* f = std::fopen(temp.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "popups=%d\n", popups ? 1 : 0);
    for (const auto& [id, when] : unlocked) std::fprintf(f, "%s=%lld\n", id.c_str(), (long long)when);
    std::fclose(f);
    std::error_code ec;
    sfs::rename(temp, path, ec);
}

// A save file's (hash, value) table.
bool ReadTable(const sfs::path& path, std::unordered_map<u32, s32>& out) {
    FILE* f = std::fopen(path.string().c_str(), "rb");
    if (!f) return false;
    std::vector<u8> d;
    u8 chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) d.insert(d.end(), chunk, chunk + n);
    std::fclose(f);
    if (d.size() < 0x20) return false;
    u32 magic, data_at;
    std::memcpy(&magic, d.data(), 4);
    std::memcpy(&data_at, d.data() + 8, 4);
    if (magic != 0x01020304) return false;
    const size_t end = std::min<size_t>(data_at ? data_at : d.size(), d.size());
    for (size_t at = 0x20; at + 8 <= end; at += 8) {
        u32 hash; s32 value;
        std::memcpy(&hash, d.data() + at, 4);
        std::memcpy(&value, d.data() + at + 4, 4);
        if (hash) out[hash] = value;
    }
    return true;
}

// Reads every save slot; unlocks what any of them reached. Returns how many were new.
int Scan(bool announce) {
    static const std::vector<std::vector<u32>> hashes = [] {
        std::vector<std::vector<u32>> all;
        for (const Def& d : Defs()) {
            std::vector<u32> h;
            for (const char* name : d.flags) h.push_back(name[0] == '+' ? Murmur3(("HavePlayedGetEvent." + std::string(name + 1)).c_str()) : Murmur3(name));
            all.push_back(h);
        }
        return all;
    }();
    static const u32 tokens_hash = Murmur3("CaptionData.GoldToken");
    std::set<int> reached;
    std::error_code ec;
    const sfs::path root = sfs::path(SettingsDirectory()) / "save";
    for (const auto& slot : sfs::directory_iterator(root, ec)) {
        const std::string name = slot.path().filename().string();
        if (!slot.is_directory(ec) || name.rfind("slot_", 0) != 0) continue;
        std::unordered_map<u32, s32> progress, caption;
        if (!ReadTable(slot.path() / "progress.sav", progress)) continue;
        ReadTable(slot.path() / "caption.sav", caption);
        const auto tokens_at = caption.find(tokens_hash);
        const int tokens = tokens_at == caption.end() ? 0 : tokens_at->second;
        for (size_t i = 0; i < Defs().size(); ++i) {
            const Def& d = Defs()[i];
            bool all = true;
            for (u32 h : hashes[i]) { const auto it = progress.find(h); if (it == progress.end() || it->second == 0) { all = false; break; } }
            if (d.tokens && tokens < d.tokens) all = false;
            if (all) reached.insert((int)i);
        }
    }
    std::lock_guard lock{mutex};
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    int added = 0;
    for (int i : reached) {
        if (unlocked.count(Defs()[i].id)) continue;
        unlocked[Defs()[i].id] = now;
        ++added;
        RT_LOG("achievements: %s (%s)", Defs()[i].title, announce ? "new" : "from the saves");
        if (announce && popups) queue.push_back(i);
    }
    if (added) Save();
    return added;
}
}

void Start() {
    static std::once_flag once;
    std::call_once(once, [] {
        { std::lock_guard lock{mutex}; Load(); }
        std::thread([] {
            rt::SetThreadName("achievements");
            const char* announce = std::getenv("OOT_ACHIEVEMENTS_ANNOUNCE"); // (tests: pop-ups for what the saves already reached, after <seconds>)
            if (announce) std::this_thread::sleep_for(std::chrono::seconds(std::atoi(announce)));
            const int found = Scan(announce != nullptr);
            int total = 0;
            const int have = Unlocked(&total);
            RT_LOG("achievements: %d of %d unlocked (%d found in the saves now)", have, total, found);
            for (;;) {
                {
                    std::unique_lock lock{mutex};
                    wake.wait(lock, [] { return pending; });
                    pending = false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500)); // (the rest of the save's files are written)
                const int added = Scan(true);
                static int reads = 0;
                if (++reads <= 3 || added) RT_LOG("achievements: saves read again (%d new)", added);
            }
        }).detach();
    });
}

void SavesWritten() {
    { std::lock_guard lock{mutex}; pending = true; }
    wake.notify_one();
}

std::vector<Shown> List() {
    std::lock_guard lock{mutex};
    std::vector<Shown> out;
    for (const Def& d : Defs()) {
        const auto it = unlocked.find(d.id);
        out.push_back({d.title, d.description, it != unlocked.end(), it != unlocked.end() ? it->second : 0});
    }
    return out;
}

int Unlocked(int* total) {
    std::lock_guard lock{mutex};
    if (total) *total = (int)Defs().size();
    int n = 0;
    for (const Def& d : Defs()) n += unlocked.count(d.id) ? 1 : 0;
    return n;
}

const Shown* Popup(float& age) {
    static Shown current;
    std::lock_guard lock{mutex};
    const auto now = std::chrono::steady_clock::now();
    constexpr float seconds = 6.0f;
    if (showing >= 0 && std::chrono::duration<float>(now - shown_at).count() >= seconds) showing = -1;
    if (showing < 0) {
        if (queue.empty()) return nullptr;
        showing = queue.front(); queue.pop_front(); shown_at = now;
        RT_LOG("achievements: pop-up %s", Defs()[showing].title);
    }
    age = std::chrono::duration<float>(now - shown_at).count();
    const Def& d = Defs()[showing];
    current = {d.title, d.description, true, 0};
    return &current;
}

bool PopupsOn() { std::lock_guard lock{mutex}; return popups; }
void SetPopups(bool on) { std::lock_guard lock{mutex}; popups = on; Save(); }

}
