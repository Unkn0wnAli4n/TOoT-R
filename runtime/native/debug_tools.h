// The Cheats tab's tools that act inside the game (debug_tools.cpp): teleport, the time of day, Link's age, the
// inventory, flight, a free camera, damage and movement speed. The settings window asks; the game's own thread does it
// at Link's next update.
#pragma once
#include <string>
#include <vector>

struct Cpu;
namespace tools {

// ---- the hooks (engine_hooks.cpp calls them at tools/oot_patches.py's sites) ----
void PlayerUpdate(Cpu* c, unsigned long long actor);            // Link's movement update, before its ground test
void FlightPose(Cpu* c, unsigned long long actor, unsigned long long matrix);
void FlightRotation(Cpu* c, unsigned long long controller, unsigned long long matrix);
void FreeView(Cpu* c, unsigned long long view);
void Speed(Cpu* c, unsigned long long controller, bool per_second);
void Life(Cpu* c, bool damage_site);
bool SaveRequest(Cpu* c);                                        // true: the request is dropped

// ---- for the settings window ----
struct Item { const char* label; const char* name; int kind; const char* section; }; // kind 0 owned or not, 1 a quantity
const std::vector<Item>& Items();
int ItemCount(int index);            // -1 while no save is loaded
int Age();                           // 0 child, 1 adult, -1 no save loaded
std::string Map();                   // the map Link is in ("" before a save is loaded)
const std::vector<const char*>& Destinations();
std::string LastMessage();           // what the last action did, for the window
bool Available();                    // false: this game version lacks what the tools use
bool InGame();                       // Link was updated within the last quarter second

void AskTeleport(int destination);
void AskTime(int hour);
void AskAge();
void AskItem(int index, int delta);  // delta 0: owned <-> not owned
void AskMark();
void AskReturn();
inline bool flight = false;          // (not saved: off at every start)
inline bool free_camera = false;

} // namespace tools
