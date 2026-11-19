// Achievements, read from the game's own save files (achievements.cpp).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace achievements {

struct Shown {
    const char* title;
    const char* description;
    bool unlocked;
    int64_t when; // unix seconds (0: not unlocked, or unlocked before times were kept)
};

void Start();          // at boot: the saves there are read once, without pop-ups
void SavesWritten();   // the game committed its save data: read it again (on a thread of its own)
std::vector<Shown> List();
int Unlocked(int* total);

// The pop-up for a new achievement: what to show now, or nullptr. `age` is how long it has been shown, seconds.
const Shown* Popup(float& age);
bool PopupsOn();
void SetPopups(bool on);

}
