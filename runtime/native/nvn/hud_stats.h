#pragma once
// What the stats HUD shows of the hardware (the FPS Counter's HUD levels, overlay.cpp): sampled at most twice a second, on a
// thread of its own, so the frame never waits for a file or a driver. Unknown values are negative.
#include <vector>

namespace nvn::hud {
struct Stats {
    float gpu_load = -1, gpu_temp = -1, gpu_clock = -1, gpu_power = -1, gpu_fan = -1; // (%, °C, MHz, W, %)
    float cpu_load = -1, cpu_temp = -1, cpu_clock = -1;                             // (%, °C, MHz)
    std::vector<float> core_loads;                                                  // (% each)
    float game_cpu = -1;                                                            // (the game's share of the whole CPU, %)
    double game_ram = 0, used_ram = 0, total_ram = 0, swap_used = 0;                // (bytes)
    float battery = -1, battery_watts = -1; bool charging = false;                 // (%, W drawn)
    const char* gpu_source = "";                                                    // (where the GPU figures came from)
};
Stats Current(); // (the latest sample; the first call starts the sampling)
}
