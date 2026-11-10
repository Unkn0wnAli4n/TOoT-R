// PC graphics settings (the Graphics screen of the pause menu's System tab, nvn/overlay.cpp).
// Stored in ~/.local/share/tootr/graphics.ini. Every setting here changes something real;
// "Original" leaves the game as it ships.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

namespace gfx {

enum Preset { PresetOriginal, PresetLow, PresetMedium, PresetHigh, PresetUltra, PresetCustom };

// The frame rate caps offered, in the menu's order (0: no limit). The game's speed follows the clock at each.
inline constexpr int FrameCaps[] = {15, 20, 30, 60, 90, 120, 144, 0};

struct Settings {
    int preset = PresetOriginal;
    int resolution = 0;      // window size: 0 unchanged, 1 1280x720, 2 1600x900, 3 1920x1080, 4 2560x1440, 5 3840x2160
    int display_mode = 0;    // 0 windowed, 1 fullscreen
    int render_scale = 0;    // 0: the game's own; else 15..100 (engine fixed path: not effective yet, hidden in the menu)
    int frame_cap = 60;      // frames per second: one of FrameCaps (0: no limit)
    int vsync = 0;           // 0 off, 1 on
    int filtering = 0;       // 0 original, 1 bilinear, 2 trilinear, 3..6 anisotropic 2x/4x/8x/16x
    int texture_detail = 0;  // 0 original, 1 low, 2 medium, 3 high, 4 ultra (mip LOD bias)
    int texture_memory = 0;  // 0 original, 1 low (more BC3), 2 quality (no BC3: RGBA8)
    int upscaling = 0;       // the game's 1080p frame to the window: 0 linear, 1 Catmull-Rom, 2 Lanczos, 3 FSR 1
    int sharpness = 0;       // percent, 0..100
    int show_fps = 0;        // fps and frame time in a corner of the picture: 0 off, 1 on
    int ai_upscaler = 0;     // the game's neural upscaler: 0 auto (on where the GPU runs it well), 1 off (linear upscale), 2 on
    int volume = 100;        // master volume, percent (0..100, steps of 10)
    int handheld = 0;        // quality profile: 0 console (docked), 1 handheld (the game's own lower resolution; at start)
};

// The values in effect (read from any thread; ints, so a torn read is impossible).
const Settings& Current();
// Bumped by every change (samplers and the swapchain rebuild when it moves).
uint64_t Generation();
void Set(const Settings& s); // applies and saves
Settings PresetValues(int preset);
void Load();

// Derived values.
float LodBias();          // added to the game's sampler LOD bias
int Anisotropy();         // 0: the game's own, else 1/2/4/8/16
bool ForceTrilinear();
bool ForceBilinear();

} // namespace gfx
