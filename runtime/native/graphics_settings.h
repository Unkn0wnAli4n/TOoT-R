// PC graphics settings (the Graphics screen of the pause menu's System tab, nvn/overlay.cpp).
// Stored in ~/.local/share/tootr/graphics.ini. Every setting here changes something real;
// "Original" leaves the game as it ships.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

namespace gfx {

enum Preset { PresetOriginal, PresetLow, PresetMedium, PresetHigh, PresetUltra, PresetCustom, PresetDeck };

// The frame rate caps offered, in the menu's order (0: no limit). The game's speed follows the clock at each.
inline constexpr int FrameCaps[] = {15, 20, 30, 40, 45, 60, 90, 120, 144, 165, 240, 0};

struct Settings {
    int preset = PresetOriginal;
    int resolution = 0;      // window size: 0 unchanged, 1 1280x720, 2 1600x900, 3 1920x1080, 4 2560x1440, 5 3840x2160, 6 1280x800, 7 1920x1200, 8 2560x1600, 9 3440x1440
    int display_mode = 0;    // 0 windowed, 1 fullscreen
    int render_scale = 0;    // the 3D scene's resolution: 0 the game's own dynamic one; else percent of the 1440p picture (35..100; 75 = 1080p), fixed
    int frame_cap = 60;      // frames per second: one of FrameCaps (0: no limit)
    int vsync = 0;           // 0 off, 1 on, 2 adaptive: for a variable refresh rate display (frames held just under its refresh rate)
    int filtering = 0;       // 0 original, 1 bilinear, 2 trilinear, 3..6 anisotropic 2x/4x/8x/16x
    int texture_detail = 0;  // 0 original, 1 low, 2 medium, 3 high, 4 ultra (mip LOD bias)
    int texture_memory = 0;  // 0 original, 1 low (more BC3), 2 quality (no BC3: RGBA8)
    int upscaling = 0;       // the game's 1080p frame to the window: 0 linear, 1 Catmull-Rom, 2 Lanczos, 3 FSR 1
    int sharpness = 0;       // percent, 0..200
    int show_fps = 0;        // fps and frame time in a corner of the picture: 0 off, 1 on
    int ai_upscaler = 0;     // the game's neural upscaler: 0 auto (on where the GPU runs it well), 1 off (linear upscale), 2 on
    int volume = 100;        // master volume, percent (0..100, steps of 10)
    int handheld = 0;        // quality profile: 0 console (docked), 1 handheld (the game's own lower resolution; at start)
    int anti_aliasing = 0;   // 0 the game's own (its neural upscaler, where AI Upscaler lets it run), 1 off (no network, no filter), 2 FXAA (no network; a fast filter on the finished picture)
    int aspect = 0;          // 0 16:9 with bars, 1 16:9 stretched to the window; a picture of another shape (at start): 2 the display's, 3 16:10, 4 21:9, 5 32:9, 6 4:3
    int shadow_quality = 0;  // the sun's shadow maps, texels a side: 0 the game's (1600), else 512..3072
    int foliage_shadows = 1; // grass and other GPU-culled foliage drawn into the shadow maps: 1 on (the game's), 0 off
    int fov = 100;           // gameplay camera's vertical angle, percent of the game's (80..130)
    int text_speed = 100;    // dialogue letters, percent of the game's speed (100..1000)
    int camera_speed = 100;  // right-stick camera turn, percent of the game's (50..300)
    int mute_background = 0; // 1: no sound while the window is not focused
    int probe_rate = 1;      // the game's light probe is drawn every n-th frame: 1 (the game's), 2, 4
    int skip_opening = 0;    // 1: a new game starts in Link's house, past the opening's movies (its state is set as the game sets it)
    int hud_edges = 1;       // on a picture wider than 16:9: 1 the HUD's side groups at the screen's edges, 0 in the middle 16:9
    int upscaler = 0;        // what makes the picture from the 3D scene: 0 the game's own network, 1 NVIDIA DLSS (DLAA at Upscaler Quality Native), 3 AMD FSR 3.1, 4 AMD FSR 4, 6 none (a plain stretch), 7 none + FXAA
    int upscale_quality = 2; // with DLSS or FSR 3.1: the 3D's share of the picture: 0 the Render Resolution row's, 1 native (100%), 2 Quality (67%), 3 Balanced (58%), 4 Performance (50%), 5 Ultra Performance (35%)
    int dlss_model = 0;      // DLSS's network: 0 the default (K), 1 K, 2 M, 3 L (Longshot's upscaler_rows.h)
    int frame_gen = 0;       // frames shown for each of the game's: 0 off, 2, 3, 4, 6, 7 dynamic (up to the display's refresh rate)
    int frame_gen_by = 0;    // who makes those frames: 0 auto (NVIDIA DLSS Frame Generation where it runs, else AMD FSR 3.1's), 1 AMD FSR 3.1
    int smooth_motion = 0;   // 1: NVIDIA's driver makes a frame between every two (Smooth Motion; at start, NVIDIA only)
    int hdr = 0;             // 1: the picture goes out in high dynamic range where the system offers such a surface
    int hdr_peak = 1000;     // the display's peak brightness, nits
    int hdr_paper = 200;     // the brightness of the game's white, nits
    int hdr_boost = 50;      // how far highlights are raised towards the peak, percent (0: the SDR picture as it is)
    int cheat_health = 0, cheat_rupees = 0, cheat_magic = 0; // the Cheats tab: hearts stay full; rupees never go down; magic stays full
    int game_speed = 100;    // the Hacks tab: the game's speed, percent (25..300)
    int occlusion = 0;       // the game's occlusion tests: 0 everything counts as visible (default), 1 counted on the GPU (at start; can flicker)
};

// The 3D scene's scale in effect, percent (0: the game's own dynamic resolution): the Render Resolution row's, or the
// Upscaler Quality row's while DLSS or FSR 3.1 draws, 100 for DLAA.
int EffectiveRenderScale();
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
// The 3D scene's networks (neural anti-aliasing and upscaling) are replaced by a plain upscale: the Anti-aliasing row.
bool NetworksOff();
bool Fxaa();             // the Upscaler row at None + FXAA
bool TemporalUpscaler(); // DLSS, FSR 3.1 or FSR 4 chosen
// Whether the game's window has the keyboard focus (present.cpp, each frame): Mute in Background.
inline std::atomic<bool> window_focused{true};
// The shape of the display the window is on (vk.cpp, when the window is made): Aspect Ratio at Display's shape.
inline std::atomic<double> display_ratio{0.0};
inline std::atomic<double> display_refresh{0.0}; // the display's refresh rate, Hz (0: not known)

} // namespace gfx
