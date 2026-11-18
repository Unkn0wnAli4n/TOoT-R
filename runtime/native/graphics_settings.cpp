#include <cmath>
#include "graphics_settings.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>

#include "../rt.h"

namespace gfx {
namespace {
Settings current;
std::atomic<uint64_t> generation{1};
std::mutex mutex;

std::string Path() {
    return rt::DataDir() + "/graphics.ini";
}

struct Field { const char* name; int Settings::*member; };
constexpr Field fields[] = {
    {"preset", &Settings::preset}, {"resolution", &Settings::resolution}, {"display_mode", &Settings::display_mode},
    {"render_scale", &Settings::render_scale}, {"frame_cap", &Settings::frame_cap}, {"vsync", &Settings::vsync},
    {"filtering", &Settings::filtering}, {"texture_detail", &Settings::texture_detail}, {"texture_memory", &Settings::texture_memory}, {"handheld", &Settings::handheld}, {"upscaling", &Settings::upscaling}, {"sharpness", &Settings::sharpness},
    {"show_fps", &Settings::show_fps}, {"ai_upscaler", &Settings::ai_upscaler}, {"volume", &Settings::volume},
    {"anti_aliasing", &Settings::anti_aliasing}, {"aspect", &Settings::aspect}, {"shadow_quality", &Settings::shadow_quality},
    {"foliage_shadows", &Settings::foliage_shadows}, {"fov", &Settings::fov}, {"text_speed", &Settings::text_speed},
    {"camera_speed", &Settings::camera_speed}, {"mute_background", &Settings::mute_background}, {"occlusion", &Settings::occlusion}, {"probe_rate", &Settings::probe_rate}, {"skip_opening", &Settings::skip_opening}, {"hud_edges", &Settings::hud_edges}, {"hdr", &Settings::hdr}, {"upscaler", &Settings::upscaler}, {"dlss_model", &Settings::dlss_model}, {"upscale_quality", &Settings::upscale_quality}, {"cheat_health", &Settings::cheat_health}, {"cheat_rupees", &Settings::cheat_rupees}, {"cheat_magic", &Settings::cheat_magic}, {"game_speed", &Settings::game_speed}, {"audio_device", &Settings::audio_device}, {"output_height", &Settings::output_height}, {"gpu", &Settings::gpu}, {"low_latency", &Settings::low_latency}, {"text_size", &Settings::text_size}, {"fps_corner", &Settings::fps_corner}, {"language", &Settings::language}, {"move_speed", &Settings::move_speed}, {"autosave", &Settings::autosave}, {"damage_taken", &Settings::damage_taken}, {"frame_gen", &Settings::frame_gen}, {"frame_gen_by", &Settings::frame_gen_by}, {"smooth_motion", &Settings::smooth_motion}, {"hdr_peak", &Settings::hdr_peak}, {"hdr_paper", &Settings::hdr_paper}, {"hdr_boost", &Settings::hdr_boost},
};

void Save(const Settings& s) {
    std::filesystem::create_directories(std::filesystem::path(Path()).parent_path());
    if (FILE* f = std::fopen(Path().c_str(), "w")) {
        std::fprintf(f, "# Ocarina of Time recomp graphics settings (the Graphics screen in the pause menu's System tab)\n");
        for (const auto& field : fields) std::fprintf(f, "%s=%d\n", field.name, s.*field.member);
        std::fclose(f);
    }
}

Settings Sanitized(Settings s) {
    s.preset = std::clamp(s.preset, 0, 6);
    s.resolution = std::clamp(s.resolution, 0, 9);
    s.display_mode = std::clamp(s.display_mode, 0, 1);
    if (s.render_scale) s.render_scale = std::clamp(s.render_scale, 35, 100);
    { int cap = 0; if (s.frame_cap > 0) for (int c : FrameCaps) if (c && s.frame_cap <= c) { cap = c; break; } s.frame_cap = cap; } // (the next cap up; past the highest: no limit)
    s.vsync = std::clamp(s.vsync, 0, 2);
    s.filtering = std::clamp(s.filtering, 0, 6);
    s.texture_detail = std::clamp(s.texture_detail, 0, 4);
    s.texture_memory = std::clamp(s.texture_memory, 0, 2);
    s.handheld = s.handheld ? 1 : 0;
    s.upscaling = std::clamp(s.upscaling, 0, 3);
    s.sharpness = std::clamp(s.sharpness / 25 * 25, 0, 200);
    s.show_fps = std::clamp(s.show_fps, 0, 3);
    s.ai_upscaler = std::clamp(s.ai_upscaler, 0, 2);
    s.volume = std::clamp((s.volume + 5) / 10 * 10, 0, 100);
    // (one Upscaler row since 2026-11-17: 0 the game's own, 1 NVIDIA DLSS, 3 AMD FSR 3.1, 4 AMD FSR 4, 6 none (a plain
    // stretch), 7 none + FXAA. Older files: Anti-aliasing Off / FXAA were 1 / 2 of their own row; DLAA (2) and FSR 4
    // Native AA (5) were choices of this one - now the Native step of Upscaler Quality.)
    if (s.anti_aliasing && s.upscaler == 0) s.upscaler = 5 + std::clamp(s.anti_aliasing, 1, 2);
    s.anti_aliasing = 0;
    if (s.upscaler == 2) { s.upscaler = 1; s.upscale_quality = 1; }
    if (s.upscaler == 5) { s.upscaler = 4; s.upscale_quality = 1; }
    // (one Render Resolution row since 2026-11-18: Upscaler Quality's steps are shares of it - Native 100, Quality 67,
    // Balanced 58, Performance 50, Ultra Performance 35. An older file's step becomes that share where DLSS or FSR is
    // the upscaler; with the game's own upscaler the Render Resolution it had stays.)
    if (s.upscale_quality > 0 && (s.upscaler == 1 || s.upscaler == 3 || s.upscaler == 4)) { static constexpr int share[] = {0, 100, 67, 58, 50, 35}; s.render_scale = share[std::clamp(s.upscale_quality, 1, 5)]; }
    s.upscale_quality = 0;

    s.smooth_motion = 0; // (the row is gone: Frame Generation is the one way)
    s.aspect = std::clamp(s.aspect, 0, 6);
    if (s.shadow_quality) s.shadow_quality = std::clamp(s.shadow_quality / 8 * 8, 512, 3072);
    s.foliage_shadows = s.foliage_shadows ? 1 : 0;
    s.fov = std::clamp(s.fov / 5 * 5, 80, 130);
    s.text_speed = std::clamp(s.text_speed, 100, 1000);
    s.camera_speed = std::clamp(s.camera_speed / 25 * 25, 50, 300);
    s.mute_background = s.mute_background ? 1 : 0;
    s.occlusion = s.occlusion ? 1 : 0;
    s.skip_opening = s.skip_opening ? 1 : 0;
    s.hud_edges = s.hud_edges ? 1 : 0;
    s.frame_gen = s.frame_gen == 2 || s.frame_gen == 3 || s.frame_gen == 4 || s.frame_gen == 6 || s.frame_gen == 7 ? s.frame_gen : 0; s.smooth_motion = s.smooth_motion ? 1 : 0;
    s.cheat_health = !!s.cheat_health; s.cheat_rupees = !!s.cheat_rupees; s.cheat_magic = !!s.cheat_magic; s.game_speed = std::clamp(s.game_speed, 25, 300); s.audio_device = std::clamp(s.audio_device, 0, 16); s.output_height = std::clamp(s.output_height, 0, 4320); s.gpu = std::clamp(s.gpu, 0, 8); s.low_latency = std::clamp(s.low_latency, 0, 2); s.text_size = std::clamp(s.text_size, 100, 175); s.fps_corner = std::clamp(s.fps_corner, 0, 3); s.language = std::clamp(s.language, 0, 15); s.move_speed = std::clamp(s.move_speed, 100, 400); s.autosave = std::clamp(s.autosave, 0, 1); s.damage_taken = std::clamp(s.damage_taken, 0, 4);
    s.upscaler = std::clamp(s.upscaler, 0, 7); s.dlss_model = std::clamp(s.dlss_model, 0, 3); s.upscale_quality = std::clamp(s.upscale_quality, 0, 5);
    s.hdr = s.hdr ? 1 : 0; s.hdr_peak = std::clamp(s.hdr_peak, 300, 4000); s.hdr_paper = std::clamp(s.hdr_paper, 80, 500); s.hdr_boost = std::clamp(s.hdr_boost, 0, 100);
    s.probe_rate = s.probe_rate >= 4 ? 4 : s.probe_rate >= 2 ? 2 : 1;
    return s;
}
} // namespace

void EnsureLoaded() { static std::once_flag once; std::call_once(once, [] { Load(); }); }
const Settings& Current() { EnsureLoaded(); return current; }
uint64_t Generation() { return generation.load(std::memory_order_acquire); }

void Set(const Settings& s) {
    std::lock_guard lock{mutex};
    current = Sanitized(s);
    generation.fetch_add(1, std::memory_order_acq_rel);
    Save(current);
}

Settings PresetValues(int preset) {
    Settings s;
    s.preset = preset;
    s.resolution = current.resolution;     // (the window is the player's choice in every preset)
    s.upscaling = current.upscaling; s.sharpness = current.sharpness; // (and how it is filled)
    s.display_mode = current.display_mode;
    s.show_fps = current.show_fps; s.ai_upscaler = current.ai_upscaler; s.volume = current.volume; // (and what is not about quality)
    s.aspect = current.aspect; s.fov = current.fov; s.text_speed = current.text_speed; s.camera_speed = current.camera_speed; s.mute_background = current.mute_background; s.occlusion = current.occlusion; s.skip_opening = current.skip_opening; s.hud_edges = current.hud_edges; s.hdr = current.hdr; s.upscaler = current.upscaler; s.dlss_model = current.dlss_model; s.upscale_quality = current.upscale_quality; s.cheat_health = current.cheat_health; s.cheat_rupees = current.cheat_rupees; s.cheat_magic = current.cheat_magic; s.game_speed = current.game_speed; s.audio_device = current.audio_device; s.output_height = current.output_height; s.gpu = current.gpu; s.low_latency = current.low_latency; s.text_size = current.text_size; s.fps_corner = current.fps_corner; s.language = current.language; s.move_speed = current.move_speed; s.autosave = current.autosave; s.damage_taken = current.damage_taken; s.frame_gen = current.frame_gen; s.frame_gen_by = current.frame_gen_by; s.smooth_motion = current.smooth_motion; s.hdr_peak = current.hdr_peak; s.hdr_paper = current.hdr_paper; s.hdr_boost = current.hdr_boost;
    switch (preset) {
    case PresetLow:    s.handheld = 1; s.frame_cap = 30; s.filtering = 1; s.texture_detail = 1; s.texture_memory = 1; s.render_scale = 50; s.upscaler = 7; s.shadow_quality = 1024; s.foliage_shadows = 0; break;
    case PresetMedium: s.handheld = 1; s.frame_cap = 60; s.filtering = 2; s.texture_detail = 2; s.texture_memory = 1; break;
    case PresetHigh:   s.handheld = 0; s.frame_cap = 60; s.filtering = 5; s.texture_detail = 3; s.texture_memory = 0; break;
    case PresetUltra:  s.handheld = 0; s.frame_cap = 60; s.filtering = 6; s.texture_detail = 4; s.texture_memory = 2; break;
    // (the Steam Deck: the handheld profile, the scene at a fixed 720p, no network, small shadow maps without
    // grass in them, 40 frames a second - set the Deck's screen to 40 Hz or 80 Hz with it)
    case PresetDeck:   s.handheld = 1; s.frame_cap = 40; s.filtering = 2; s.texture_detail = 2; s.texture_memory = 1; s.render_scale = 50;
                       s.upscaler = 7; s.shadow_quality = 1024; s.foliage_shadows = 0; s.upscaling = 3; s.sharpness = 25; break;
    default: break; // Original: the game's own (dynamic resolution, its filtering, 60)
    }
    return s;
}

namespace { bool first_run = false; }
void Load() {
    Settings s;
    first_run = !std::filesystem::exists(Path());
    if (FILE* f = std::fopen(Path().c_str(), "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            char* eq = std::strchr(line, '=');
            if (!eq || line[0] == '#') continue;
            *eq = 0;
            for (const auto& field : fields) if (!std::strcmp(line, field.name)) s.*field.member = std::atoi(eq + 1);
        }
        std::fclose(f);
    }
    // First start on a Steam Deck (its board names itself Jupiter, the OLED one Galileo): the Steam Deck preset, fullscreen.
    if (first_run) {
        char board[64]{};
        if (FILE* f = std::fopen("/sys/devices/virtual/dmi/id/product_name", "r")) { if (!std::fgets(board, sizeof board, f)) board[0] = 0; std::fclose(f); }
        if (!std::strncmp(board, "Jupiter", 7) || !std::strncmp(board, "Galileo", 7)) {
            { std::lock_guard lock{mutex}; current = Settings{}; }
            s = PresetValues(PresetDeck); s.display_mode = 1;
            Save(Sanitized(s));
            RT_LOG("graphics settings: first start on a Steam Deck: the Steam Deck preset");
        }
    }
    std::lock_guard lock{mutex};
    current = Sanitized(s);
    generation.fetch_add(1);
    RT_LOG("graphics settings: preset %d, %s profile, cap %d, vsync %d, filtering %d, texture detail %d, texture memory %d",
           current.preset, current.handheld ? "handheld" : "console", current.frame_cap, current.vsync, current.filtering, current.texture_detail, current.texture_memory);
}

bool TemporalUpscaler() { return current.upscaler == 1 || current.upscaler == 3 || current.upscaler == 4; }
int EffectiveRenderScale() {
    // (DLSS and FSR draw from a fixed size: the game's own dynamic resolution is not theirs to follow - Quality's 67%)
    if (TemporalUpscaler() && !current.render_scale) return 67;
    return current.render_scale;
}
float LodBias() {
    // (Low: big textures are loaded without their largest level - nvn/texture.cpp MipSkipWanted - which is a whole
    // level of bias for them already; with the 0.5 here they look as Low's 1.5 did, and the small ones a little sharper)
    static constexpr float bias[] = {0.0f, 0.5f, 0.75f, 0.0f, -0.5f};
    float b = bias[current.texture_detail];
    // (a temporal upscaler rebuilds detail finer than the 3D's pixels from many frames: textures are sampled as sharp as
    // the finished picture wants them, log2 of the 3D's share, as NVIDIA's and AMD's guides ask)
    if (TemporalUpscaler()) if (const int scale = EffectiveRenderScale(); scale > 0 && scale < 100) b += std::max(-1.5f, std::log2(scale / 100.0f));
    return b;
}
int Anisotropy() {
    static constexpr int aniso[] = {0, 1, 1, 2, 4, 8, 16};
    return aniso[current.filtering];
}
bool ForceTrilinear() { return current.filtering >= 2; }
bool ForceBilinear() { return current.filtering == 1; }
bool NetworksOff() { return current.upscaler == 6 || current.upscaler == 7; } // (the Upscaler row at None)
bool Fxaa() { return current.upscaler == 7; }

} // namespace gfx

// Read by the patched engine code (tools/oot_patches.py): the fixed-resolution path and its scale.
extern "C" {
volatile int oot_fixed_resolution = 0;
volatile int oot_dynres_runs = 0;
volatile unsigned oot_fixed_scale_bits = 0x3f800000u; // 1.0f
volatile unsigned long long oot_dynres_context = 0; // (x19 of the resolution controller, for diagnosis)
}
