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
    s.preset = std::clamp(s.preset, 0, 5);
    s.resolution = std::clamp(s.resolution, 0, 5);
    s.display_mode = std::clamp(s.display_mode, 0, 1);
    if (s.render_scale) s.render_scale = std::clamp(s.render_scale, 15, 100);
    s.frame_cap = s.frame_cap <= 30 ? 30 : 60;
    s.vsync = s.vsync ? 1 : 0;
    s.filtering = std::clamp(s.filtering, 0, 6);
    s.texture_detail = std::clamp(s.texture_detail, 0, 4);
    s.texture_memory = std::clamp(s.texture_memory, 0, 2);
    s.handheld = s.handheld ? 1 : 0;
    s.upscaling = std::clamp(s.upscaling, 0, 3);
    s.sharpness = std::clamp(s.sharpness / 25 * 25, 0, 100);
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
    switch (preset) {
    case PresetLow:    s.handheld = 1; s.frame_cap = 30; s.filtering = 1; s.texture_detail = 1; s.texture_memory = 1; break;
    case PresetMedium: s.handheld = 1; s.frame_cap = 60; s.filtering = 2; s.texture_detail = 2; s.texture_memory = 1; break;
    case PresetHigh:   s.handheld = 0; s.frame_cap = 60; s.filtering = 5; s.texture_detail = 3; s.texture_memory = 0; break;
    case PresetUltra:  s.handheld = 0; s.frame_cap = 60; s.filtering = 6; s.texture_detail = 4; s.texture_memory = 2; break;
    default: break; // Original: the game's own (dynamic resolution, its filtering, 60)
    }
    return s;
}

void Load() {
    Settings s;
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
    std::lock_guard lock{mutex};
    current = Sanitized(s);
    generation.fetch_add(1);
    RT_LOG("graphics settings: preset %d, %s profile, cap %d, vsync %d, filtering %d, texture detail %d, texture memory %d",
           current.preset, current.handheld ? "handheld" : "console", current.frame_cap, current.vsync, current.filtering, current.texture_detail, current.texture_memory);
}

float LodBias() {
    static constexpr float bias[] = {0.0f, 1.5f, 0.75f, 0.0f, -0.5f};
    return bias[current.texture_detail];
}
int Anisotropy() {
    static constexpr int aniso[] = {0, 1, 1, 2, 4, 8, 16};
    return aniso[current.filtering];
}
bool ForceTrilinear() { return current.filtering >= 2; }
bool ForceBilinear() { return current.filtering == 1; }

} // namespace gfx

// Read by the patched engine code (tools/oot_patches.py): the fixed-resolution path and its scale.
extern "C" {
volatile int oot_fixed_resolution = 0;
volatile int oot_dynres_runs = 0;
volatile unsigned oot_fixed_scale_bits = 0x3f800000u; // 1.0f
volatile unsigned long long oot_dynres_context = 0; // (x19 of the resolution controller, for diagnosis)
}
