/* (longshot) Intel XeSS Super Resolution's choices as plain functions (gpu/vk/vk_xess.cpp): which quality preset takes a render
   size, where Intel's library is looked for. No Vulkan, no Intel headers: tests/test_xess_rules.py compiles them as they are. */
#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace mkwvk::xess_rules {

/// XeSS's quality presets, smallest scale first (xess.h xess_quality_settings_t: their values in kQuality). Since XeSS 1.3
/// each scaled preset takes any input from its own ratio up to the output (the SR developer guide's "Resolution scaling range
/// (dynamic)": Ultra Quality 1.0x - 1.5x ... Ultra Performance 1.0x - 3.0x), so the game's dynamic resolution and the Render
/// resolution row's stops run under the one preset whose range holds them. Native AA: the input at the output's size
enum Preset { ULTRA_QUALITY_PLUS, ULTRA_QUALITY, QUALITY, BALANCED, PERFORMANCE, ULTRA_PERFORMANCE, NATIVE_AA, kPresets };
constexpr float kRatio[kPresets] = {1.3f, 1.5f, 1.7f, 2.0f, 2.3f, 3.0f, 1.0f};
constexpr int kQuality[kPresets] = {105, 104, 103, 102, 101, 100, 106};
constexpr const char* kName[kPresets] = {"Ultra Quality Plus", "Ultra Quality", "Quality", "Balanced", "Performance",
                                         "Ultra Performance", "Native AA"};

/// The smallest input a scaled preset takes for an output side: the output over the preset's ratio, rounded up (a rounding
/// of the library's own: vk_xess.cpp asks the library, xessGetOptimalInputResolution's minimum, and uses this only when it
/// has no answer)
inline unsigned min_input(unsigned out, int preset) {
    if (preset < 0 || preset >= kPresets) return out;
    const double v = (double)out / (double)kRatio[preset];
    const unsigned r = (unsigned)std::ceil(v - 1e-6);
    return r ? r : 1;
}

/// The range of inputs a preset takes for this output (min_w x min_h up to the output)
struct Range {
    unsigned min_w = 0, min_h = 0;
};

/// The preset for render size rw x rh into ow x oh: Native AA when native_aa (the Upscaler's Intel XeSS Native AA; the input
/// is then the output's size), else the first scaled preset, smallest scale first, whose range holds the size (ranges[p]: the
/// library's minimum for preset p, or min_input's). -1: none (the input past 3x smaller, or larger than the output)
inline int preset_for(unsigned rw, unsigned rh, unsigned ow, unsigned oh, bool native_aa, const Range* ranges) {
    if (!rw || !rh || rw > ow || rh > oh) return -1;
    if (native_aa) return rw == ow && rh == oh ? NATIVE_AA : -1;
    for (int p = ULTRA_QUALITY_PLUS; p <= ULTRA_PERFORMANCE; p++)
        if (rw >= ranges[p].min_w && rh >= ranges[p].min_h) return p;
    return -1;
}
/// The same with min_input's ranges (the menu's names, and a library that does not say)
inline int preset_for(unsigned rw, unsigned rh, unsigned ow, unsigned oh, bool native_aa) {
    Range r[kPresets];
    for (int p = 0; p < kPresets; p++) r[p] = {min_input(ow, p), min_input(oh, p)};
    return preset_for(rw, rh, ow, oh, native_aa, r);
}
/// The preset a Render resolution share takes (percent of the output, the menu's twelfths), for its name in the menu: the
/// share of a 3840x2160 output as vk_menu.cpp game_size makes it. nullptr at Auto (0)
inline const char* share_name(int percent) {
    if (percent <= 0 || percent > 100) return nullptr;
    const int k = (int)(percent * 12 / 100.0f + 0.5f);   // (the twelfth the share stands for)
    const unsigned w = (3840u * (unsigned)k + 11u) / 12u, h = 2160u * (unsigned)k / 12u;
    const int p = preset_for(w, h, 3840, 2160, false);
    return p < 0 ? nullptr : kName[p];
}

/// Where Intel's library (libxess.dll, the player's own or --fetch-xess-runtime's) is looked for, in order: RX_VK_XESS_DIR (env) when set, the
/// executable's folder (bin\ beside the launcher), then the game folder when it is another. As AMD FSR 4's (vk_fsr4.cpp)
inline std::vector<std::string> library_dirs(const char* env, const std::string& exe_dir, const std::string& game_dir) {
    std::vector<std::string> d;
    if (env && *env) d.emplace_back(env);
    if (!exe_dir.empty() && (d.empty() || d.back() != exe_dir)) d.push_back(exe_dir);
    if (!game_dir.empty() && (d.empty() || d.back() != game_dir)) d.push_back(game_dir);
    return d;
}

/// RX_VK_XESS: -1 unset (the Upscaler row chooses), 0 XeSS out for this run, 1 Intel XeSS over the row, 2 (and up) its
/// Native AA. Read as the other upscalers' switches are (atoi)
inline int env_mode(const char* e) {
    if (!e || !*e) return -1;
    const int v = std::atoi(e);
    return v < 0 ? -1 : v;
}

}  // namespace mkwvk::xess_rules
