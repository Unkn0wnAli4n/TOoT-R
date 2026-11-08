// mkw-recomp: the settings Eden's shader recompiler reads (gpu/vk/eden), as constants. The Vulkan backend
// renders at the game's own resolution, so resolution scaling is off.
#pragma once

#include "common/common_types.h"

namespace Settings {

template <typename T>
struct Constant {
    T value{};
    T GetValue() const {
        return value;
    }
    operator T() const {
        return value;
    }
};

struct ResolutionScalingInfo {
    u32 up_scale{1};
    u32 down_shift{0};
    f32 up_factor{1.0f};
    f32 down_factor{1.0f};
    bool active{false};
    bool downscale{false};
};

struct Values {
    bool disable_shader_loop_safety_checks{false};
    Constant<bool> fix_bloom_effects{};
    bool renderer_debug{false};
    ResolutionScalingInfo resolution_info{};
};

inline Values values{};

} // namespace Settings
