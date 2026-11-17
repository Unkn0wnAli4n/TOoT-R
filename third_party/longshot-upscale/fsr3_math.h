// (longshot) AMD FSR 3.1's camera, from the scene's own projection: plain functions, no Vulkan. tests/test_fsr3_math.cpp
// checks them.
//
// FSR 3.1's passes (the upscaler's and frame interpolation's) turn a depth into a view distance with four factors,
// device_to_view_depth (the SDK's setupDeviceDepthToViewSpaceDepthParams), and frame interpolation scales its disocclusion
// test by the tangent of half the field of view. The SDK builds them from a near plane, a far plane and a field of view; the
// projection the scene was drawn with has all three in it, so they are read from there instead (the camera's uniform buffer,
// gpu/vk/vk_fsr3_inputs.cpp: the same buffer NVIDIA DLSS reads its camera from), and no hook into the game is needed.
#pragma once

#include <cfloat>
#include <cmath>

namespace mkwvk::fsr3_math {

struct Camera {
    /// view distance = [1] / (device depth - [0]); view x = [2] * ndc x * distance, y = [3] * ndc y * distance. For the
    /// depth FSR is given, which is reversed (1 at the near plane: the passes are compiled with INVERTED_DEPTH)
    float device_to_view_depth[4]{};
    float tan_half_fov_x = 0.0f;   ///< frame interpolation's fTanHalfFOV (the SDK's: horizontal, from the render aspect)
    float fov_y = 0.0f;            ///< radians
    float near_plane = 0.0f, far_plane = 0.0f;   ///< far: FLT_MAX for an infinite projection
    /// The game's own depth runs the other way (0 at the near plane): FSR's copy is 1 - it (vk_fsr3_inputs.cpp)
    bool flip_depth = false;
};

/// The camera of a perspective projection p (row major, clip = p * view position, its w row 0 0 -1 0, as the game fills its
/// views' uniform buffers), whose clip depth is NVN's -1 to 1, so the depth buffer holds (ndc z + 1) / 2 (shaders/
/// dlss_prep.comp reads it the same way). With a view distance v (> 0 in front of the camera), ndc z = -p22 + p23 / v, so the
/// stored depth is d = a + b / v with a = (1 - p22) / 2 and b = p23 / 2: reversed when b > 0. False for a projection that is
/// not one (no perspective, no depth range, a degenerate field of view)
inline bool camera_from_projection(const float p[16], Camera& c) {
    c = Camera{};
    if (p[12] != 0.0f || p[13] != 0.0f || p[14] != -1.0f || p[15] != 0.0f) return false;
    if (!(p[0] > 0.0f) || !(p[5] > 0.0f) || !std::isfinite(p[0]) || !std::isfinite(p[5])) return false;
    double a = (1.0 - (double)p[10]) * 0.5, b = (double)p[11] * 0.5;
    if (!std::isfinite(a) || !std::isfinite(b) || std::fabs(b) < 1e-12) return false;
    c.flip_depth = b < 0.0;
    if (c.flip_depth) a = 1.0 - a, b = -b;   // (FSR's copy: 1 - d = (1 - a) - b / v)
    // (an infinite projection has a = 0, which puts the far plane at device depth 0 exactly: the SDK's own factor there is
    // -FLT_EPSILON, so distances stay finite and positive)
    if (a > -FLT_EPSILON) a = -FLT_EPSILON;
    c.device_to_view_depth[0] = (float)a;
    c.device_to_view_depth[1] = (float)b;
    c.device_to_view_depth[2] = 1.0f / p[0];   // (the SDK's 1 / (cot(fov / 2) / aspect))
    c.device_to_view_depth[3] = 1.0f / p[5];   // (1 / cot(fov / 2))
    c.tan_half_fov_x = 1.0f / p[0];
    c.fov_y = 2.0f * std::atan(1.0f / p[5]);
    c.near_plane = (float)(b / (1.0 - a));
    c.far_plane = a <= -1e-6 ? (float)(-b / a) : FLT_MAX;
    return c.near_plane > 0.0f && std::isfinite(c.near_plane);
}

/// FSR's view distance for a device depth (as its shaders compute it: GetViewSpaceDepth), for the test
inline float view_distance(const Camera& c, float device_depth) {
    return c.device_to_view_depth[1] / (device_depth - c.device_to_view_depth[0]);
}

} // namespace mkwvk::fsr3_math
