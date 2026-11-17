// (longshot) AMD FSR 3.1 frame interpolation, and the optical flow it runs on, recorded as our own Vulkan compute passes:
// the made frames between two of the game's, at any multiplier and at Frame generation's Auto.
//
// The renderer already makes frames: vk_framegen.cpp interpolates between the last two presented frames from colour alone.
// This answers the same question with AMD's passes, which use the scene's depth and motion as well (vk_fsr3_inputs.h):
// seventeen compute passes -- seven of optical flow over two frames' luma, ten of interpolation over the game's motion and
// depth -- all of them our own dispatches over AMD's vendored GLSL, with no FidelityFX host library linked and nothing of
// AMD's loaded at run time.
//
// Where it cannot run -- not asked for, a device without the storage formats, image features or subgroup operations the
// passes need, a frame without its inputs, a reset -- the renderer's own interpolation runs exactly as before. Nothing here
// changes what the game submits.
//
// FSR 3.1's interpolation has no phase input of its own (ffx_frameinterpolation.h fixes t at 0.5): AMD's passes make the
// midpoint and only the midpoint. The passes here are AMD's with t taken from the constant block instead
// (scripts/fsr3_shaders.py FI_PATCHES), so a set has as many made frames as the multiplier asks, each at its own place
// between the two game frames (1/3 and 2/3 at 3x; at Auto, each display refresh's), and the same passes keep the HUD where
// it is (the game draws it into the same picture and has no HUD-less colour). The optical flow and the dilated inputs are
// once a game frame; the rest once a made frame.
#pragma once

#include <cstdint>

#include "fsr3_math.h"
#include "vk_internal.h"

namespace mkwvk {
/// The most made frames between two of the game's: Frame generation's 8x (vk_framegen.cpp kMaxMult)
constexpr uint32_t kFsr3fiMaxMade = 7;
}

namespace mkwvk::fsr3fi {

/// One made frame's inputs.
///
/// The colour is the frame the game has just finished, at the size it is presented, and it is the only colour this backend
/// is given: the frame before it is kept here, because the interpolation needs both ends of the interval and the present
/// path only ever hands over the newest one. The depth (reversed) and the motion are the scene's at the render size
/// (vk_fsr3_inputs.h), with `motion_scale` taking a vector to render pixels, as the SDK's
/// FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS permutation the vendored GLSL was compiled under expects.
struct Inputs {
    VkImageView color = VK_NULL_HANDLE;
    VkImage color_image = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    /// Encoding of the captured present, independent of its UNORM storage format.
    bool native_pq = false;
    float max_luminance = 1.0f; // Nits for PQ; ignored for the existing SDR path.
    uint32_t width = 0, height = 0;

    VkImageView depth = VK_NULL_HANDLE;
    VkImageView motion = VK_NULL_HANDLE;
    uint32_t render_width = 0, render_height = 0;

    /// Turns a motion vector into render pixels (1 for vectors already in them).
    float motion_scale[2]{1.0f, 1.0f};
    float jitter[2]{};

    /// The scene's camera: the device-to-view-depth factors the disocclusion passes compare depths with, and the tangent
    /// of half the field of view (fsr3_math.h, from the scene's projection).
    fsr3_math::Camera camera;
    float frame_time_ms = 16.667f;

    /// A cut, a load or a resize: the preparation passes are skipped for this frame, as the SDK's own dispatch skips them,
    /// because every history they read belongs to a scene that is gone.
    bool reset = false;

    /// The caller's count of its frames (vk_framegen.cpp: one a present). The frame before this one is kept here as the
    /// interval's start; when the last frame given was not the one before this (a present this backend did not see), that
    /// start is not this frame's predecessor and the call is a reset.
    uint64_t sequence = 0;
};

/// Whether this device can run the passes: push descriptors, every storage format the SDK's internal images need, and the
/// image features and subgroup operations the vendored passes compiled down to. Asked once; the one line saying why it
/// cannot is logged once.
[[nodiscard]] bool capable();

/// Whether made frames are asked for: Frame generation on, at FSR 3.1 FG or DLSS FG on the Frame generation by row (RX_VK_FSR_FG=0
/// never, =1 over the row), on a device that is capable(). Read every frame: the rows change while the game runs. vk_framegen.cpp asks
/// NVIDIA DLSS Frame Generation first; this makes what it does not.
[[nodiscard]] uint32_t frames_wanted();

/// Records `count` made frames between the frame before and this one, made frame i at times[i] (0: the frame before, 1: this
/// one; kept within 0.01 to 0.99) into made_image(i), making the pipelines and the images for these sizes first (again when
/// the sizes or the presented format change, and more made frames' images when a set asks for more). Returns how many were
/// made: 0 when it cannot (the inputs not usable, a reset frame), which leaves the caller to show whatever it would otherwise
/// have shown. Called once a game frame, with count 0 too (Auto, a refresh that shows the game's own frame): the history the
/// next set interpolates from is kept either way.
[[nodiscard]] uint32_t generate(VkCommandBuffer cmd, const Inputs& in, const float* times, uint32_t count);

/// Made frame i of the last `generate`, or null.
[[nodiscard]] VkImage made_image(uint32_t i);

/// Forgets the sequence, so the next frame is made with the reset flag set: a cut, a load, a resize.
void reset();

} // namespace mkwvk::fsr3fi
