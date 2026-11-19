// NVIDIA DLSS 5 neural rendering: the engine's side, ready for NVIDIA's SDK.
//
// What NVIDIA has said publicly about it (developer blog, September 2026): a final stage after the frame is drawn, one
// frame in and one frame out, that takes the frame's colour and the engine's motion vectors and adds lighting and
// material detail; with controls for the model, "Structure Intensity" and "Tone Intensity", and masks an engine gives
// for what the model should leave alone or treat differently (foliage, water, glass). It runs on RTX 50 GPUs.
//
// Its SDK is not public: NVIDIA gives it to partner developers only, and has said general access will be announced. So
// this file holds what does not depend on it - where the stage runs, what it is handed, its settings - and the calls
// into NVIDIA's runtime are left for the day the SDK is out. Nothing of NVIDIA's is in this project: like DLSS, the SDK
// would be downloaded onto the player's PC under NVIDIA's licence, and its runtime comes with NVIDIA's driver or app.
// Builds without it (all of them, now) have no stage: Available() is false and the settings rows stay hidden.
//
// Where: the upscaler (third_party/longshot-upscale, vk_temporal.cpp) calls mkwvk::neural_stage right after it writes
// the upscaled picture - before the game draws its HUD over it, so text and icons are never touched - with that frame's
// colour, depth and motion (TemporalFrame). The picture is in frame.output, GENERAL layout.
//
// CMake: -DOOT_DLSS_NR_SDK=<folder of NVIDIA's DLSS 5 SDK> builds it in (OOT_DLSS_NR).
#include "../../rt.h"
#include "../graphics_settings.h"
#include "vk_temporal.h"

namespace nvn::neural {
namespace {

#ifdef OOT_DLSS_NR
struct State {
    bool started = false, failed = false;
    uint32_t width = 0, height = 0; // (the picture's size the feature was made for)
};
State state;

// Once: NVIDIA's runtime found and its neural rendering feature supported on this GPU.
bool Start() {
    if (state.started || state.failed) return !state.failed;
    state.started = true;
    // TODO(DLSS 5 SDK): initialise NVIDIA's runtime for Vulkan (the NGX instance the DLSS path already starts may be
    // shared), and ask whether neural rendering is supported here. On failure: RT_LOG why, state.failed = true.
    RT_LOG("DLSS 5 neural rendering: this build has the stage, but the SDK calls are not written yet");
    state.failed = true;
    return false;
}

void Stage(VkCommandBuffer cmd, const mkwvk::TemporalFrame& frame, bool motion) {
    const gfx::Settings& s = gfx::Current();
    if (!s.neural_render || !frame.output.image || !Start()) return;
    if (frame.output.width != state.width || frame.output.height != state.height) {
        state.width = frame.output.width; state.height = frame.output.height;
        // TODO(DLSS 5 SDK): (re)create the feature for this output size and the chosen model.
    }
    // What the SDK will be given:
    //  - colour in and out: frame.output (the upscaled picture, before the HUD)
    //  - motion: frame.motion when `motion` (render size: frame.render_w x frame.render_h, scaled up as DLSS takes it)
    //  - depth: frame.depth (frame.depth_reversed says which way), camera near/far/fov in frame
    //  - reset: frame.reset (a camera cut or a gap: no history)
    //  - strength: s.neural_strength percent, for the Structure / Tone Intensity controls
    //  - masks: none yet (foliage, water and glass could be marked from the game's materials)
    // TODO(DLSS 5 SDK): record the evaluation into cmd.
    (void)cmd; (void)motion;
}
#endif

} // namespace

bool Available() {
#ifdef OOT_DLSS_NR
    return Start();
#else
    return false;
#endif
}

void Install() {
#ifdef OOT_DLSS_NR
    mkwvk::neural_stage = &Stage;
#endif
}

} // namespace nvn::neural
