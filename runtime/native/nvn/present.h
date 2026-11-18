#pragma once

#include "overlay.h"
#include "vk.h"

namespace nvn {

void CreateSwapchain();
// Shows texture `index` of a guest NVNwindow.
void PresentWindowTexture(u64 window, s32 index);
struct FrameCosts { float gpu_ms, record_ms, submissions; }; // per frame, over the last 300 frames
FrameCosts LastFrameCosts();
float ShownFps(); // pictures a second on the display while frame generation makes some (0: none made)
void SetWindowCrop(s32 x, s32 y, s32 w, s32 h); // (nvnWindowSetCrop: the picture within the window's textures)
void GetWindowCrop(s32* out);
bool PresentProgress(const overlay::Progress& progress); // (prewarm.cpp: one frame of the pre-game screen; true: Skip clicked)
void PresentPacing(); // (the game's thread, before the queued present)

} // namespace nvn
