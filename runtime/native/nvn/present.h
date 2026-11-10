#pragma once

#include "overlay.h"
#include "vk.h"

namespace nvn {

void CreateSwapchain();
// Shows texture `index` of a guest NVNwindow.
void PresentWindowTexture(u64 window, s32 index);
bool PresentProgress(const overlay::Progress& progress); // (prewarm.cpp: one frame of the pre-game screen; true: Skip clicked)
void PresentPacing(); // (the game's thread, before the queued present)

} // namespace nvn
