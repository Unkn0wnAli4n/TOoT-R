#pragma once

#include "vk.h"

namespace nvn {

void CreateSwapchain();
// Shows texture `index` of a guest NVNwindow.
void PresentWindowTexture(u64 window, s32 index);
void PresentPacing(); // (the game's thread, before the queued present)

} // namespace nvn
