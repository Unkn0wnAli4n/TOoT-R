// The game's frame to the window with the Graphics screen's Upscaling and Sharpness (upscale.cpp).
#pragma once
#include "vk.h"

namespace nvn {
// src: TRANSFER_SRC_OPTIMAL; dst: TRANSFER_DST_OPTIMAL; flip: the game's rows are bottom-up.
// False: the setting is the plain linear blit (or unavailable): the caller blits.
bool UpscaleToWindow(VkCommandBuffer cmd, VkImage src, VkExtent2D src_extent, bool flip, VkImage dst, VkRect2D rect);
}
