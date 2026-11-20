// The finished picture's way to the display: high dynamic range output (hdr.comp) and, later in this file's life, frames
// made between the game's (frame generation). present.cpp composes the picture as it always did - the game's frame
// scaled into the window, the settings window over it - but into an image of this module when one of these is on, and
// hands it over here to reach the swapchain.
#pragma once
#include "vk.h"

namespace nvn::output {
// What the swapchain is made with: the composition format in the sRGB colour space, or an HDR surface (the HDR row on
// and the system offering one). hdr: 0 none, 1 scRGB (16-bit float, linear), 2 HDR10 (10 bits, PQ).
struct Surface { VkFormat format; VkColorSpaceKHR space; int hdr; };
Surface Choose(VkFormat composition);
// Whether the picture is composed into an image of ours (Target) and not into the swapchain's.
bool Intermediate();
VkImage Target(VkExtent2D extent, VkFormat composition);
// The composed picture (in `layout`, a transfer layout) into the swapchain's image, which is left in TRANSFER_DST_OPTIMAL.
void ToSwapchain(VkCommandBuffer cmd, VkImageLayout layout, VkImage swapchain_image, VkExtent2D extent);
// What the settings in effect ask of the swapchain: a change means it is to be built again.
int Signature();

// ---- frames made between the game's (the Frame Generation rows) ----
// The frames are made by the vendored engines (third_party/longshot-upscale through longshot_upscale.cpp): NVIDIA DLSS
// Frame Generation where it runs, AMD FSR 3.1's frame interpolation otherwise, both from the scene's depth and motion
// as Longshot's temporal chain makes them. Here are the pictures they work on: the game's finished frame is composed
// into Compose(), kept as an RGBA8 picture in a ring (Stored), frames between it and the one before are made (Generate),
// and present.cpp's presenting thread shows them in turn, the game's own last. All of fg's images stay in GENERAL.
namespace fg {
constexpr int Slots = 3, MostMade = 7;
int Multiplier(double game_fps);                    // frames shown for each of the game's, by the setting (1: off)
bool Size(VkExtent2D extent, VkFormat composition); // (device idle) the images for this window size; false: cannot run here
VkImage Compose();                                  // where the frame is composed (the swapchain's format)
u64 Release();                                      // (device idle) the images let go - frame generation is off; bytes freed
void Stored(VkCommandBuffer cmd, int slot, VkImageLayout compose_layout); // the composed frame into the ring
// `count` frames between the ring's last picture and `slot`'s: how many were made (0: none - no scene this frame, or the
// engines cannot). blend: the picture before is the frame before's.
int Generate(VkCommandBuffer cmd, int slot, int count, bool blend, uint64_t sequence);
void ShowMade(VkCommandBuffer cmd, int i, VkImage swapchain_image, VkExtent2D extent); // (swapchain image left in TRANSFER_DST_OPTIMAL)
void ShowSlot(VkCommandBuffer cmd, int slot, VkImage swapchain_image, VkExtent2D extent);
}
}
