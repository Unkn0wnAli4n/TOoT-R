// The vendored upscalers' renderer side (longshot_upscale.cpp).
#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
namespace mkwvk { struct Texture; }
namespace nvn { struct Texture; }
namespace nvn::ls {
void Start();                               // once the device exists: Longshot's `dev` filled in
mkwvk::Texture* Texture(nvn::Texture& t);   // the chain's view of one of this renderer's textures (as it is now)
void TextureGone(const nvn::Texture* t);    // (texture.cpp, when one is destroyed)
void SyncOptions();
// Frames between the picture before and this one (color: RGBA8, GENERAL, w x h) into out[0..count): NVIDIA DLSS Frame
// Generation where it runs, AMD FSR 3.1's frame interpolation otherwise. How many were made (0: none).
int MakeFrames(VkCommandBuffer cmd, VkImage color, VkImageView view, uint32_t w, uint32_t h, int count, const VkImage* out, const VkImageView* out_views, bool blend, uint64_t sequence);
const char* FrameMaker(); // who made the last frames: "DLSS FG", "FSR 3.1 FG" or ""                         // the settings window's rows into Longshot's `opts`
}
