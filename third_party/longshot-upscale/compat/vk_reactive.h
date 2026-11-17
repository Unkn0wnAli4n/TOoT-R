#pragma once
#include <cstdint>
#include "vk_internal.h"
namespace mkwvk { struct Texture;
namespace reactive { // (Longshot's reactive mask, off by default there: not brought)
struct Mask { VkImage image = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkFormat format = VK_FORMAT_UNDEFINED; uint32_t image_width = 0, image_height = 0; };
inline bool wanted() { return false; }
inline bool copy_scene(VkCommandBuffer, Texture*, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
inline bool mask(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t, VkImageView, Mask&) { return false; }
} }
