// Host Vulkan context for the native NVN implementation.
#pragma once

#include <mutex>

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include "../../rt.h"

namespace nvn {
// vkCreateBuffer, logging the handle and size with OOT_ADDRESS_REPORT (which buffer a GPU fault address held).
VkResult TracedCreateBuffer(const VkBufferCreateInfo* info, VkBuffer* buffer, const char* file, int line);

struct VulkanContext {
    SDL_Window* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkDevice device = VK_NULL_HANDLE;
    u32 queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkQueue copy_queue = VK_NULL_HANDLE; // (same family: write-back copies beside the main queue; else = queue)
    std::mutex copy_mutex;               // (copy_queue's submissions, when it is its own queue)
    u32 queue_count = 1;
    bool mesh_shader_queries = false;
    bool dma_buf = false;
    bool placed_maps = false; // VK_EXT_map_memory_placed (memory.cpp: pools in GPU memory at their guest addresses)
    bool address_replay = false; // bufferDeviceAddressCaptureReplay (memory.cpp: GlobalAddressTable)
    bool robust = true; // bounds-checked buffer access (vk.cpp) // (VK_KHR_external_memory_fd + VK_EXT_external_memory_dma_buf enabled) // (VkPhysicalDeviceMeshShaderFeaturesEXT::meshShaderQueries enabled)
    VkCommandPool command_pool = VK_NULL_HANDLE;
    u64 host_import_alignment = 0x1000;
    bool buffer_float32_atomic_add=false;
    bool device_fault=false;
    PFN_vkCmdPushDescriptorSetKHR push_descriptor_set=nullptr; // VK_KHR_push_descriptor
    u32 max_push_descriptors=0; // VK_EXT_device_fault enabled
    bool conditional_rendering=false; // VK_EXT_conditional_rendering enabled
    const char* calibrated_timestamps=nullptr; // name of vkGetCalibratedTimestamps (KHR or EXT) when enabled
    std::mutex queue_mutex;
};

// Creates the window and Vulkan device on first use.
VulkanContext& Vk();
// The game's NVNdevice (+0: window origin mode, +4: depth mode; 0 = NEAR_IS_MINUS_W).
extern u64 device_object;
inline bool DepthNearIsMinusW() { return !device_object || *reinterpret_cast<const u32*>(device_object + 4) == 0; }
inline bool WindowOriginLowerLeft() { return !device_object || *reinterpret_cast<const u32*>(device_object) == 0; }

#define VK_CHECK(expr)                                                                        \
    do {                                                                                      \
        const VkResult vk_check_result = (expr);                                              \
        if (vk_check_result == VK_ERROR_DEVICE_LOST) ::nvn::ReportDeviceLost();               \
        if (vk_check_result != VK_SUCCESS) ::rt::Fatal("%s failed: %d", #expr, vk_check_result); \
    } while (0)

u32 FindMemoryType(u32 type_bits, VkMemoryPropertyFlags flags);
// Logs what VK_EXT_device_fault knows about a GPU fault (address, access type).
void ReportDeviceLost();
// Driver pipeline cache, loaded from and saved to <data>/pipeline_cache.bin so pipelines built in
// earlier runs are not compiled again. SavePipelineCache writes it when it has grown.
VkPipelineCache PipelineCache();
void SavePipelineCache();
void SavePipelineCacheBlocking(); // (at exit)

} // namespace nvn
