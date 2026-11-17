// What Longshot's upscaler sources (third_party/longshot-upscale, GPL-3.0-or-later) take from the rest of Longshot's
// renderer, declared here for this project's renderer: runtime/native/nvn/longshot_upscale.cpp defines every one of
// them on top of nvn::. Names and meanings are Longshot's (gpu/vk/vk_internal.h, vk_texture.h, vk_draw.h, vk_menu.h);
// only what the vendored files use is here.
#pragma once
#include <vulkan/vulkan.h>
#include "vk_api.h"
#include <cstddef>
#include <filesystem>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
typedef void* HMODULE; typedef unsigned int DWORD;
#ifndef MAX_PATH
#define MAX_PATH 4096
#endif
#define CP_UTF8 65001
inline int WideCharToMultiByte(unsigned, unsigned, const wchar_t* w, int, char* out, int cap, void*, void*) { if (!w || cap <= 0) return 0; const size_t n = wcstombs(out, w, (size_t)cap); if (n == (size_t)-1) { out[0] = 0; return 0; } return (int)n; }
inline DWORD GetModuleFileNameA(void*, char* out, DWORD cap) { const ssize_t n = readlink("/proc/self/exe", out, cap - 1); if (n <= 0) return 0; out[n] = 0; return (DWORD)n; }
inline void* GetProcAddress(HMODULE m, const char* name) { return dlsym(m, name); }
#endif
struct LsAllocation; typedef LsAllocation* VmaAllocation; // (Longshot uses VMA; here an allocation of this renderer's)

namespace mkwvk {
void logf(const char* fmt, ...);
enum LogLevel : int { log_error, log_warn, log_info, log_debug, log_trace };
void logf_at(LogLevel level, const char* fmt, ...);

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceFeatures features{};
    bool push_descriptor = false;
    uint32_t subgroup_size = 32, subgroup_max = 32;
    uint32_t max_push_descriptors = 0;
    bool hdr = false;
    VkPhysicalDeviceVulkan12Features vk12{};
    VkPhysicalDeviceVulkan13Features vk13{};
    VkFormat swap_format = VK_FORMAT_UNDEFINED;
    VkExtent2D swap_extent{};
    std::mutex present_mu;
};
extern Device dev;

struct IntelDriverGate {}; // (Windows Intel's one-compile-at-a-time gate: nothing to hold here)

// A texture of the renderer's, as the chain looks at one (longshot_upscale.cpp keeps one of these for each of ours).
struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    VkImageAspectFlags aspect = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = 0;
    void* owner = nullptr; // (this renderer's texture)
};
uint64_t texture_frame_no();            // the frame being recorded
uint64_t texture_destroy_generation();  // changes when any texture is destroyed
void full_barrier(VkCommandBuffer cmd, const char* why = "barrier (other)");
void descriptors_disturbed();           // the command buffer's bindings were changed behind the renderer's back

struct BufferRef {
    VkBuffer buffer = VK_NULL_HANDLE;
    uint64_t offset = 0, size = 0;
    VkDeviceAddress address = 0;
    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};
BufferRef gpu_buffer_bind(uint64_t gpu, uint64_t size, bool written);
bool memory_read_as_submitted(uint64_t gpu, void* dst, uint64_t n);
bool ub_read(uint64_t addr, void* dst, uint64_t n);
// Constants for one recording: a piece of a host-visible buffer that lives until the recording has run.
struct UniformUpload { BufferRef binding; uint8_t* mapped = nullptr; VmaAllocation memory = nullptr; };
UniformUpload uniform_upload(uint64_t size);
uint64_t vk_timeline_completed();   // the submissions the GPU has finished
uint64_t vk_retirement_value();     // the submission what is let go now may still be used by

bool create_image(const VkImageCreateInfo& ci, VkImage& image, VmaAllocation& allocation, bool dedicated = false, bool target = false, const char* what = nullptr);
void destroy_image(VkImage image, VmaAllocation allocation);
bool create_buffer(uint64_t size, VkBufferUsageFlags usage, bool host_visible, VkBuffer& buffer, VmaAllocation& allocation, void** mapped = nullptr, bool dedicated = false);
void destroy_buffer(VkBuffer buffer, VmaAllocation allocation);
bool create_buffer_cached(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, void** mapped = nullptr); // (host memory, mapped)
void flush_mapped(VmaAllocation allocation, uint64_t size);
void flush_mapped_range(VmaAllocation allocation, uint64_t offset, uint64_t size);
bool create_device_buffer(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, bool dedicated = false);
uint64_t allocation_size(VmaAllocation allocation);

VkResult extern_guarded_graphics(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl);
VkResult extern_guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl);
VkResult extern_guarded_shader_module(const VkShaderModuleCreateInfo* ci, VkShaderModule* m);

// (what vk_dlss.cpp takes besides, in a build with NVIDIA's SDK)
namespace rec { enum { B_DIRECT }; inline void boundary(int) {} } // (Longshot's recorder thread: commands are recorded directly here)
inline void note_work() {}
void device_wait_idle_locked();              // every queue idle (dev.present_mu held by the caller)
namespace perf_report { enum { kDlssRecreated }; }
inline void perf_event(int) {}
void invalidate_mapped(VmaAllocation allocation, uint64_t size);
PFN_vkGetInstanceProcAddr instance_proc();
uint64_t device_vram_bytes();
// (what vk_fsr4.cpp takes besides: a few of Windows' names it uses on every system, as Longshot's host/posix/windows.h gives them)
struct GprofSpan { GprofSpan(VkCommandBuffer, const char*) {} };
unsigned game_module_path(char* out, unsigned cap); // a file's path in the game's folder (its folder is what is looked in)
// (vk_menu.h) the settings the vendored files read
struct Options { int dlss = 0, dlss_model = 0, framegen = 0, framegen_by = 0, sharpness = 0, render_scale = 0; };
extern Options opts;
std::filesystem::path game_dir_path();   // the folder the game's files are in (DLSS's runtime is looked for in its ngx folder)
bool scene_tonemap_on(); // (Longshot's native HDR tone mapper: not in this renderer)
} // namespace mkwvk
