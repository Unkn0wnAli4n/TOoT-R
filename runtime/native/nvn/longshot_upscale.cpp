// The renderer's side of the vendored upscalers (third_party/longshot-upscale: Longshot's temporal chain, NVIDIA DLSS,
// AMD FSR 3.1 and its frame interpolation; GPL-3.0-or-later, see its PROVENANCE.md). Those files are built unchanged;
// what they take from Longshot's own renderer is declared in third_party/longshot-upscale/compat/vk_internal.h and
// defined here on top of this renderer (nvn::). The hooks that call into the chain are in render.cpp (LsUpscale...).
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <unordered_map>

#include "../../rt.h"
#include "../../platform.h"
#include "../graphics_settings.h"
#include "batch.h"
#include "longshot_upscale.h"
#include "memory.h"
#include "texture.h"
#include "vk.h"

#include "vk_internal.h"   // (third_party/longshot-upscale/compat)
#include "vk_fsr4.h"
#include "vk_xess.h"
#include "vk_temporal.h"
#include "vk_dlss.h"
#include "vk_fsr3_inputs.h"
#include "vk_fsr3fi.h"

struct LsAllocation { VkDeviceMemory memory = VK_NULL_HANDLE; u64 size = 0; void* mapped = nullptr; bool coherent = true; };

namespace nvn { bool LsReadSubmitted(u64 address, void* dst, u64 size); } // (render.cpp)

namespace mkwvk {
Device dev;
Options opts;

void logf(const char* fmt, ...) {
    char line[1024]; va_list a; va_start(a, fmt); std::vsnprintf(line, sizeof line, fmt, a); va_end(a);
    RT_LOG("%s", line);
}
void logf_at(LogLevel level, const char* fmt, ...) {
    static const bool chatty = std::getenv("OOT_UPSCALER_LOG") != nullptr;
    if (level > log_info && !chatty) return;
    char line[1024]; va_list a; va_start(a, fmt); std::vsnprintf(line, sizeof line, fmt, a); va_end(a);
    RT_LOG("%s", line);
}

uint64_t texture_frame_no() { return nvn::CurrentFrameNumber() + 1; } // (never 0: 0 is the chain's "not yet")
std::atomic<uint64_t> destroy_generation{1};
uint64_t texture_destroy_generation() { return destroy_generation.load(std::memory_order_relaxed); }
void full_barrier(VkCommandBuffer cmd, const char*) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}
void descriptors_disturbed() {} // (the chain records inside an item of its own: this renderer rebinds for the next)

BufferRef gpu_buffer_bind(uint64_t gpu, uint64_t size, bool written) {
    const nvn::BufferRange r = nvn::GpuLookupAddress(gpu);
    if (!r.buffer || r.available < size || written) return {}; // (the chain only reads guest memory here)
    VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; da.buffer = r.buffer;
    const VkDeviceAddress base = vkGetBufferDeviceAddress(dev.device, &da);
    return {r.buffer, r.offset, size, base ? base + r.offset : 0};
}
bool memory_read_as_submitted(uint64_t gpu, void* dst, uint64_t n) { return nvn::LsReadSubmitted(gpu, dst, n); }
bool ub_read(uint64_t addr, void* dst, uint64_t n) { return nvn::LsReadSubmitted(addr, dst, n); }
// What the chain lets go is freed three frames later: the game is never more than one frame ahead of the GPU
// (present pacing), so nothing in flight uses it by then.
uint64_t vk_timeline_completed() { return nvn::CurrentFrameNumber(); }
uint64_t vk_retirement_value() { return nvn::CurrentFrameNumber() + 3; }

namespace {
VmaAllocation Allocate(const VkMemoryRequirements& req, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback, bool addressable) {
    auto& vk = nvn::Vk();
    u32 type = ~0u; VkMemoryPropertyFlags got = 0;
    for (VkMemoryPropertyFlags flags : {want, fallback}) {
        for (u32 t = 0; t < vk.memory_properties.memoryTypeCount && type == ~0u; ++t)
            if ((req.memoryTypeBits >> t & 1) && (vk.memory_properties.memoryTypes[t].propertyFlags & flags) == flags) { type = t; got = vk.memory_properties.memoryTypes[t].propertyFlags; }
        if (type != ~0u) break;
    }
    if (type == ~0u) return nullptr;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = type; if (addressable) ai.pNext = &flags;
    auto* a = new LsAllocation;
    if (vkAllocateMemory(vk.device, &ai, nullptr, &a->memory) != VK_SUCCESS) { delete a; return nullptr; }
    a->size = req.size; a->coherent = (got & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return a;
}
void Free(VmaAllocation a) { if (!a) return; if (a->memory) vkFreeMemory(dev.device, a->memory, nullptr); delete a; }
bool MakeBuffer(uint64_t size, VkBufferUsageFlags usage, bool host, bool cached, VkBuffer& buffer, VmaAllocation& allocation, void** mapped) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; // (as Longshot's allocator makes every buffer: its users ask any of them for its address)
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
    if (vkCreateBuffer(dev.device, &bi, nullptr, &buffer) != VK_SUCCESS) { buffer = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev.device, buffer, &req);
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    allocation = host ? Allocate(req, visible | (cached ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0), visible, (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0)
                      : Allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0);
    if (!allocation || vkBindBufferMemory(dev.device, buffer, allocation->memory, 0) != VK_SUCCESS) { Free(allocation); allocation = nullptr; vkDestroyBuffer(dev.device, buffer, nullptr); buffer = VK_NULL_HANDLE; return false; }
    if (host && vkMapMemory(dev.device, allocation->memory, 0, VK_WHOLE_SIZE, 0, &allocation->mapped) != VK_SUCCESS) { destroy_buffer(buffer, allocation); buffer = VK_NULL_HANDLE; allocation = nullptr; return false; }
    if (mapped) *mapped = allocation->mapped;
    return true;
}
} // namespace
bool create_image(const VkImageCreateInfo& ci, VkImage& image, VmaAllocation& allocation, bool, bool, const char*) {
    if (vkCreateImage(dev.device, &ci, nullptr, &image) != VK_SUCCESS) { image = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(dev.device, image, &req);
    allocation = Allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false);
    if (!allocation || vkBindImageMemory(dev.device, image, allocation->memory, 0) != VK_SUCCESS) { Free(allocation); allocation = nullptr; vkDestroyImage(dev.device, image, nullptr); image = VK_NULL_HANDLE; return false; }
    return true;
}
void destroy_image(VkImage image, VmaAllocation allocation) { if (image) vkDestroyImage(dev.device, image, nullptr); Free(allocation); }
bool create_buffer(uint64_t size, VkBufferUsageFlags usage, bool host_visible, VkBuffer& buffer, VmaAllocation& allocation, void** mapped, bool) { return MakeBuffer(size, usage, host_visible, false, buffer, allocation, mapped); }
bool create_buffer_cached(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, void** mapped) { return MakeBuffer(size, usage, true, true, buffer, allocation, mapped); }
bool create_device_buffer(uint64_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, bool) { return MakeBuffer(size, usage, false, false, buffer, allocation, nullptr); }
void destroy_buffer(VkBuffer buffer, VmaAllocation allocation) { if (buffer) vkDestroyBuffer(dev.device, buffer, nullptr); Free(allocation); }
void flush_mapped_range(VmaAllocation a, uint64_t offset, uint64_t size) {
    if (!a || a->coherent || !size) return;
    const u64 atom = std::max<u64>(nvn::Vk().properties.limits.nonCoherentAtomSize, 1), from = offset / atom * atom;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = a->memory; r.offset = from; r.size = std::min<u64>((offset + size - from + atom - 1) / atom * atom, a->size - from);
    vkFlushMappedMemoryRanges(dev.device, 1, &r);
}
void flush_mapped(VmaAllocation a, uint64_t size) { flush_mapped_range(a, 0, size); }
uint64_t allocation_size(VmaAllocation a) { return a ? a->size : 0; }

// Constants for one recording: pieces of one mapped buffer handed out in turn. A piece is used again only after the
// whole ring went round (8 MiB against a few kilobytes a frame: many seconds).
UniformUpload uniform_upload(uint64_t size) {
    static VkBuffer buffer = VK_NULL_HANDLE; static VmaAllocation memory = nullptr; static void* mapped = nullptr; static u64 at = 0, address = 0; static bool failed = false;
    constexpr u64 Ring = 8u << 20;
    if (failed || !size || size > Ring / 4) return {};
    if (!buffer) {
        if (!MakeBuffer(Ring, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, false, buffer, memory, &mapped)) { failed = true; return {}; }
        VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; da.buffer = buffer; address = vkGetBufferDeviceAddress(dev.device, &da);
    }
    const u64 align = std::max<u64>(nvn::Vk().properties.limits.minUniformBufferOffsetAlignment, 256);
    u64 from = (at + align - 1) / align * align;
    if (from + size > Ring) from = 0;
    at = from + size;
    UniformUpload u; u.binding = {buffer, from, size, address + from}; u.mapped = static_cast<uint8_t*>(mapped) + from; u.memory = memory;
    return u;
}

VkResult extern_guarded_graphics(VkPipelineCache cache, const VkGraphicsPipelineCreateInfo* ci, VkPipeline* pl) { return vkCreateGraphicsPipelines(dev.device, cache, 1, ci, nullptr, pl); }
VkResult extern_guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* pl) { return vkCreateComputePipelines(dev.device, cache, 1, ci, nullptr, pl); }
VkResult extern_guarded_shader_module(const VkShaderModuleCreateInfo* ci, VkShaderModule* m) { return vkCreateShaderModule(dev.device, ci, nullptr, m); }
VkResult shared_compute_pipeline(const VkComputePipelineCreateInfo* ci, VkPipeline* out) { return vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, ci, nullptr, out); }
bool scene_tonemap_on() { return false; }
std::atomic<bool> present_native_pq{false}; // (Longshot's native HDR10 present: not in this renderer)
void device_wait_idle_locked() {
    auto& vk = nvn::Vk();
    std::lock_guard queue{vk.queue_mutex}; std::lock_guard copy{vk.copy_mutex};
    vkDeviceWaitIdle(vk.device);
}
void invalidate_mapped(VmaAllocation a, uint64_t size) {
    if (!a || a->coherent || !size) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = a->memory; r.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(dev.device, 1, &r);
}
PFN_vkGetInstanceProcAddr instance_proc() { return vkGetInstanceProcAddr; }
uint64_t device_vram_bytes() {
    const auto& m = nvn::Vk().memory_properties; uint64_t most = 0;
    for (u32 h = 0; h < m.memoryHeapCount; ++h) if (m.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) most = std::max<uint64_t>(most, m.memoryHeaps[h].size);
    return most;
}
// (DLSS's runtime libraries are looked for in <this folder>/ngx: the game's data folder here, where the installer puts them)
std::filesystem::path game_dir_path() { return std::filesystem::path(rt::DataDir()); }

// (AMD's FSR 4 library is looked for in RX_VK_FSR_DIR, the executable's folder and this file's folder: the data folder)
unsigned game_module_path(char* out, unsigned cap) { const std::string p = rt::DataDir() + "/tootr"; if (p.size() + 1 > cap) return 0; std::memcpy(out, p.c_str(), p.size() + 1); return (unsigned)p.size(); }
void pipeline_cache_save_play() {} // (Longshot saves its pipeline cache after FSR 4's tuned passes are built: the driver's own cache keeps them here)
// (Intel XeSS: not brought - Windows only in Longshot)
XessWhy xess_why() { return XESS_NOT_BUILT; }
const TemporalEngine* xess_engine() { return nullptr; }
} // namespace mkwvk

namespace nvn::ls {
namespace {
std::mutex texture_mutex;
std::unordered_map<const nvn::Texture*, std::unique_ptr<mkwvk::Texture>> textures;
}
void Start() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto& vk = nvn::Vk(); auto& d = mkwvk::dev;
        d.instance = vk.instance; d.physical = vk.physical; d.props = vk.properties; d.device = vk.device; d.queue_family = vk.queue_family; d.queue = vk.queue;
        vkGetPhysicalDeviceFeatures(vk.physical, &d.features);
        d.push_descriptor = vk.push_descriptor_set != nullptr; d.max_push_descriptors = vk.max_push_descriptors;
        d.subgroup_max = vk.subgroup_max ? vk.subgroup_max : 32; d.subgroup_size = d.subgroup_max;
        d.vk12.bufferDeviceAddress = VK_TRUE; d.vk13.synchronization2 = VK_TRUE; d.vk13.dynamicRendering = VK_TRUE;
    });
}
mkwvk::Texture* Texture(nvn::Texture& t) {
    std::lock_guard lock{texture_mutex};
    auto& slot = textures[&t];
    if (!slot) slot = std::make_unique<mkwvk::Texture>();
    slot->image = t.image; slot->format = t.info.vk; slot->width = t.width; slot->height = t.height;
    slot->aspect = t.info.depth ? (VK_IMAGE_ASPECT_DEPTH_BIT | (t.info.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)) : VK_IMAGE_ASPECT_COLOR_BIT;
    slot->samples = (VkSampleCountFlagBits)std::max(1u, t.samples); slot->usage = t.usage; slot->owner = &t;
    return slot.get();
}
// The chain keeps one texture between frames' steps: the scene's depth, from the draw that named it to the upscale
// (render.cpp LsNoteDraw tells which). Only that one's end makes it skip the frame. Every depth texture's end used to:
// the game ends two dozen render targets a frame, depth targets among them, and each end between the scene's draws and
// the upscale made the chain skip, which the game's own upscale then drew - a one-frame change of the whole picture,
// 143 to 202 times in 35 s of turning in Castle Town.
static const nvn::Texture* held_depth = nullptr; // (texture_mutex)
void SceneDepth(const nvn::Texture* depth) { std::lock_guard lock{texture_mutex}; held_depth = depth; }
void TextureGone(const nvn::Texture* t) {
    std::lock_guard lock{texture_mutex};
    if (t == held_depth) { held_depth = nullptr; mkwvk::destroy_generation.fetch_add(1, std::memory_order_relaxed); }
    textures.erase(t);
}
void SyncOptions() {
    const gfx::Settings& g = gfx::Current();
    // (Longshot's Upscaler row: 0 game, 1 DLSS, 2 DLAA, 3 FSR 3.1, 4 FSR 4, 5 FSR 4 Native AA. Here DLAA and the native AA are
    // DLSS and FSR 4 at Upscaler Quality's Native)
    const bool native = gfx::EffectiveRenderScale() >= 100; // (the Render Resolution row at 100%)
    mkwvk::opts.dlss = g.upscaler == 1 ? (native ? 2 : 1) : g.upscaler == 3 ? 3 : g.upscaler == 4 ? (native ? 5 : 4) : 0;
    mkwvk::opts.dlss_model = g.dlss_model;
    // (Longshot's rows: Frame generation 0 off, 2.. a multiplier; Frame generation by 0 DLSS FG, 1 built-in, 2 FSR 3.1 FG)
    static const char* forced = std::getenv("OOT_FRAME_GEN");
    const int fg = forced ? std::atoi(forced) : g.frame_gen;
    mkwvk::opts.framegen = fg <= 1 ? 0 : fg >= 7 ? 2 : fg; // (Dynamic: the count is this renderer's, a multiplier for the engines)
    mkwvk::opts.framegen_by = g.frame_gen_by == 1 ? 2 : 0;
}
namespace { const char* maker = ""; }
const char* FrameMaker() { return maker; }
// Frames in a row for which frame generation was on and made nothing: a menu or a movie, which have no 3D scene to
// make frames from. The presenter lifts a frame cap under 60 meanwhile (FramesHeld): with a cap of 30 and generated
// frames the game showed 60 in play and its movies at 30, which read as stutter.
namespace { std::atomic<int> held_frames{0}; int MakeFramesNow(VkCommandBuffer cmd, VkImage color, VkImageView view, uint32_t w, uint32_t h, int count, const VkImage* out, const VkImageView* out_views, bool blend, uint64_t sequence); }
bool FramesHeld() { return held_frames.load(std::memory_order_relaxed) >= 4; }
int MakeFrames(VkCommandBuffer cmd, VkImage color, VkImageView view, uint32_t w, uint32_t h, int count, const VkImage* out, const VkImageView* out_views, bool blend, uint64_t sequence) {
    const int made = MakeFramesNow(cmd, color, view, w, h, count, out, out_views, blend, sequence);
    if (made > 0 || count <= 0 || !mkwvk::opts.framegen) held_frames.store(0, std::memory_order_relaxed);
    else held_frames.fetch_add(1, std::memory_order_relaxed);
    return made;
}
namespace {
int MakeFramesNow(VkCommandBuffer cmd, VkImage color, VkImageView view, uint32_t w, uint32_t h, int count, const VkImage* out, const VkImageView* out_views, bool blend, uint64_t sequence) {
    using namespace mkwvk;
    maker = "";
    if (count <= 0 || !mkwvk::opts.framegen) { dlss_fg_idle(); return 0; }
    if (dlss_fg_wanted() && dlss_fg_runs(count)) {
        const int made = dlss_fg_frames(cmd, color, view, w, h, count, out, out_views, blend, sequence);
        if (made == count) { maker = "DLSS FG"; return made; }
        if (dlss_fg_holds()) return 0; // (no 3D this frame - a menu, a movie: the game's frame alone)
    }
    if (!fsr3fi::frames_wanted()) return 0;
    fsr3_inputs::Frame f;
    if (!fsr3_inputs::take(f)) { { static const bool log = std::getenv("OOT_UPSCALER_LOG") != nullptr; static int shown = 0; if (log && shown++ < 20) RT_LOG("frame generation: no inputs to take (frame %llu)", (unsigned long long)texture_frame_no()); } static bool said = false; if (!said) { said = true; RT_LOG("frame generation: no scene depth and motion this frame (a menu or a movie): the game's frames alone meanwhile"); } return 0; }
    fsr3fi::Inputs in;
    in.color = view; in.color_image = color; in.color_format = VK_FORMAT_R8G8B8A8_UNORM; in.width = w; in.height = h;
    in.depth = f.depth; in.motion = f.motion; in.render_width = f.width; in.render_height = f.height;
    in.jitter[0] = f.jitter[0]; in.jitter[1] = f.jitter[1]; in.camera = f.camera; in.frame_time_ms = f.frame_ms; in.reset = f.reset || !blend; in.sequence = sequence;
    float times[kFsr3fiMaxMade]; for (int i = 0; i < count; ++i) times[i] = (float)(i + 1) / (float)(count + 1);
    const int generated = (int)fsr3fi::generate(cmd, in, times, (uint32_t)count);
    { static const bool log = std::getenv("OOT_UPSCALER_LOG") != nullptr; static int shown = 0; if (log && shown++ < 20) RT_LOG("frame generation: asked %d, FSR 3.1 made %d (reset %d, blend %d, %ux%u from %ux%u, %.1f ms)", count, generated, (int)in.reset, (int)blend, w, h, f.width, f.height, f.frame_ms); }
    if (generated != count) return 0;
    const VkImageCopy copy{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    full_barrier(cmd, "fsr3fi made");
    for (int i = 0; i < count; ++i) { const VkImage image = fsr3fi::made_image((uint32_t)i); if (!image) return 0; vkCmdCopyImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, out[i], VK_IMAGE_LAYOUT_GENERAL, 1, &copy); }
    full_barrier(cmd, "fsr3fi copied");
    maker = "FSR 3.1 FG";
    return count;
}
} // namespace
} // namespace nvn::ls

// Extension entry points the vendored files call by name (Longshot's own loader provides them there; the system's
// Vulkan loader exports core functions only).
extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSetKHR(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes) {
    nvn::Vk().push_descriptor_set(cmd, point, layout, set, count, writes);
}
