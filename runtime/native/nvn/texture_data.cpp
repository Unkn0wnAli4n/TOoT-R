#include <atomic>
// CPU texel transfers and Vulkan image uploads, using the SDK's mip/layer layout.
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../../native.h"
#include "texture.h"
#include "commands.h"
#include "memory.h"
#include "shader_compiler.h"
#include "batch.h"
#include "astc_gpu_spv.h"
#include "bc_encoder_spv.h"

using namespace rt;
namespace nvn {
namespace {
u16 ToHalf(float value) {
    u32 bits; std::memcpy(&bits, &value, 4);
    const u32 sign = bits >> 16 & 0x8000;
    const s32 exponent = (s32)(bits >> 23 & 0xff) - 127 + 15;
    const u32 mantissa = bits & 0x7fffff;
    if (exponent <= 0) return (u16)sign;            // tiny values flush to zero
    if (exponent >= 31) return (u16)(sign | 0x7c00); // overflow to infinity
    return (u16)(sign | exponent << 10 | mantissa >> 13);
}
// RGB9E5 texels to RGBA16F (alpha 1).
void DecodeRgb9e5(const u8* in, u64 texels, u16* out) {
    for (u64 i = 0; i < texels; ++i) {
        u32 v; std::memcpy(&v, in + i * 4, 4);
        const float scale = std::ldexp(1.0f, (int)(v >> 27) - 15 - 9);
        out[i * 4 + 0] = ToHalf((float)(v & 0x1ff) * scale);
        out[i * 4 + 1] = ToHalf((float)(v >> 9 & 0x1ff) * scale);
        out[i * 4 + 2] = ToHalf((float)(v >> 18 & 0x1ff) * scale);
        out[i * 4 + 3] = 0x3c00;
    }
}
} // namespace

namespace {
u64 Deposit(u64 value, u64 mask) {
    if (mask == 0x12f) return (value & 15) | (value & 16) << 1 | (value & 32) << 3; // (GOB x)
    if (mask == 0xd0) return (value & 1) << 4 | (value & 6) << 5;                   // (GOB y)
    u64 result = 0;
    for (u64 bit = 1; mask; bit <<= 1) {
        const u64 low = mask & -mask;
        if (value & bit) result |= low;
        mask &= mask - 1;
    }
    return result;
}
void ReadStorage(u64 address, void* output, u64 size) {
    auto* dst = static_cast<u8*>(output);
    while (size) {
        const auto range = LookupAddress(address);
        const u64 count = std::min(size, range.pool ? range.available : 0x10000 - (address & 0xffff));
        if (range.pool) std::memcpy(dst, reinterpret_cast<const void*>(range.cpu_address), count);
        else std::memset(dst, 0, count); // uncommitted sparse tiles read as zero
        size -= count; dst += count; address += count;
    }
}
void WriteStorage(u64 address, const void* input, u64 size) {
    const auto* src = static_cast<const u8*>(input);
    while (size) {
        const auto range = LookupAddress(address);
        if (!range.pool) Fatal("CPU texel write into uncommitted GPU storage %#llx", (unsigned long long)address);
        const u64 count = std::min(size, range.available);
        std::memcpy(reinterpret_cast<void*>(range.cpu_address), src, count);
        size -= count; src += count; address += count;
    }
}
struct Staging {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    explicit Staging(u64 size) {
        auto& vk = Vk();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &buffer));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk.device, buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
        ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &memory));
        VK_CHECK(vkBindBufferMemory(vk.device, buffer, memory, 0));
        VK_CHECK(vkMapMemory(vk.device, memory, 0, size, 0, &mapped));
    }
    ~Staging() {
        // Batched copies may still read it: release once the recorded work completes.
        AfterFlush([buffer = buffer, memory = memory] {
            auto& vk = Vk(); vkUnmapMemory(vk.device, memory);
            vkDestroyBuffer(vk.device, buffer, nullptr); vkFreeMemory(vk.device, memory, nullptr);
        });
    }
};
}

UploadStats upload_stats;
u64 TexelStorageOffset(const Texture& t, u32 level, u32 x, u32 y, u32 z) {
    const auto& m = t.mips.at(level);
    const u32 bh = m.block_height, bd = m.block_depth;
    const u64 row_bytes = (u64)((m.width + t.info.block_w - 1) / t.info.block_w) * t.info.bytes;
    const u32 height = (m.height + t.info.block_h - 1) / t.info.block_h;
    if (t.flags & 0x50) return ((u64)z * height + y) * t.stride + x;
    const u64 gob_align = 1ULL << t.block_width;
    const u64 gobs_x = (((row_bytes + 63) / 64) + gob_align - 1) & ~(gob_align - 1);
    const u64 block_size = gobs_x << (9 + bh + bd);
    const u64 slice_size = ((height + (8u << bh) - 1) >> (3 + bh)) * block_size;
    const u64 offset = (z >> bd) * slice_size + ((z & ((1u << bd) - 1)) << (9 + bh)) +
        (y >> (3 + bh)) * block_size + (((y >> 3) & ((1u << bh) - 1)) << 9) +
        ((u64)(x >> 6) << (9 + bh + bd)) + Deposit(x & 63, 0x12f) + Deposit(y & 7, 0xd0);
    // Switch 2 exchanges the two GOB address bits used by the older layout.
    const u64 difference = ((offset >> 5) ^ (offset >> 6)) & 1;
    return offset ^ (difference << 5) ^ (difference << 6);
}


// A small pool for data-parallel texture work (deswizzling, ASTC decoding): uploads happen on the
// GPU command thread, and a large texture decoded on one core stalls a frame for tens of ms.
void ParallelFor(u32 count, const std::function<void(u32)>& fn) {
    static const u32 workers = std::max(2u, std::min(12u, std::thread::hardware_concurrency() - 2));
    if (count <= 1) { if (count) fn(0); return; }
    struct Pool {
        std::mutex m; std::condition_variable cv, done_cv; const std::function<void(u32)>* job = nullptr;
        std::atomic<u32> next{0}; u32 total = 0, active = 0; u64 generation = 0;
        void Work() { for (u32 i; (i = next.fetch_add(1)) < total;) (*job)(i); }
    };
    static Pool pool;
    static std::once_flag started;
    std::call_once(started, [] {
        for (u32 k = 0; k < workers; ++k) std::thread([] {
            rt::SetThreadName("texture-work");
            u64 seen = 0;
            for (;;) {
                { std::unique_lock lock{pool.m}; pool.cv.wait(lock, [&] { return pool.generation != seen; }); seen = pool.generation; ++pool.active; }
                pool.Work();
                { std::lock_guard lock{pool.m}; if (--pool.active == 0) pool.done_cv.notify_all(); }
            }
        }).detach();
    });
    static std::mutex serial; std::lock_guard one{serial}; // one parallel job at a time
    { std::lock_guard lock{pool.m}; pool.job = &fn; pool.total = count; pool.next = 0; ++pool.generation; }
    pool.cv.notify_all();
    pool.Work(); // the caller helps
    std::unique_lock lock{pool.m};
    pool.done_cv.wait(lock, [&] { return pool.active == 0 && pool.next.load() >= count; });
}

// ASTC on the GPU (astc_gpu.comp): the blocks are deswizzled into a staging buffer and one
// dispatch per mip level and layer decodes them into the RGBA8 image. Decoding large ASTC
// textures on the CPU took tens of ms each, stalling the frame that first used them.
struct AstcDecoder { VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE; } astc_decoder;
bool GpuAstc(const Texture& t) {
    static const bool cpu = std::getenv("OOT_CPU_ASTC") != nullptr;
    return !cpu && t.decoded_astc && t.target != Target::T3D && ((t.usage & VK_IMAGE_USAGE_STORAGE_BIT) || t.bc3);
}
// BC3 re-encoding (Texture::bc3; from Hookshot): each part is decoded into a 2048x2048 RGBA8
// scratch image, encoded by Eden's bc_encoder.comp into a block buffer and copied into the image.
struct BcEncoder {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE;
    VkImage scratch = VK_NULL_HANDLE; VkDeviceMemory scratch_memory = VK_NULL_HANDLE; VkImageView scratch_view = VK_NULL_HANDLE;
    VkBuffer out = VK_NULL_HANDLE; VkDeviceMemory out_memory = VK_NULL_HANDLE; bool fresh = true;
} bc_encoder;
constexpr u32 BcScratchDim = 2048;
void EnsureBcEncoder() {
    auto& vk = Vk(); auto& e = bc_encoder;
    if (e.pipeline) return;
    const VkDescriptorSetLayoutBinding bindings[2]{{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                                   {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; di.bindingCount = 2; di.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &di, nullptr, &e.set_layout));
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; li.setLayoutCount = 1; li.pSetLayouts = &e.set_layout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &li, nullptr, &e.layout));
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize = sizeof BC_ENCODER_COMP_SPV; mi.pCode = BC_ENCODER_COMP_SPV;
    VkShaderModule module; VK_CHECK(vkCreateShaderModule(vk.device, &mi, nullptr, &module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; ci.layout = e.layout;
    VK_CHECK(vkCreateComputePipelines(vk.device, PipelineCache(), 1, &ci, nullptr, &e.pipeline));
    vkDestroyShaderModule(vk.device, module, nullptr);
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {BcScratchDim, BcScratchDim, 1}; ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = VK_IMAGE_USAGE_STORAGE_BIT; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &e.scratch));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(vk.device, e.scratch, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &e.scratch_memory)); VK_CHECK(vkBindImageMemory(vk.device, e.scratch, e.scratch_memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = e.scratch; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(vk.device, &vi, nullptr, &e.scratch_view));
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = (u64)(BcScratchDim / 4) * (BcScratchDim / 4) * 16;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &e.out));
    vkGetBufferMemoryRequirements(vk.device, e.out, &req); ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &e.out_memory)); VK_CHECK(vkBindBufferMemory(vk.device, e.out, e.out_memory, 0));
    RT_LOG("textures: big ASTC textures are kept as BC3, re-encoded on the GPU (OOT_ASTC_RGBA8=1: as RGBA8)");
}
void FullBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}
void EnsureAstcDecoder() {
    auto& vk = Vk(); auto& d = astc_decoder;
    if (d.pipeline) return;
    const VkDescriptorSetLayoutBinding bindings[2]{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                                   {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; di.bindingCount = 2; di.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &di, nullptr, &d.set_layout));
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; li.setLayoutCount = 1; li.pSetLayouts = &d.set_layout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &li, nullptr, &d.layout));
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize = sizeof astc_gpu_spv; mi.pCode = astc_gpu_spv;
    VkShaderModule module; VK_CHECK(vkCreateShaderModule(vk.device, &mi, nullptr, &module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; ci.layout = d.layout;
    VK_CHECK(vkCreateComputePipelines(vk.device, PipelineCache(), 1, &ci, nullptr, &d.pipeline));
    vkDestroyShaderModule(vk.device, module, nullptr);
}
void UploadAstcOnGpu(Texture& t) {
    auto& vk = Vk();
    EnsureAstcDecoder();
    const u64 align = std::max<u64>(vk.properties.limits.minStorageBufferOffsetAlignment, 16);
    struct Part { u32 level, layer, row_blocks, block_rows; u64 offset, size; };
    std::vector<Part> parts; u64 total = 0;
    for (u32 level = 0; level < t.levels; ++level) {
        const auto& m = t.mips[level];
        const u32 bx = (m.width + t.info.block_w - 1) / t.info.block_w, by = (m.height + t.info.block_h - 1) / t.info.block_h;
        for (u32 layer = 0; layer < t.layers; ++layer) {
            total = (total + align - 1) & ~(align - 1);
            parts.push_back({level, layer, bx, by, total, (u64)bx * by * 16});
            total += (u64)bx * by * 16;
        }
    }
    Staging staging(std::max<u64>(total, 16));
    for (u32 level = 0; level < t.levels; ++level) {
        const auto& m = t.mips[level];
        std::vector<u8> storage(m.storage_size);
        for (u32 layer = 0; layer < t.layers; ++layer) {
            const Part& part = parts[level * t.layers + layer];
            ReadStorage(t.storage + m.offset + layer * t.layer_stride, storage.data(), storage.size());
            u8* const base = static_cast<u8*>(staging.mapped) + part.offset;
            const u64 row_bytes = (u64)part.row_blocks * 16;
            const u32 run = (t.flags & 0x50) ? (u32)row_bytes : 16u;
            const u32 band = std::max(1u, 32768u / (u32)std::max<u64>(row_bytes, 1));
            ParallelFor((part.block_rows + band - 1) / band, [&](u32 b) {
                for (u32 y = b * band; y < std::min(part.block_rows, (b + 1) * band); ++y) {
                    u8* out = base + (u64)y * row_bytes;
                    for (u64 x = 0; x < row_bytes; x += run) {
                        const u64 n = std::min<u64>(run, row_bytes - x);
                        const u64 offset = TexelStorageOffset(t, level, (u32)x, y, 0);
                        if (offset + n > storage.size()) Fatal("texture layout exceeds mip storage");
                        std::memcpy(out, storage.data() + offset, n); out += n;
                    }
                }
            });
            upload_stats.count += 1; upload_stats.bytes += part.size;
        }
    }
    // Storage views (UNORM: sRGB formats are not storage formats; the bytes are the same).
    std::vector<VkImageView> views;
    std::vector<VkDescriptorSet> sets;
    if (t.bc3) {
        EnsureBcEncoder(); auto& e = bc_encoder;
        std::vector<VkDescriptorSet> encode_sets;
        for (const Part& part : parts) {
            const VkDescriptorSet set = AllocateBatchSet(astc_decoder.set_layout); sets.push_back(set);
            const VkDescriptorBufferInfo buffer{staging.buffer, part.offset, std::max<u64>(part.size, 16)};
            const VkDescriptorImageInfo image{VK_NULL_HANDLE, e.scratch_view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet writes[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            writes[0].dstSet = set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[0].pBufferInfo = &buffer;
            writes[1].dstSet = set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &image;
            vkUpdateDescriptorSets(vk.device, 2, writes, 0, nullptr);
        }
        {
            const VkDescriptorSet set = AllocateBatchSet(e.set_layout); encode_sets.push_back(set);
            const VkDescriptorImageInfo image{VK_NULL_HANDLE, e.scratch_view, VK_IMAGE_LAYOUT_GENERAL};
            const VkDescriptorBufferInfo buffer{e.out, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet writes[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            writes[0].dstSet = set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &image;
            writes[1].dstSet = set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].pBufferInfo = &buffer;
            vkUpdateDescriptorSets(vk.device, 2, writes, 0, nullptr);
        }
        Immediate([&](VkCommandBuffer cmd) {
            Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL);
            if (e.fresh) {
                e.fresh = false;
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = e.scratch; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            }
            for (size_t i = 0; i < parts.size(); ++i) {
                const Part& part = parts[i]; const auto& m = t.mips[part.level];
                FullBarrier(cmd); // (the scratch and the block buffer belong to the previous part)
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, astc_decoder.pipeline);
                const u32 push[4]{t.info.block_w, t.info.block_h, part.row_blocks, part.block_rows};
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, astc_decoder.layout, 0, 1, &sets[i], 0, nullptr);
                vkCmdPushConstants(cmd, astc_decoder.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
                vkCmdDispatch(cmd, (part.row_blocks + 7) / 8, (part.block_rows + 7) / 8, 1);
                FullBarrier(cmd);
                const u32 w = m.width, h = m.height, encode[6]{(w + 3) / 4, (h + 3) / 4, w, h, 1, 0};
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e.layout, 0, 1, &encode_sets[0], 0, nullptr);
                vkCmdPushConstants(cmd, e.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof encode, encode);
                vkCmdDispatch(cmd, (encode[0] + 7) / 8, (encode[1] + 7) / 8, 1);
                FullBarrier(cmd);
                const VkBufferImageCopy copy{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, part.level, part.layer, 1}, {0, 0, 0}, {w, h, 1}};
                vkCmdCopyBufferToImage(cmd, e.out, t.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
            }
            FullBarrier(cmd);
        });
        t.uploaded = true;
        return;
    }
    for (const Part& part : parts) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO}; usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        vi.pNext = &usage; vi.image = t.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, part.level, 1, part.layer, 1};
        VkImageView view; VK_CHECK(vkCreateImageView(vk.device, &vi, nullptr, &view)); views.push_back(view);
        const VkDescriptorSet set = AllocateBatchSet(astc_decoder.set_layout); sets.push_back(set);
        const VkDescriptorBufferInfo buffer{staging.buffer, part.offset, std::max<u64>(part.size, 16)};
        const VkDescriptorImageInfo image{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        writes[0].dstSet = set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[0].pBufferInfo = &buffer;
        writes[1].dstSet = set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &image;
        vkUpdateDescriptorSets(vk.device, 2, writes, 0, nullptr);
    }
    Immediate([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, astc_decoder.pipeline);
        for (size_t i = 0; i < parts.size(); ++i) {
            const Part& part = parts[i];
            const u32 push[4]{t.info.block_w, t.info.block_h, part.row_blocks, part.block_rows};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, astc_decoder.layout, 0, 1, &sets[i], 0, nullptr);
            vkCmdPushConstants(cmd, astc_decoder.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
            vkCmdDispatch(cmd, (part.row_blocks + 7) / 8, (part.block_rows + 7) / 8, 1);
        }
    });
    AfterFlush([views] { for (VkImageView v : views) vkDestroyImageView(Vk().device, v, nullptr); });
    t.uploaded = true;
}
void UploadTexels(Texture& t);
std::atomic<u64> upload_ns_frame{0}, uploads_frame{0}, upload_staging_ns{0}, upload_swizzle_ns{0}, upload_decode_ns{0}; // (slow-frame diagnosis, render.cpp)
extern std::atomic<u64> cpu_reuploads; // (memory.cpp reports)
void EnsureUploaded(Texture& t) {
    EnsureImage(t);
    // Diagnostic: CPU-sourced textures uploaded again each frame (finds textures the game keeps
    // writing on the CPU after their first use).
    static const bool reupload = std::getenv("OOT_REUPLOAD_TEXTURES") != nullptr;
    if (reupload && t.uploaded && !t.gpu_written && t.upload_frame != CurrentFrameNumber()) t.uploaded = false;
    // CPU writes to the texture's memory since its upload (menus fill label and icon textures
    // after first use): upload again.
    // (without page protection, small textures (menus' labels and icons) are hashed again every 16 frames)
    constexpr u64 HashedMax = 512u << 10;
    const auto hash_of = [](const Texture& tex, u64* out) {
        const auto range = LookupAddress(tex.storage);
        if (!range.pool || range.available < tex.storage_size) return false;
        const u64* w = reinterpret_cast<const u64*>(range.cpu_address);
        u64 h = 0x9e3779b97f4a7c15ull ^ tex.storage_size;
        for (u64 i = 0, n = tex.storage_size / 8; i < n; ++i) h = (h ^ w[i]) * 0x100000001b3ull + (h >> 29);
        *out = h;
        return true;
    };
    const bool by_content = !PageProtectionUsable();
    if (by_content && t.uploaded && !t.gpu_written && t.storage && t.storage_size <= HashedMax && CurrentFrameNumber() >= t.hash_frame + 16) {
        t.hash_frame = CurrentFrameNumber();
        u64 h; if (hash_of(t, &h) && h != t.content_hash) { t.uploaded = false; ++cpu_reuploads; }
    }
    if (!by_content && t.uploaded && !t.gpu_written && t.storage) {
        const u32 now = TextureWriteSequence();
        if (now != t.checked_sequence) {
            t.checked_sequence = now;
            if (TextureMemoryWritten(t.storage, t.storage_size, t.upload_sequence)) { t.uploaded = false; ++cpu_reuploads; }
        }
    }
    if (t.uploaded || t.gpu_written) return;
    t.upload_frame = CurrentFrameNumber();
    // Protect first, then read: writes from here on are seen by the next check.
    if (t.storage && t.samples == 1) t.upload_sequence = t.checked_sequence = ArmTextureMemory(t.storage, t.storage_size);
    if (by_content && t.storage && t.storage_size <= HashedMax) { t.hash_frame = CurrentFrameNumber(); if (!hash_of(t, &t.content_hash)) t.content_hash = 0; }
    static const bool trace_r8 = std::getenv("OOT_R8_TRACE") != nullptr;
    if (trace_r8 && t.format == 1 && t.storage) {
        std::vector<u8> bytes(std::min<u64>(t.storage_size, 1u << 20)); ReadStorage(t.storage, bytes.data(), bytes.size());
        u64 nonzero = 0; for (u8 b : bytes) nonzero += b != 0;
        static u32 shown = 0; if (shown++ < 40) RT_LOG("R8 upload %ux%u at %#llx: %llu/%zu nonzero bytes (frame %llu)", t.width, t.height, (unsigned long long)t.storage, (unsigned long long)nonzero, bytes.size(), (unsigned long long)CurrentFrameNumber());
    }
    const auto t0 = std::chrono::steady_clock::now();
    UploadTexels(t);
    upload_ns_frame += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    ++uploads_frame;
}
void UploadTexelsNow(Texture& t);
// (large uploads are reported with their time: a 288 MB texture uploaded on first use stalled the executor)
void UploadTexels(Texture& t) {
    const auto t0 = std::chrono::steady_clock::now();
    UploadTexelsNow(t);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > 30 || t.storage_size > (32ull << 20)) {
        static std::atomic<u32> shown{0};
        if (shown++ < 40) RT_LOG("texture upload: %ux%u fmt %u levels %u layers %u, %.0f MiB at %#llx, %.0f ms%s", t.width, t.height, t.format, t.levels, t.layers,
                                 t.storage_size / 1048576.0, (unsigned long long)t.storage, ms, t.gpu_written ? " (GPU-written)" : "");
    }
}
void UploadTexelsNow(Texture& t) {
    if (t.samples != 1) return; // multisampled images receive GPU writes only
    // Texel memory GPU work of unfinished batches writes (the game produces some textures with
    // buffer writes): read it only once those results are back in guest memory.
    if (t.storage && t.storage_size && HasPendingWrite(t.storage, t.storage_size)) {
        static const bool trace = std::getenv("OOT_WAIT_TRACE") != nullptr;
        if (trace) { static u32 shown = 0; if (shown++ < 10) RT_LOG("texture upload %ux%u fmt %u at %#llx waits for GPU writes", t.width, t.height, t.format, (unsigned long long)t.storage); }
        Flush();
    }
    // Textures with the NVN IMAGE / COMPRESSIBLE flags are uploaded like any other: the game
    // fills some of them from the CPU (lighting data and look-up tables; leaving those cleared
    // turns every lit surface black). Images the GPU has written are recycled with their contents
    // (AdoptCachedImage) and never re-uploaded. OOT_CLEAR_TARGETS=1 starts such textures cleared
    // instead (diagnostic).
    static const bool clear_targets = std::getenv("OOT_CLEAR_TARGETS") != nullptr;
    if ((t.flags & 0xc) && clear_targets) {
        Immediate([&](VkCommandBuffer cmd) {
            Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL);
            const VkImageSubresourceRange range{AspectOf(t), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            if (t.info.depth || t.info.stencil) {
                const VkClearDepthStencilValue zero{0.f, 0};
                vkCmdClearDepthStencilImage(cmd, t.image, t.layout, &zero, 1, &range);
            } else if (t.info.block_w == 1) {
                const VkClearColorValue zero{};
                vkCmdClearColorImage(cmd, t.image, t.layout, &zero, 1, &range);
            }
        });
        t.uploaded = true;
        return;
    }
    if (GpuAstc(t)) { UploadAstcOnGpu(t); return; }
    std::vector<VkBufferImageCopy> copies;
    u64 linear_size = 0;
    for (u32 level = 0; level < t.levels; ++level) {
        const auto& m = t.mips[level];
        linear_size = (linear_size + 15) & ~15ULL;
        const u64 bytes = t.decoded_astc ? (u64)m.width*m.height*m.depth*4 : t.decoded_rgb9e5 ? (u64)m.width*m.height*m.depth*8 : (u64)((m.width + t.info.block_w - 1) / t.info.block_w) *
            ((m.height + t.info.block_h - 1) / t.info.block_h) * m.depth * t.info.bytes;
        for (u32 layer = 0; layer < t.layers; ++layer) {
            VkBufferImageCopy copy{}; copy.bufferOffset = linear_size;
            copy.imageSubresource = {t.info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : t.info.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
            copy.imageExtent = {m.width, m.height, m.depth}; copies.push_back(copy);
            linear_size = (linear_size + bytes + 15) & ~15ULL;
        }
    }
    if (t.info.depth && t.info.stencil) Fatal("combined depth/stencil CPU uploads need separate aspect packing");
    const auto _t_st=std::chrono::steady_clock::now();
    Staging staging(linear_size);
    upload_staging_ns+=(u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-_t_st).count();
    size_t index = 0;
    for (u32 level = 0; level < t.levels; ++level) {
        const auto& m = t.mips[level];
        const u32 width = (m.width + t.info.block_w - 1) / t.info.block_w;
        const u32 height = (m.height + t.info.block_h - 1) / t.info.block_h;
        std::vector<u8> storage(m.storage_size);
        for (u32 layer = 0; layer < t.layers; ++layer) {
            ReadStorage(t.storage + m.offset + layer * t.layer_stride, storage.data(), storage.size());
            auto* dst = static_cast<u8*>(staging.mapped) + copies[index++].bufferOffset;
            auto* output = dst;
            std::vector<u8> compressed;
            if(t.decoded_astc||t.decoded_rgb9e5){compressed.resize((u64)width*height*m.depth*t.info.bytes);dst=compressed.data();}
            // Rows are copied in 16-byte runs: pitch-linear rows are contiguous, and a block-linear GOB
            // keeps each aligned 16-byte run of a row contiguous (only higher address bits move).
            const u64 row_bytes = (u64)width * t.info.bytes;
            const u32 run = (t.flags & 0x50) ? (u32)row_bytes : 16 % t.info.bytes == 0 ? 16u : t.info.bytes;
            // Rows in bands across the pool (each row's destination is known).
            const u32 rows = height * m.depth, band = std::max(1u, 32768u / (u32)std::max<u64>(row_bytes, 1));
            u8* const base = dst;
            const auto _t_sw=std::chrono::steady_clock::now();
            ParallelFor((rows + band - 1) / band, [&](u32 b) {
                for (u32 r = b * band; r < std::min(rows, (b + 1) * band); ++r) {
                    const u32 z = r / height, y = r % height;
                    u8* out = base + (u64)r * row_bytes;
                    for (u64 x = 0; x < row_bytes; x += run) {
                        const u64 n = std::min<u64>(run, row_bytes - x);
                        const u64 offset = TexelStorageOffset(t, level, (u32)x, y, z);
                        if (offset + n > storage.size()) Fatal("texture layout exceeds mip storage");
                        std::memcpy(out, storage.data() + offset, n); out += n;
                    }
                }
            });
            upload_swizzle_ns+=(u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-_t_sw).count();
            upload_stats.count += 1; upload_stats.bytes += row_bytes * height * m.depth;
            const auto _t_dc=std::chrono::steady_clock::now();
            if(t.decoded_astc){
                // ASTC blocks decode independently: bands of block rows across the pool.
                if(m.depth!=1)nvshader::DecodeAstc(compressed.data(),compressed.size(),m.width,m.height,m.depth,t.info.block_w,t.info.block_h,output);
                else{const u32 block_rows=height,per=std::max(1u,256u/std::max(1u,width));
                    ParallelFor((block_rows+per-1)/per,[&](u32 b){
                        const u32 r0=b*per,r1=std::min(block_rows,r0+per);
                        const u32 h=std::min(m.height-r0*t.info.block_h,(r1-r0)*t.info.block_h);
                        nvshader::DecodeAstc(compressed.data()+(u64)r0*width*16,(u64)(r1-r0)*width*16,m.width,h,1,t.info.block_w,t.info.block_h,
                            output+(u64)r0*t.info.block_h*m.width*4);});}
            }
            upload_decode_ns+=(u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-_t_dc).count();
            if(t.decoded_rgb9e5)DecodeRgb9e5(compressed.data(),compressed.size()/4,reinterpret_cast<u16*>(output));
        }
    }
    Immediate([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmd, staging.buffer, t.image, t.layout, (u32)copies.size(), copies.data());
        Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL);
    });
    t.uploaded = true;
}

// CopyTextureToBuffer: reads a texture region (from the host image, which holds GPU-rendered
// contents) into guest memory, rows/images placed at the given strides (0 = tightly packed).
void CopyTextureToGuest(Texture& t, const TextureCopy& r, u64 address, u64 row_stride, u64 image_stride) {
    EnsureUploaded(t);
    if (t.decoded_astc || t.decoded_rgb9e5) Fatal("readback of a host-converted texture format");
    const u32 level = r.view.levels_base;
    if (level >= t.levels) Fatal("texture readback mip outside image");
    s32 x = r.region[0], y = r.region[1], z = r.region[2], w = r.region[3], h = r.region[4], d = r.region[5];
    if (t.target == Target::T1DArray) { z = y; d = h; y = 0; h = 1; }
    if (w <= 0 || h <= 0 || d <= 0) return;
    const bool array = t.target != Target::T3D;
    const u32 bw = t.info.block_w, bh = t.info.block_h, bytes = t.info.bytes;
    const u64 blocks_w = (w + bw - 1) / bw, blocks_h = (h + bh - 1) / bh;
    const u64 tight_row = blocks_w * bytes, tight_image = tight_row * blocks_h;
    const u64 row = row_stride ? row_stride : tight_row, image = image_stride ? image_stride : row * blocks_h;
    Staging staging(tight_image * d);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {AspectOf(t) == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : AspectOf(t),
                             level, array ? r.view.layers_base + (u32)z : 0, array ? (u32)d : 1};
    copy.imageOffset = {x, y, array ? 0 : z};
    copy.imageExtent = {(u32)w, (u32)h, array ? 1u : (u32)d};
    Immediate([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkCmdCopyImageToBuffer(cmd, t.image, t.layout, staging.buffer, 1, &copy);
    });
    Flush(); // the readback must complete before the CPU copies it out
    u8* out = static_cast<u8*>(ResolveAddress(address, image * (d - 1) + row * (blocks_h - 1) + tight_row));
    if (!out) Fatal("texture readback destination outside mapped memory");
    const u8* in = static_cast<const u8*>(staging.mapped);
    for (s32 slice = 0; slice < d; ++slice)
        for (u64 line = 0; line < blocks_h; ++line)
            std::memcpy(out + slice * image + line * row, in + slice * tight_image + line * tight_row, tight_row);
}

// Mip 0 / layer 0 texels of a small texture, tightly packed (debugging).
std::vector<u8> ReadTexels(Texture& t) {
    EnsureUploaded(t);
    const u32 depth = t.target == Target::T3D ? t.depth : 1;
    const u64 size = (u64)t.width * t.height * depth * t.info.bytes;
    Staging staging(size);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {AspectOf(t) == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : AspectOf(t), 0, 0, 1};
    copy.imageExtent = {t.width, t.height, depth};
    Immediate([&](VkCommandBuffer cmd) { Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL); vkCmdCopyImageToBuffer(cmd, t.image, t.layout, staging.buffer, 1, &copy); });
    Flush();
    const u8* p = static_cast<const u8*>(staging.mapped);
    return std::vector<u8>(p, p + size);
}

void CopyTexture(Texture& source, const TextureCopy& src, Texture& destination, const TextureCopy& dst, u32 flags) {
    EnsureUploaded(source);
    EnsureUploaded(destination); // preserve pixels outside the destination region
    const auto region = [](const Texture& t, const TextureCopy& r, VkImageSubresourceLayers& sub,
                           VkOffset3D (&offsets)[2]) {
        const u32 level = r.view.levels_base;
        if (level >= t.levels) Fatal("texture copy mip outside image");
        const auto& mip = t.mips[level];
        s32 x=r.region[0], y=r.region[1], z=r.region[2];
        s32 w=r.region[3], h=r.region[4], d=r.region[5];
        if (t.target == Target::T1DArray) { z=y; d=h; y=0; h=1; }
        const bool array = t.target != Target::T3D;
        if (x<0 || y<0 || z<0 || w<=0 || h<=0 || d<=0 ||
            (u64)x+w>mip.width || (u64)y+h>mip.height ||
            (u64)z+d+(array?r.view.layers_base:0)>(array?t.layers:mip.depth))
            Fatal("texture copy region outside image");
        sub = {AspectOf(t),level,array?r.view.layers_base+(u32)z:0,array?(u32)d:1};
        offsets[0]={x,y,array?0:z}; offsets[1]={x+w,y+h,array?1:z+d};
    };
    VkImageBlit blit{};
    region(source,src,blit.srcSubresource,blit.srcOffsets);
    region(destination,dst,blit.dstSubresource,blit.dstOffsets);
    if (blit.srcSubresource.layerCount != blit.dstSubresource.layerCount || source.samples!=1 || destination.samples!=1)
        Fatal("texture copy requires matching layers and single-sample images");
    if (flags & ~1u) Fatal("texture copy flags %#x are not supported",flags);
    const bool same_size = blit.srcOffsets[1].x-blit.srcOffsets[0].x==blit.dstOffsets[1].x-blit.dstOffsets[0].x &&
        blit.srcOffsets[1].y-blit.srcOffsets[0].y==blit.dstOffsets[1].y-blit.dstOffsets[0].y &&
        blit.srcOffsets[1].z-blit.srcOffsets[0].z==blit.dstOffsets[1].z-blit.dstOffsets[0].z;
    Immediate([&](VkCommandBuffer cmd) {
        // GENERAL also permits copies between non-overlapping subresources of one image.
        Transition(cmd,source,VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmd,destination,VK_IMAGE_LAYOUT_GENERAL);
        const VkImageAspectFlags aspects=AspectOf(source);
        if (aspects!=AspectOf(destination)) Fatal("texture copy aspect mismatch");
        for (VkImageAspectFlags bit=1;bit<=VK_IMAGE_ASPECT_STENCIL_BIT;bit<<=1) {
            if (!(aspects&bit)) continue;
            blit.srcSubresource.aspectMask=blit.dstSubresource.aspectMask=bit;
            if (same_size && source.info.vk==destination.info.vk) {
                VkImageCopy copy{};copy.srcSubresource=blit.srcSubresource;copy.dstSubresource=blit.dstSubresource;
                copy.srcOffset=blit.srcOffsets[0];copy.dstOffset=blit.dstOffsets[0];
                copy.extent={(u32)(blit.srcOffsets[1].x-blit.srcOffsets[0].x),(u32)(blit.srcOffsets[1].y-blit.srcOffsets[0].y),(u32)(blit.srcOffsets[1].z-blit.srcOffsets[0].z)};
                vkCmdCopyImage(cmd,source.image,source.layout,destination.image,destination.layout,1,&copy);
            } else {
                vkCmdBlitImage(cmd,source.image,source.layout,destination.image,destination.layout,1,&blit,
                               flags&1?VK_FILTER_LINEAR:VK_FILTER_NEAREST);
            }
        }
    });
    destination.gpu_written=true;destination.uploaded=true;
}

namespace {
void Transfer(Cpu* c, bool write, bool strided) {
    Texture* t = HostTexture(c->x[0]);
    const TextureView view = c->x[1] ? ReadView(c->x[1]) : TextureView{};
    struct Region { s32 x, y, z, width, height, depth; } region;
    std::memcpy(&region, Ptr(c, 2), sizeof region);
    const u32 level = view.levels_base;
    if (!t || level >= t->levels) Fatal("invalid texture texel-transfer view");
    const auto& m = t->mips[level];
    if (t->gpu_written) Fatal("CPU read/write of rendered texture requires GPU download");
    if (t->target == Target::T1DArray) { region.z = region.y; region.depth = region.height; region.y = 0; region.height = 1; }
    const bool array = t->layers > 1;
    if (region.x < 0 || region.y < 0 || region.z < 0 || region.width <= 0 || region.height <= 0 || region.depth <= 0 ||
        (u64)region.x + region.width > m.width || (u64)region.y + region.height > m.height ||
        (u64)region.z + region.depth + (array ? view.layers_base : 0) > (array ? t->layers : m.depth))
        Fatal("texture texel-transfer region outside image");
    const u32 width = (region.width + t->info.block_w - 1) / t->info.block_w;
    const u32 height = (region.height + t->info.block_h - 1) / t->info.block_h;
    const u64 row_stride = strided && c->x[4] ? c->x[4] : width * t->info.bytes;
    const u64 image_stride = strided && c->x[5] ? c->x[5] : row_stride * height;
    auto* pixels = Ptr<u8>(c, 3);
    // (the movie player writes each picture's planes like this: rows are copied in runs, not texel by
    // texel. Block-linear storage keeps 16 bytes of a row contiguous (TexelStorageOffset maps x bits
    // 0-3 straight through); pitch-linear storage, a whole row.)
    thread_local std::vector<u8> storage;
    if (storage.size() < m.storage_size) storage.resize(m.storage_size);
    u8* const buffer = storage.data(); // (the helper threads use the caller's buffer)
    const bool linear = t->flags & 0x50;
    const u64 x0 = (u64)(region.x / t->info.block_w) * t->info.bytes, row_len = (u64)width * t->info.bytes;
    const u32 y0 = region.y / t->info.block_h;
    const u32 mip_w = (m.width + t->info.block_w - 1) / t->info.block_w, mip_h = (m.height + t->info.block_h - 1) / t->info.block_h;
    const bool whole = x0 == 0 && y0 == 0 && width == mip_w && height == mip_h; // (a write needs no read first)
    std::atomic<bool> outside{false};
    for (s32 z = 0; z < region.depth; ++z) {
        const u32 layer = array ? view.layers_base + region.z + z : 0;
        const u32 slice = array ? 0 : region.z + z;
        const u64 address = t->storage + m.offset + layer * t->layer_stride;
        if (!write || !whole) ReadStorage(address, buffer, m.storage_size);
        u8* const plane = pixels + z * image_stride;
        const auto rows = [&](u32 first, u32 last) {
            for (u32 y = first; y < last; ++y) {
                u8* row = plane + y * row_stride;
                for (u64 xb = 0; xb < row_len;) {
                    const u64 xs = x0 + xb;
                    const u64 n = linear ? row_len : std::min<u64>(16 - (xs & 15), row_len - xb);
                    const u64 offset = TexelStorageOffset(*t, level, (u32)xs, y0 + y, slice);
                    if (offset + n > m.storage_size) { outside = true; return; }
                    if (write) std::memcpy(buffer + offset, row + xb, n);
                    else std::memcpy(row + xb, buffer + offset, n);
                    xb += n;
                }
            }
        };
        constexpr u32 Band = 64;
        if (height <= Band) rows(0, height);
        else ParallelFor((height + Band - 1) / Band, [&](u32 b) { rows(b * Band, std::min(height, (b + 1) * Band)); });
        if (outside) Fatal("texture transfer exceeds mip storage");
        if (write) WriteStorage(address, buffer, m.storage_size);
    }
    if (write) { t->uploaded = false; EnsureUploaded(*t); }
}
}
} // namespace nvn
NATIVE("nvnTextureWriteTexels") { nvn::Transfer(c, true, false); }
NATIVE("nvnTextureWriteTexelsStrided") { nvn::Transfer(c, true, true); }
NATIVE("nvnTextureReadTexels") { nvn::Transfer(c, false, false); }
NATIVE("nvnTextureReadTexelsStrided") { nvn::Transfer(c, false, true); }
