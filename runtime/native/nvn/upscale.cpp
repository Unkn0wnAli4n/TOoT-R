// The game's 1920x1080 frame to the window's size with a better filter than the plain linear blit
// (the Graphics screen's Upscaling and Sharpness): upscale.comp, from Hookshot.
#include "upscale.h"

#include <cstring>

#include "../graphics_settings.h"
#include "upscale_spv.h"
#include "fsr_easu_spv.h"
#include "fsr_rcas_spv.h"
// FSR 1's constants on the CPU (AMD FidelityFX, MIT: third_party/fsr1).
#define A_CPU 1
#include "../../../third_party/fsr1/ffx_a.h"
#include "../../../third_party/fsr1/ffx_fsr1.h"

namespace nvn {
namespace {
struct Image { VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkExtent2D extent{}; };
struct State {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline easu = VK_NULL_HANDLE, rcas = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE, set_easu = VK_NULL_HANDLE, set_rcas = VK_NULL_HANDLE; VkSampler sampler = VK_NULL_HANDLE;
    Image input, output, middle;
    bool failed = false;
} s;

void Destroy(Image& i) {
    auto& vk = Vk();
    if (i.view) vkDestroyImageView(vk.device, i.view, nullptr);
    if (i.image) vkDestroyImage(vk.device, i.image, nullptr);
    if (i.memory) vkFreeMemory(vk.device, i.memory, nullptr);
    i = {};
}
// (the previous present has completed when this runs: its fence was waited for)
void Ensure(Image& i, VkExtent2D extent, VkImageUsageFlags usage) {
    if (i.image && i.extent.width == extent.width && i.extent.height == extent.height) return;
    Destroy(i);
    auto& vk = Vk();
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {extent.width, extent.height, 1}; ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &i.image));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(vk.device, i.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &i.memory)); VK_CHECK(vkBindImageMemory(vk.device, i.image, i.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = i.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(vk.device, &vi, nullptr, &i.view));
    i.extent = extent;
}
bool Initialize() {
    if (s.pipeline) return true;
    if (s.failed) return false;
    auto& vk = Vk();
    const VkDescriptorSetLayoutBinding bindings[2]{{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                                   {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; di.bindingCount = 2; di.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &di, nullptr, &s.set_layout));
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; li.setLayoutCount = 1; li.pSetLayouts = &s.set_layout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &li, nullptr, &s.layout));
    const auto make = [&](const u32* code, size_t size, VkPipeline& out) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize = size; mi.pCode = code;
        VkShaderModule module; VK_CHECK(vkCreateShaderModule(vk.device, &mi, nullptr, &module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; ci.layout = s.layout;
        const VkResult r = vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &out);
        vkDestroyShaderModule(vk.device, module, nullptr);
        return r == VK_SUCCESS;
    };
    if (!make(upscale_spv, sizeof upscale_spv, s.pipeline) || !make(fsr_easu_spv, sizeof fsr_easu_spv, s.easu) || !make(fsr_rcas_spv, sizeof fsr_rcas_spv, s.rcas)) {
        s.failed = true; RT_LOG("upscaling: pipeline creation failed; using the linear blit"); return false; }
    const VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 3; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(vk.device, &pi, nullptr, &s.pool));
    const VkDescriptorSetLayout layouts[3]{s.set_layout, s.set_layout, s.set_layout};
    VkDescriptorSet sets[3];
    VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; sa.descriptorPool = s.pool; sa.descriptorSetCount = 3; sa.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &sa, sets));
    s.set = sets[0]; s.set_easu = sets[1]; s.set_rcas = sets[2];
    RT_LOG("upscaling: Catmull-Rom, Lanczos and FSR 1 (EASU + RCAS) available");
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(vk.device, &si, nullptr, &s.sampler));
    return true;
}
void Barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
             VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.srcAccessMask = src; b.dstAccessMask = dst; b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}
} // namespace

bool UpscaleToWindow(VkCommandBuffer cmd, VkImage src, VkExtent2D src_extent, bool flip, VkImage dst, VkRect2D rect) {
    int mode = gfx::Current().upscaling; // 0 linear blit, 1 Catmull-Rom, 2 Lanczos, 3 FSR 1
    if (mode == 3 && (rect.extent.width <= src_extent.width || rect.extent.height <= src_extent.height)) mode = 2; // (EASU only upscales)
    const float sharpness = gfx::Current().sharpness / 100.0f;
    if ((mode == 0 && sharpness <= 0) || rect.extent.width < 16 || rect.extent.height < 16 || !Initialize()) return false;
    auto& vk = Vk();
    Ensure(s.input, src_extent, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Ensure(s.output, rect.extent, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // The game's frame, upright, into the input image (the same conversion as the plain blit).
    Barrier(cmd, s.input.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.srcOffsets[1] = {(s32)src_extent.width, (s32)src_extent.height, 1};
    copy.dstOffsets[1] = {(s32)src_extent.width, (s32)src_extent.height, 1};
    if (flip) std::swap(copy.dstOffsets[0].y, copy.dstOffsets[1].y);
    vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.input.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy, VK_FILTER_NEAREST);
    Barrier(cmd, s.input.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    Barrier(cmd, s.output.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const auto bind = [&](VkDescriptorSet set, const Image& from, const Image& to) {
        const VkDescriptorImageInfo input{s.sampler, from.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo output{VK_NULL_HANDLE, to.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &input;
        w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &output;
        vkUpdateDescriptorSets(vk.device, 2, w, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, 1, &set, 0, nullptr);
    };
    if (mode == 3) {
        // FSR 1: EASU into the middle image, then RCAS into the output (sharpness: RCAS stops, 0 the sharpest).
        Ensure(s.middle, rect.extent, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        Barrier(cmd, s.middle.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        varAU4(con0); varAU4(con1); varAU4(con2); varAU4(con3);
        FsrEasuCon(con0, con1, con2, con3, (AF1)src_extent.width, (AF1)src_extent.height, (AF1)src_extent.width, (AF1)src_extent.height,
                   (AF1)rect.extent.width, (AF1)rect.extent.height);
        u32 easu[16]; std::memcpy(easu, con0, 16); std::memcpy(easu + 4, con1, 16); std::memcpy(easu + 8, con2, 16); std::memcpy(easu + 12, con3, 16);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.easu);
        bind(s.set_easu, s.input, s.middle);
        vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof easu, easu);
        vkCmdDispatch(cmd, (rect.extent.width + 15) / 16, (rect.extent.height + 15) / 16, 1);
        Barrier(cmd, s.middle.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        varAU4(rcon);
        FsrRcasCon(rcon, (AF1)(2.0f * (1.0f - sharpness)));
        u32 rcas[4]; std::memcpy(rcas, rcon, 16);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.rcas);
        bind(s.set_rcas, s.middle, s.output);
        vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof rcas, rcas);
        vkCmdDispatch(cmd, (rect.extent.width + 15) / 16, (rect.extent.height + 15) / 16, 1);
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
        bind(s.set, s.input, s.output);
        struct { s32 mode; float sharp; } push{mode, sharpness};
        vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
        vkCmdDispatch(cmd, (rect.extent.width + 7) / 8, (rect.extent.height + 7) / 8, 1);
    }
    Barrier(cmd, s.output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit out{};
    out.srcSubresource = out.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    out.srcOffsets[1] = {(s32)rect.extent.width, (s32)rect.extent.height, 1};
    out.dstOffsets[0] = {rect.offset.x, rect.offset.y, 0};
    out.dstOffsets[1] = {rect.offset.x + (s32)rect.extent.width, rect.offset.y + (s32)rect.extent.height, 1};
    vkCmdBlitImage(cmd, s.output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &out, VK_FILTER_NEAREST);
    return true;
}

} // namespace nvn
