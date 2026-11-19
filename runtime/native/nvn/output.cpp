#include "output.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "../../rt.h"
#include "../graphics_settings.h"
#include "hdr_spv.h"
#include "hdr_pq_spv.h"
#include "longshot_upscale.h"

namespace nvn { int DynamicFrames(); } // (present.cpp: Frame Generation at Dynamic with a frame cap)
namespace nvn::output {
namespace {
struct Image { VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkExtent2D extent{}; VkFormat format = VK_FORMAT_UNDEFINED; };
struct State {
    Surface surface{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 0};
    Image composed, hdr;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline to_scrgb = VK_NULL_HANDLE, to_pq = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE; VkSampler sampler = VK_NULL_HANDLE;
    bool failed = false;
} s;

void Destroy(Image& i) {
    auto& vk = Vk();
    if (i.view) vkDestroyImageView(vk.device, i.view, nullptr);
    if (i.image) vkDestroyImage(vk.device, i.image, nullptr);
    if (i.memory) vkFreeMemory(vk.device, i.memory, nullptr);
    i = Image{};
}
// (the caller has the device idle or the image unused: Build, or the first use at a size)
void Ensure(Image& i, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) {
    if (i.image && i.extent.width == extent.width && i.extent.height == extent.height && i.format == format) return;
    auto& vk = Vk();
    if (i.image) vkDeviceWaitIdle(vk.device);
    Destroy(i);
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = format;
    ii.extent = {extent.width, extent.height, 1}; ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &i.image));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(vk.device, i.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &i.memory)); VK_CHECK(vkBindImageMemory(vk.device, i.image, i.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = i.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(vk.device, &vi, nullptr, &i.view));
    i.extent = extent; i.format = format;
}
bool Initialize() {
    if (s.to_scrgb) return true;
    if (s.failed) return false;
    auto& vk = Vk();
    const VkDescriptorSetLayoutBinding bindings[2]{{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                                   {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; di.bindingCount = 2; di.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &di, nullptr, &s.set_layout));
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; li.setLayoutCount = 1; li.pSetLayouts = &s.set_layout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &li, nullptr, &s.layout));
    const auto make = [&](const u32* code, size_t size, VkPipeline& out) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize = size; mi.pCode = code;
        VkShaderModule module; if (vkCreateShaderModule(vk.device, &mi, nullptr, &module) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; ci.layout = s.layout;
        const VkResult r = vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &out);
        vkDestroyShaderModule(vk.device, module, nullptr);
        return r == VK_SUCCESS;
    };
    if (!make(hdr_spv, sizeof hdr_spv, s.to_scrgb) || !make(hdr_pq_spv, sizeof hdr_pq_spv, s.to_pq)) { s.failed = true; RT_LOG("HDR: its pipelines could not be made; the picture stays SDR"); return false; }
    const VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 1; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(vk.device, &pi, nullptr, &s.pool));
    VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; sa.descriptorPool = s.pool; sa.descriptorSetCount = 1; sa.pSetLayouts = &s.set_layout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &sa, &s.set));
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(vk.device, &si, nullptr, &s.sampler));
    return true;
}
void Barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.srcAccessMask = src; b.dstAccessMask = dst; b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}
} // namespace

int Signature() { return gfx::Current().hdr ? 1 : 0; }

Surface Choose(VkFormat composition) {
    auto& vk = Vk();
    s.surface = {composition, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 0};
    if (!gfx::Current().hdr || !Initialize()) return s.surface;
    u32 count = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, vk.surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count); vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, vk.surface, &count, formats.data());
    static const char* want = std::getenv("OOT_HDR_SURFACE"); // ("pq" or "scrgb": that one only)
    const auto usable = [&](VkFormat f) { VkFormatProperties p; vkGetPhysicalDeviceFormatProperties(vk.physical, f, &p); return (p.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0; };
    Surface found = s.surface;
    for (const auto& f : formats) { // (HDR10 first: what displays and compositors take most readily)
        if (f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT && (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) && usable(f.format) && !(want && want[0] == 's')) { found = {f.format, f.colorSpace, 2}; break; }
        if (f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && f.format == VK_FORMAT_R16G16B16A16_SFLOAT && found.hdr == 0 && !(want && want[0] == 'p')) found = {f.format, f.colorSpace, 1};
    }
    static int said = -1;
    if (said != found.hdr) { said = found.hdr;
        if (found.hdr) RT_LOG("HDR: the picture goes out as %s, paper white %d nits, peak %d nits", found.hdr == 2 ? "HDR10 (10 bits, PQ, BT.2020)" : "scRGB (16-bit float, linear)", gfx::Current().hdr_paper, gfx::Current().hdr_peak);
        else RT_LOG("HDR: the system offers no HDR surface for this window (is HDR switched on for the display?); the picture stays SDR"); }
    return s.surface = found;
}
bool Intermediate() { return s.surface.hdr != 0; }
VkImage Target(VkExtent2D extent, VkFormat composition) {
    Ensure(s.composed, extent, composition, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    return s.composed.image;
}
namespace {
// The HDR conversion of a picture (its view, in SHADER_READ_ONLY_OPTIMAL) into the swapchain's image, with `set`.
void Convert(VkCommandBuffer cmd, VkImageView source, VkDescriptorSet set, VkImage swap, VkExtent2D extent) {
    auto& vk = Vk();
    Ensure(s.hdr, extent, s.surface.format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    Barrier(cmd, s.hdr.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkDescriptorImageInfo in{s.sampler, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}, out{VK_NULL_HANDLE, s.hdr.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &in;
    w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &out;
    vkUpdateDescriptorSets(vk.device, 2, w, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.surface.hdr == 2 ? s.to_pq : s.to_scrgb);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, 1, &set, 0, nullptr);
    const gfx::Settings& g = gfx::Current();
    struct { float paper, peak, strength, knee; } push{(float)g.hdr_paper, (float)std::max(g.hdr_peak, g.hdr_paper), g.hdr_boost / 100.0f, 0.55f};
    vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
    vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    Barrier(cmd, s.hdr.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    Barrier(cmd, swap, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageCopy copy{}; copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.extent = {extent.width, extent.height, 1};
    vkCmdCopyImage(cmd, s.hdr.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swap, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
}
} // namespace
void ToSwapchain(VkCommandBuffer cmd, VkImageLayout layout, VkImage swap, VkExtent2D extent) {
    Barrier(cmd, s.composed.image, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    Convert(cmd, s.composed.view, s.set, swap, extent); // (the presenting thread's set: its last use has completed, the fence waited for)
}

namespace fg {
namespace {
struct Fg { Image compose, frame[Slots], made[MostMade]; VkExtent2D extent{}; bool fresh = true, failed = false; } f;
void Show(VkCommandBuffer cmd, Image& src, VkImage swap, VkExtent2D extent) { // (src in GENERAL, and left so)
    if (s.surface.hdr && Initialize()) {
        static VkDescriptorPool pool = VK_NULL_HANDLE; static VkDescriptorSet sets[16]{}; static int next = 0;
        if (!pool) { auto& vk = Vk();
            const VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 16; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
            VK_CHECK(vkCreateDescriptorPool(vk.device, &pi, nullptr, &pool));
            for (auto& set : sets) { VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; sa.descriptorPool = pool; sa.descriptorSetCount = 1; sa.pSetLayouts = &s.set_layout; VK_CHECK(vkAllocateDescriptorSets(vk.device, &sa, &set)); } }
        Barrier(cmd, src.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        Convert(cmd, src.view, sets[next++ % 16], swap, extent); // (a set is used again 16 pictures later: long shown by then)
        Barrier(cmd, src.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        return;
    }
    VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; written.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; written.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &written, 0, nullptr, 0, nullptr);
    Barrier(cmd, swap, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{}; blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = blit.dstOffsets[1] = {(s32)extent.width, (s32)extent.height, 1};
    vkCmdBlitImage(cmd, src.image, VK_IMAGE_LAYOUT_GENERAL, swap, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST); // (same size: the channel order alone changes)
}
} // namespace

int Multiplier(double game_fps) {
    static const char* forced = std::getenv("OOT_FRAME_GEN"); // (a multiplier, for tests)
    const int choice = forced ? std::atoi(forced) : gfx::Current().frame_gen;
    if (choice <= 1) return 1;
    if (const int k = nvn::DynamicFrames(); k > 0 && !forced) return k; // (Dynamic with a frame cap: present.cpp)
    // Never more than the display shows: frames made past its refresh rate are only work (and where the display paces
    // the presents, a brake). Dynamic is just that limit, six at most.
    const double hz = gfx::display_refresh.load(std::memory_order_relaxed) > 30 ? gfx::display_refresh.load(std::memory_order_relaxed) : 0.0;
    const int fits = hz > 0 && game_fps > 5 && !forced ? std::max(1, (int)(hz * 1.04 / game_fps)) : 6;
    return std::min(choice <= 6 ? choice : 6, fits);
}
bool Size(VkExtent2D extent, VkFormat composition) {
    if (f.failed) return false;
    if (f.extent.width == extent.width && f.extent.height == extent.height && f.compose.format == composition) return true;
    { VkFormatProperties p; vkGetPhysicalDeviceFormatProperties(Vk().physical, VK_FORMAT_R8G8B8A8_UNORM, &p);
      if (!(p.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) || !(p.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) { f.failed = true; RT_LOG("frame generation: this GPU cannot write the pictures it needs"); return false; } }
    Ensure(f.compose, extent, composition, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const VkImageUsageFlags picture = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    for (auto& i : f.frame) Ensure(i, extent, VK_FORMAT_R8G8B8A8_UNORM, picture);
    for (auto& i : f.made) Ensure(i, extent, VK_FORMAT_R8G8B8A8_UNORM, picture);
    f.extent = extent; f.fresh = true;
    return true;
}
VkImage Compose() { return f.compose.image; }
void Stored(VkCommandBuffer cmd, int slot, VkImageLayout layout) {
    if (f.fresh) { f.fresh = false;
        for (auto& i : f.frame) Barrier(cmd, i.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        for (auto& i : f.made) Barrier(cmd, i.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT); }
    Barrier(cmd, f.compose.image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT; before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    VkImageBlit blit{}; blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = blit.dstOffsets[1] = {(s32)f.extent.width, (s32)f.extent.height, 1};
    vkCmdBlitImage(cmd, f.compose.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f.frame[slot].image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
}
int Generate(VkCommandBuffer cmd, int slot, int count, bool blend, uint64_t sequence) {
    count = std::clamp(count, 0, (int)MostMade);
    VkImage images[MostMade]; VkImageView views[MostMade];
    for (int i = 0; i < MostMade; ++i) { images[i] = f.made[i].image; views[i] = f.made[i].view; }
    return ls::MakeFrames(cmd, f.frame[slot].image, f.frame[slot].view, f.extent.width, f.extent.height, count, images, views, blend, sequence);
}
void ShowMade(VkCommandBuffer cmd, int i, VkImage swap, VkExtent2D extent) { Show(cmd, f.made[i], swap, extent); }
void ShowSlot(VkCommandBuffer cmd, int slot, VkImage swap, VkExtent2D extent) { Show(cmd, f.frame[slot], swap, extent); }
} // namespace fg
} // namespace nvn::output
