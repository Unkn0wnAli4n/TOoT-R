#include "../graphics_settings.h"
// NVN textures: the object keeps the texture builder bytes (so builder getters serve as texture
// getters) and the host texture pointer at +TextureBuilderSize.

#include "../../native.h"
#include "memory.h"
#include <list>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cstdlib>
#include "texture.h"
#include "batch.h"

using namespace rt;
using namespace nvn;

namespace nvn {
namespace {
std::atomic<u64> image_live_bytes{0},image_cached_bytes{0};
// GPU-written images kept for re-created texture objects (history buffers, transient targets).
// 256 MiB / 512 entries evicted this game's per-frame targets (2560x1440 RGBA16F alone is 29 MiB):
// temporal effects (volume fog, anti-aliasing and exposure history) restarted from empty images.
const u64 ImageCacheBudget=std::getenv("OOT_IMAGE_CACHE_MB")?std::strtoull(std::getenv("OOT_IMAGE_CACHE_MB"),nullptr,10)<<20:512ull<<20;
const size_t ImageCacheEntries=std::getenv("OOT_IMAGE_CACHE_ENTRIES")?std::strtoull(std::getenv("OOT_IMAGE_CACHE_ENTRIES"),nullptr,10):1024;
std::atomic<u64> image_cache_adopted{0},image_cache_evicted{0},image_astc_bytes{0};
}

u64 DriverFunction(const char* name);

FormatInfo GetFormat(u32 f) {
    // NVNformat (NVN1 numbering, confirmed against the SDK driver's format table).
    struct Entry { VkFormat vk; u32 bw, bh, bytes; };
    static const Entry table[] = {
        /*0*/ {VK_FORMAT_UNDEFINED, 1, 1, 0},
        {VK_FORMAT_R8_UNORM, 1, 1, 1}, {VK_FORMAT_R8_SNORM, 1, 1, 1}, {VK_FORMAT_R8_UINT, 1, 1, 1}, {VK_FORMAT_R8_SINT, 1, 1, 1},
        {VK_FORMAT_R16_SFLOAT, 1, 1, 2}, {VK_FORMAT_R16_UNORM, 1, 1, 2}, {VK_FORMAT_R16_SNORM, 1, 1, 2}, {VK_FORMAT_R16_UINT, 1, 1, 2}, {VK_FORMAT_R16_SINT, 1, 1, 2},
        /*10*/ {VK_FORMAT_R32_SFLOAT, 1, 1, 4}, {VK_FORMAT_R32_UINT, 1, 1, 4}, {VK_FORMAT_R32_SINT, 1, 1, 4},
        {VK_FORMAT_R8G8_UNORM, 1, 1, 2}, {VK_FORMAT_R8G8_SNORM, 1, 1, 2}, {VK_FORMAT_R8G8_UINT, 1, 1, 2}, {VK_FORMAT_R8G8_SINT, 1, 1, 2},
        {VK_FORMAT_R16G16_SFLOAT, 1, 1, 4}, {VK_FORMAT_R16G16_UNORM, 1, 1, 4}, {VK_FORMAT_R16G16_SNORM, 1, 1, 4},
        /*20*/ {VK_FORMAT_R16G16_UINT, 1, 1, 4}, {VK_FORMAT_R16G16_SINT, 1, 1, 4},
        {VK_FORMAT_R32G32_SFLOAT, 1, 1, 8}, {VK_FORMAT_R32G32_UINT, 1, 1, 8}, {VK_FORMAT_R32G32_SINT, 1, 1, 8},
        /*25 RGB8 (3 bytes)*/ {VK_FORMAT_R8G8B8_UNORM, 1, 1, 3}, {VK_FORMAT_R8G8B8_SNORM, 1, 1, 3}, {VK_FORMAT_R8G8B8_UINT, 1, 1, 3}, {VK_FORMAT_R8G8B8_SINT, 1, 1, 3},
        /*29 RGB16*/ {VK_FORMAT_R16G16B16_SFLOAT, 1, 1, 6},
        /*30*/ {VK_FORMAT_R16G16B16_UNORM, 1, 1, 6}, {VK_FORMAT_R16G16B16_SNORM, 1, 1, 6}, {VK_FORMAT_R16G16B16_UINT, 1, 1, 6}, {VK_FORMAT_R16G16B16_SINT, 1, 1, 6},
        /*34 RGB32*/ {VK_FORMAT_R32G32B32_SFLOAT, 1, 1, 12}, {VK_FORMAT_R32G32B32_UINT, 1, 1, 12}, {VK_FORMAT_R32G32B32_SINT, 1, 1, 12},
        /*37*/ {VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_SNORM, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_UINT, 1, 1, 4},
        /*40*/ {VK_FORMAT_R8G8B8A8_SINT, 1, 1, 4},
        {VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_UNORM, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_SNORM, 1, 1, 8},
        {VK_FORMAT_R16G16B16A16_UINT, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_SINT, 1, 1, 8},
        {VK_FORMAT_R32G32B32A32_SFLOAT, 1, 1, 16}, {VK_FORMAT_R32G32B32A32_UINT, 1, 1, 16}, {VK_FORMAT_R32G32B32A32_SINT, 1, 1, 16},
        /*49 STENCIL8*/ {VK_FORMAT_S8_UINT, 1, 1, 1},
        /*50*/ {VK_FORMAT_D16_UNORM, 1, 1, 2}, {VK_FORMAT_D24_UNORM_S8_UINT, 1, 1, 4}, {VK_FORMAT_D32_SFLOAT, 1, 1, 4},
        {VK_FORMAT_D24_UNORM_S8_UINT, 1, 1, 4}, {VK_FORMAT_D32_SFLOAT_S8_UINT, 1, 1, 8},
        /*55 RGBX8_SRGB*/ {VK_FORMAT_R8G8B8A8_SRGB, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_SRGB, 1, 1, 4},
        /*57 RGBA4*/ {VK_FORMAT_A4B4G4R4_UNORM_PACK16, 1, 1, 2},
        /*58 RGB5*/ {VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR, 1, 1, 2}, {VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR, 1, 1, 2},
        /*60 RGB565*/ {VK_FORMAT_B5G6R5_UNORM_PACK16, 1, 1, 2},
        /*61*/ {VK_FORMAT_A2B10G10R10_UNORM_PACK32, 1, 1, 4}, {VK_FORMAT_A2B10G10R10_UINT_PACK32, 1, 1, 4},
        {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 1, 1, 4}, {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 1, 1, 4},
        /*65*/ {VK_FORMAT_BC1_RGB_UNORM_BLOCK, 4, 4, 8}, {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, 8},
        {VK_FORMAT_BC2_UNORM_BLOCK, 4, 4, 16}, {VK_FORMAT_BC3_UNORM_BLOCK, 4, 4, 16},
        {VK_FORMAT_BC1_RGB_SRGB_BLOCK, 4, 4, 8}, {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, 4, 4, 8},
        {VK_FORMAT_BC2_SRGB_BLOCK, 4, 4, 16}, {VK_FORMAT_BC3_SRGB_BLOCK, 4, 4, 16},
        /*73*/ {VK_FORMAT_BC4_UNORM_BLOCK, 4, 4, 8}, {VK_FORMAT_BC4_SNORM_BLOCK, 4, 4, 8},
        {VK_FORMAT_BC5_UNORM_BLOCK, 4, 4, 16}, {VK_FORMAT_BC5_SNORM_BLOCK, 4, 4, 16},
        /*77*/ {VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, 16}, {VK_FORMAT_BC7_SRGB_BLOCK, 4, 4, 16},
        {VK_FORMAT_BC6H_SFLOAT_BLOCK, 4, 4, 16}, {VK_FORMAT_BC6H_UFLOAT_BLOCK, 4, 4, 16},
    };
    FormatInfo info;
    if (f < sizeof table / sizeof table[0]) {
        info.vk = table[f].vk;
        info.block_w = table[f].bw;
        info.block_h = table[f].bh;
        info.bytes = table[f].bytes;
    } else if (f >= 81 && f <= 108) { // integer-to-float (vertex) formats and RGB10A2 variants
        static const Entry conv[] = {
            /*81*/ {VK_FORMAT_R8_USCALED, 1, 1, 1}, {VK_FORMAT_R8_SSCALED, 1, 1, 1}, {VK_FORMAT_R16_USCALED, 1, 1, 2}, {VK_FORMAT_R16_SSCALED, 1, 1, 2},
            /*85*/ {VK_FORMAT_R32_UINT, 1, 1, 4}, {VK_FORMAT_R32_SINT, 1, 1, 4}, {VK_FORMAT_R8G8_USCALED, 1, 1, 2}, {VK_FORMAT_R8G8_SSCALED, 1, 1, 2},
            /*89*/ {VK_FORMAT_R16G16_USCALED, 1, 1, 4}, {VK_FORMAT_R16G16_SSCALED, 1, 1, 4}, {VK_FORMAT_R32G32_UINT, 1, 1, 8}, {VK_FORMAT_R32G32_SINT, 1, 1, 8},
            /*93*/ {VK_FORMAT_R8G8B8_USCALED, 1, 1, 3}, {VK_FORMAT_R8G8B8_SSCALED, 1, 1, 3}, {VK_FORMAT_R16G16B16_USCALED, 1, 1, 6}, {VK_FORMAT_R16G16B16_SSCALED, 1, 1, 6},
            /*97*/ {VK_FORMAT_R32G32B32_UINT, 1, 1, 12}, {VK_FORMAT_R32G32B32_SINT, 1, 1, 12}, {VK_FORMAT_R8G8B8A8_USCALED, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_SSCALED, 1, 1, 4},
            /*101*/ {VK_FORMAT_R16G16B16A16_USCALED, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_SSCALED, 1, 1, 8}, {VK_FORMAT_R32G32B32A32_UINT, 1, 1, 16}, {VK_FORMAT_R32G32B32A32_SINT, 1, 1, 16},
            /*105*/ {VK_FORMAT_A2B10G10R10_SNORM_PACK32, 1, 1, 4}, {VK_FORMAT_A2B10G10R10_SINT_PACK32, 1, 1, 4},
            /*107*/ {VK_FORMAT_A2B10G10R10_USCALED_PACK32, 1, 1, 4}, {VK_FORMAT_A2B10G10R10_SSCALED_PACK32, 1, 1, 4}};
        const Entry& e = conv[f - 81];
        info = {e.vk, e.bw, e.bh, e.bytes};
    } else if (f >= 109 && f <= 120) { // RGBX: as RGBA
        static const Entry rgbx[] = {
            {VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_SNORM, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_UINT, 1, 1, 4}, {VK_FORMAT_R8G8B8A8_SINT, 1, 1, 4},
            {VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_UNORM, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_SNORM, 1, 1, 8},
            {VK_FORMAT_R16G16B16A16_UINT, 1, 1, 8}, {VK_FORMAT_R16G16B16A16_SINT, 1, 1, 8},
            {VK_FORMAT_R32G32B32A32_SFLOAT, 1, 1, 16}, {VK_FORMAT_R32G32B32A32_UINT, 1, 1, 16}, {VK_FORMAT_R32G32B32A32_SINT, 1, 1, 16}};
        const Entry& e = rgbx[f - 109];
        info = {e.vk, e.bw, e.bh, e.bytes};
    } else if (f >= 121 && f <= 148) { // ASTC
        static const VkFormat astc[] = {
            VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_5x4_UNORM_BLOCK, VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
            VK_FORMAT_ASTC_6x5_UNORM_BLOCK, VK_FORMAT_ASTC_6x6_UNORM_BLOCK, VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
            VK_FORMAT_ASTC_8x6_UNORM_BLOCK, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
            VK_FORMAT_ASTC_10x6_UNORM_BLOCK, VK_FORMAT_ASTC_10x8_UNORM_BLOCK, VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
            VK_FORMAT_ASTC_12x10_UNORM_BLOCK, VK_FORMAT_ASTC_12x12_UNORM_BLOCK};
        static const u8 dims[][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6}, {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
        const u32 i = (f - 121) % 14;
        info.vk = (VkFormat)(astc[i] + ((f >= 135) ? 1 : 0)); // the sRGB variant follows each UNORM one
        info.block_w = dims[i][0];
        info.block_h = dims[i][1];
        info.bytes = 16;
    } else if (f >= 149 && f <= 156) {
        static const Entry bgr[] = {
            {VK_FORMAT_R5G6B5_UNORM_PACK16, 1, 1, 2}, {VK_FORMAT_A1R5G5B5_UNORM_PACK16, 1, 1, 2},
            {VK_FORMAT_A1R5G5B5_UNORM_PACK16, 1, 1, 2}, {VK_FORMAT_R5G5B5A1_UNORM_PACK16, 1, 1, 2},
            {VK_FORMAT_B8G8R8A8_UNORM, 1, 1, 4}, {VK_FORMAT_B8G8R8A8_UNORM, 1, 1, 4},
            {VK_FORMAT_B8G8R8A8_SRGB, 1, 1, 4}, {VK_FORMAT_B8G8R8A8_SRGB, 1, 1, 4}};
        const Entry& e = bgr[f - 149];
        info = {e.vk, e.bw, e.bh, e.bytes};
    } else if (f == 161 || f == 162) {
        info.vk = f == 161 ? VK_FORMAT_R64_UINT : VK_FORMAT_R64_SINT;
        info.bytes = 8;
    } else {
        Fatal("NVN texture format %u is not mapped", f);
    }
    info.depth = info.vk == VK_FORMAT_D16_UNORM || info.vk == VK_FORMAT_D24_UNORM_S8_UINT ||
                 info.vk == VK_FORMAT_D32_SFLOAT || info.vk == VK_FORMAT_D32_SFLOAT_S8_UINT;
    info.stencil = info.vk == VK_FORMAT_S8_UINT || info.vk == VK_FORMAT_D24_UNORM_S8_UINT ||
                   info.vk == VK_FORMAT_D32_SFLOAT_S8_UINT;
    return info;
}

Texture* HostTexture(u64 object) { return *reinterpret_cast<Texture**>(object + TextureBuilderSize); }
namespace {
std::mutex texture_owners_mutex;
std::unordered_map<u64,std::shared_ptr<Texture>> texture_owners;
void RetireDescriptorImage(Texture& texture) {
    if(!texture.image)return;
    // Descriptors retain immutable texture metadata. GPU allocations return to
    // the bounded cache after the API owner (or this submission) releases them.
    auto* image=new Texture(texture);
    image->owner_finalized=false;image->gpu_retirement_pending=false;
    texture.image=VK_NULL_HANDLE;texture.memory=VK_NULL_HANDLE;texture.view=VK_NULL_HANDLE;
    texture.views.clear();texture.allocation_size=0;
    texture.layout=VK_IMAGE_LAYOUT_UNDEFINED;texture.uploaded=texture.gpu_written=false;
    RetireTexture(image);
}
}
void InstallTexture(u64 object,Texture* texture) {
    auto owner=std::shared_ptr<Texture>(texture,RetireTexture);
    std::shared_ptr<Texture> retired;
    {
        std::lock_guard lock{texture_owners_mutex};
        retired=std::move(texture_owners[object]);texture_owners[object]=std::move(owner);
        *reinterpret_cast<Texture**>(object+TextureBuilderSize)=texture;
    }
}
std::shared_ptr<Texture> TextureReference(u64 object) {
    std::lock_guard lock{texture_owners_mutex};
    const auto it=texture_owners.find(object);
    if(it==texture_owners.end())Fatal("registering an uninitialized texture object %#llx",(unsigned long long)object);
    return it->second;
}
void FinalizeTexture(u64 object) {
    std::lock_guard batch_lock{BatchMutex()};
    std::shared_ptr<Texture> retired;
    {
        std::lock_guard lock{texture_owners_mutex};
        const auto it=texture_owners.find(object);
        if(it!=texture_owners.end()){retired=std::move(it->second);texture_owners.erase(it);}
        *reinterpret_cast<Texture**>(object+TextureBuilderSize)=nullptr;
    }
    if(retired){retired->owner_finalized=true;RetireDescriptorImage(*retired);}
}

VkImageAspectFlags AspectOf(const Texture& t) {
    VkImageAspectFlags a = 0;
    if (t.info.depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (t.info.stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VK_IMAGE_ASPECT_COLOR_BIT;
}

namespace {
VkImageView MakeView(Texture& t, TextureView view, bool attachment) {
    EnsureImage(t);
    if(!attachment){
        bool custom=false;for(auto component:view.swizzle)custom|=component!=0;
        if(!custom)std::memcpy(view.swizzle,t.swizzle,sizeof view.swizzle);
        if(!view.depth_stencil_mode)view.depth_stencil_mode=t.depth_stencil_mode+1;
    }
    if (attachment) {
        view.present = true; view.levels_count = 1; view.target = (u32)Target::T2DArray + 1;
        // Attachment views must use identity component mapping.
        view.swizzle[0] = 0; view.swizzle[1] = 1; view.swizzle[2] = 2; view.swizzle[3] = 3;
    }
    for (const auto& [key, cached] : t.views)
        if (!std::memcmp(&key, &view, sizeof view)) return cached;
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = t.image; info.format = view.format && view.format != t.format ? GetFormat(view.format).vk : t.info.vk;
    if (t.decoded_astc && view.format >= 121 && view.format <= 148) // (an ASTC view of a host-decoded image: its sRGB-ness)
        info.format = t.bc3 ? (view.format >= 135 ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK)
                            : (view.format >= 135 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
    const Target target = view.target ? (Target)(view.target - 1) : t.target;
    switch (target) {
    case Target::T1D: info.viewType = VK_IMAGE_VIEW_TYPE_1D; break;
    case Target::T1DArray: info.viewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY; break;
    case Target::T3D: info.viewType = VK_IMAGE_VIEW_TYPE_3D; break;
    case Target::T2DArray: case Target::T2DMultisampleArray: info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; break;
    case Target::Cube: info.viewType = VK_IMAGE_VIEW_TYPE_CUBE; break;
    case Target::CubeArray: info.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY; break;
    default: info.viewType = VK_IMAGE_VIEW_TYPE_2D; break;
    }
    info.subresourceRange = {AspectOf(t), view.levels_base,
        view.levels_count ? view.levels_count : t.levels - view.levels_base,
        view.layers_base, view.layers_count ? view.layers_count : t.layers - view.layers_base};
    if (!attachment) {
        if (t.info.depth || t.info.stencil) {
            const u32 mode = view.depth_stencil_mode ? view.depth_stencil_mode - 1 : t.depth_stencil_mode;
            info.subresourceRange.aspectMask = mode == 1 && t.info.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT :
                t.info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        // NVNtextureSwizzle: ZERO, ONE, R, G, B, A.
        const VkComponentSwizzle swizzles[] = {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        VkComponentSwizzle* components[] = {&info.components.r, &info.components.g, &info.components.b, &info.components.a};
        bool custom = false; for (auto s : view.swizzle) custom |= s != 0;
        for (u32 i = 0; i < 4; ++i) {
            const u32 s = custom ? view.swizzle[i] : t.swizzle[i];
            if (s >= 6) Fatal("invalid texture component swizzle %u", s);
            *components[i] = swizzles[s];
        }
    }
    // Views only claim the usages their own format supports (sRGB views cannot be storage images).
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(Vk().physical, info.format, &fp);
    const VkFormatFeatureFlags features = fp.optimalTilingFeatures;
    VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) usage.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) usage.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) usage.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && t.samples == 1) usage.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    usage.usage &= t.usage;
    info.pNext = &usage;
    VkImageView result; VK_CHECK(vkCreateImageView(Vk().device, &info, nullptr, &result));
    t.views.emplace_back(view, result); return result;
}
}
VkImageView ImageView(Texture& t, const TextureView& view) { return MakeView(t, view, false); }
VkImageView AttachmentView(Texture& t, const TextureView& view) { return MakeView(t, view, true); }

void Transition(VkCommandBuffer cmd, Texture& t, VkImageLayout layout) {
    EnsureImage(t);
    // A layout can stay GENERAL across attachment writes and shader reads;
    // those uses still need a memory dependency between submissions.
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = t.layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = t.layout;
    b.newLayout = layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = {AspectOf(t), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    t.layout = layout;
}

namespace {

void CreateImage(Texture& t) {
    auto& vk = Vk();
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk.physical, t.info.vk, &fp);
    if (t.format >= 121 && t.format <= 148 && !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        t.decoded_astc = true;
        t.info.vk = t.format >= 135 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        vkGetPhysicalDeviceFormatProperties(vk.physical,t.info.vk,&fp);
        // (from Hookshot, vk_texture.cpp: big ASTC as BC3. RGBA8 took GBs of VRAM in Kokiri
        // Forest (3.6 GiB here) and ran the 12 GB card out of memory. 2D, at most 2048 square,
        // blocks of 6x6 texels or more from 2 MiB as RGBA8, finer ones (UI, characters) only from
        // 64 MiB. OOT_ASTC_RGBA8=1: all as RGBA8.)
        // (the Graphics screen's Texture Memory: Low re-encodes from 256 KiB, Quality keeps RGBA8)
        const bool rgba8=std::getenv("OOT_ASTC_RGBA8")||std::getenv("OOT_CPU_ASTC")||gfx::Current().texture_memory==2;
        const bool low=gfx::Current().texture_memory==1;
        if(!rgba8&&t.target!=Target::T3D&&t.samples==1&&t.width<=2048&&t.height<=2048){
            u64 px=0;for(u32 l=0;l<t.levels;++l)px+=(u64)std::max(1u,t.width>>l)*std::max(1u,t.height>>l);
            const u64 min_bytes=low?(256ull<<10):t.info.block_w*t.info.block_h>=36?(2ull<<20):(64ull<<20);
            VkFormatProperties bc;vkGetPhysicalDeviceFormatProperties(vk.physical,VK_FORMAT_BC3_UNORM_BLOCK,&bc);
            if(px*t.layers*4>=min_bytes&&(bc.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)){
                t.bc3=true;t.info.vk=t.format>=135?VK_FORMAT_BC3_SRGB_BLOCK:VK_FORMAT_BC3_UNORM_BLOCK;
                vkGetPhysicalDeviceFormatProperties(vk.physical,t.info.vk,&fp);
            }
        }
    }
    const VkFormatFeatureFlags features = fp.optimalTilingFeatures;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = t.target == Target::T3D ? VK_IMAGE_TYPE_3D
                 : (t.target == Target::T1D || t.target == Target::T1DArray) ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
    ii.format = t.info.vk;
    ii.extent = {t.width, t.height, t.target == Target::T3D ? t.depth : 1};
    ii.mipLevels = t.levels;
    ii.arrayLayers = t.target == Target::T3D ? 1 : t.layers;
    ii.samples = (VkSampleCountFlagBits)t.samples;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ii.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ii.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ii.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && t.samples == 1) ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    // Games reinterpret textures through same-size views (e.g. RGB9E5 written as R32_UINT), so
    // color images allow the usages any such view could need; compressed images also
    // have uncompressed block views for GPU block generation. Each view restricts
    // its own usage to what its format supports (MakeView).
    if (t.decoded_astc && !t.bc3 && t.samples == 1) ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT; // (GPU decoding through a UNORM view)
    if (!t.info.depth && !t.info.stencil && !t.decoded_astc) {
        ii.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (t.samples == 1) ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    ii.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    if ((t.target == Target::Cube || t.target == Target::CubeArray) && t.layers % 6 == 0)
        ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (t.target == Target::T3D) ii.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    if (t.info.block_w > 1 && !t.decoded_astc) ii.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    t.usage = ii.usage;
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &t.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(vk.device, t.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkResult allocated=vkAllocateMemory(vk.device,&ai,nullptr,&t.memory);
    if(allocated!=VK_SUCCESS){
        // Out of device memory: release every cached image (destroyed after the GPU is done with
        // them), then retry; as a last resort use any memory type the image allows (slower).
        RT_LOG("texture allocation failed (%llu MiB requested; live %llu MiB, cached %llu MiB): releasing cached images",
            (unsigned long long)(req.size>>20),(unsigned long long)(image_live_bytes.load()>>20),(unsigned long long)(image_cached_bytes.load()>>20));
        // (No Flush here: CreateImage runs inside Immediate recordings (Transition -> EnsureImage), and
        // submitting the command buffer being recorded crashed the driver. Released images free
        // after the current batch; until then this allocation falls back to other memory.)
        TrimImageCache(0);
        allocated=vkAllocateMemory(vk.device,&ai,nullptr,&t.memory);
        if(allocated!=VK_SUCCESS){ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,0);allocated=vkAllocateMemory(vk.device,&ai,nullptr,&t.memory);
            if(allocated==VK_SUCCESS)RT_LOG("texture placed in non-device-local memory");}
    }
    if(allocated!=VK_SUCCESS)RT_LOG("texture allocation failed: %ux%ux%u format %u, requested %llu MiB; live images %llu MiB, cached %llu MiB",t.width,t.height,t.depth,t.format,
        (unsigned long long)(req.size>>20),(unsigned long long)(image_live_bytes.load()>>20),(unsigned long long)(image_cached_bytes.load()>>20));
    VK_CHECK(allocated);
    t.allocation_size=req.size;image_live_bytes+=req.size;if(t.decoded_astc)image_astc_bytes+=req.size;
    VK_CHECK(vkBindImageMemory(vk.device, t.image, t.memory, 0));
}

} // namespace
void EnsureImage(Texture& t) {
    // A finalized texture still used by recorded work keeps its image until that batch. One no shared owner
    // holds any more (a stale reference to an object on its way out, destroyed with its image) is not
    // retired again.
    std::shared_ptr<Texture> self=t.weak_from_this().lock();
    if(t.owner_finalized&&!self){
        static std::atomic<u32> told{0};
        if(told++<5)RT_LOG("texture %#llx used after its last owner released it",(unsigned long long)t.object);
    }
    if(!t.image)AdoptCachedImage(t); // a descriptor replacement may have released it since initialization
    if(!t.image)CreateImage(t);
    if(t.owner_finalized&&self&&!t.gpu_retirement_pending){
        t.gpu_retirement_pending=true;
        AfterBatch([texture=std::move(self)] {
            texture->gpu_retirement_pending=false;
            RetireDescriptorImage(*texture);
        });
    }
}
} // namespace nvn

// ---- host image recycling ---------------------------------------------------------------------------
// Engines re-create texture objects over the same memory every frame (transient targets, history
// buffers). A finalized texture whose contents the GPU produced keeps its host image here, keyed by
// storage and description, so the next texture created identically continues with it (contents
// preserved, no CPU upload of stale memory). CPU-uploaded textures are always uploaded afresh.
namespace nvn {
namespace {
using ImageKey = std::tuple<u64, u32, u32, u32, u32, u32, u32, u32, u32>;
ImageKey KeyOf(const Texture& t) {
    return {t.storage, t.format, t.width, t.height, t.depth, t.levels, t.layers, t.samples, (u32)t.target};
}
std::mutex cache_mutex;
std::list<ImageKey> cache_order;
struct CachedImage {
    std::unique_ptr<Texture> texture;
    std::list<ImageKey>::iterator position;
};
std::map<ImageKey,CachedImage> image_cache;

void DestroyLater(Texture* t) {
    AfterFlush([t] {
        auto& vk = Vk();
        if (t->view) vkDestroyImageView(vk.device, t->view, nullptr);
        for (auto& [key, view] : t->views) vkDestroyImageView(vk.device, view, nullptr);
        if (t->image) vkDestroyImage(vk.device, t->image, nullptr);
        if (t->memory) {vkFreeMemory(vk.device, t->memory, nullptr);image_live_bytes-=t->allocation_size;if(t->decoded_astc)image_astc_bytes-=t->allocation_size;}
        delete t;
    });
}
} // namespace

// (cache_mutex held)
void EvictOldest() {
    auto old = image_cache.find(cache_order.front());
    if (old != image_cache.end()) { image_cached_bytes-=old->second.texture->allocation_size;DestroyLater(old->second.texture.release()); image_cache.erase(old); ++image_cache_evicted; }
    cache_order.pop_front();
}
void AdoptCachedImage(Texture& t) {
    std::lock_guard lock{cache_mutex};
    auto it = image_cache.find(KeyOf(t));
    if (it == image_cache.end()) return;
    Texture& cached = *it->second.texture;
    t.image = cached.image; t.memory = cached.memory; t.view = cached.view; t.views = std::move(cached.views);
    t.allocation_size=cached.allocation_size;image_cached_bytes-=cached.allocation_size;
    t.layout = cached.layout; t.usage = cached.usage; t.gpu_written = cached.gpu_written; t.uploaded = cached.uploaded;
    t.decoded_astc = cached.decoded_astc; t.bc3 = cached.bc3; t.decoded_rgb9e5 = cached.decoded_rgb9e5; t.info = cached.info;
    cached.image = VK_NULL_HANDLE; cached.memory = VK_NULL_HANDLE; cached.view = VK_NULL_HANDLE; cached.views.clear();
    cache_order.erase(it->second.position);
    image_cache.erase(it);
    ++image_cache_adopted;
}

void RetireTexture(Texture* t) {
    if (!t->image || !t->gpu_written) { DestroyLater(t); return; }
    std::lock_guard lock{cache_mutex};
    const ImageKey key = KeyOf(*t);
    auto it=image_cache.find(key);
    if(it!=image_cache.end()) {
        image_cached_bytes-=it->second.texture->allocation_size;
        DestroyLater(it->second.texture.release());
        cache_order.splice(cache_order.end(),cache_order,it->second.position);
        it->second.texture.reset(t);
    } else {
        cache_order.push_back(key);
        image_cache.emplace(key,CachedImage{std::unique_ptr<Texture>(t),std::prev(cache_order.end())});
    }
    image_cached_bytes+=t->allocation_size;
    t->object = 0;
    while (image_cache.size() > ImageCacheEntries||image_cached_bytes>ImageCacheBudget) EvictOldest();
}
void TrimImageCache(u64 keep_bytes) {
    std::lock_guard lock{cache_mutex};
    while (!cache_order.empty() && image_cached_bytes > keep_bytes) EvictOldest();
}
} // namespace nvn
namespace nvn {
void ReportImageCache() {
    std::lock_guard lock{cache_mutex};
    RT_LOG("image cache: %llu adopted, %llu evicted; %zu cached (%.0f MiB), %.0f MiB live (%.0f MiB decoded ASTC)",
           (unsigned long long)image_cache_adopted.exchange(0), (unsigned long long)image_cache_evicted.exchange(0),
           image_cache.size(), image_cached_bytes / 1048576.0, image_live_bytes / 1048576.0, image_astc_bytes / 1048576.0);
}
} // namespace nvn
using nvn::AdoptCachedImage; using nvn::RetireTexture;

// NVNtextureBuilder: +0x10 memory pool, +0x18 offset, +0x20 flags, +0x24 target, +0x28 width,
// +0x2c height, +0x30 depth, +0x34 levels, +0x38 format, +0x3c samples, +0x40 swizzle,
// +0x50 depth-stencil mode, +0x58 stride.
NATIVE("nvnTextureInitialize") {
    nvn::binding_generation.fetch_add(1, std::memory_order_relaxed);
    const u8* b = Ptr<u8>(c, 1);
    std::memcpy(Ptr(c, 0), b, TextureBuilderSize);
    auto* t = new Texture{};
    t->object = c->x[0];
    u64 pool, offset;
    std::memcpy(&pool, b + 0x10, 8);
    std::memcpy(&offset, b + 0x18, 8);
    std::memcpy(&t->flags, b + 0x20, 4);
    u32 target;
    std::memcpy(&target, b + 0x24, 4);
    t->target = (Target)target;
    std::memcpy(&t->width, b + 0x28, 4);
    std::memcpy(&t->height, b + 0x2c, 4);
    std::memcpy(&t->depth, b + 0x30, 4);
    std::memcpy(&t->levels, b + 0x34, 4);
    std::memcpy(&t->format, b + 0x38, 4);
    std::memcpy(&t->samples, b + 0x3c, 4);
    std::memcpy(t->swizzle,b+0x40,16);
    std::memcpy(&t->depth_stencil_mode,b+0x50,4);
    if (!t->width) t->width = 1;
    if (!t->height) t->height = 1;
    if (!t->depth) t->depth = 1;
    if (!t->levels) t->levels = 1;
    if (!t->samples) t->samples = 1;
    switch (t->target) {
    case Target::T1DArray: t->layers = t->height; t->height = 1; break;
    case Target::T2DArray: case Target::T2DMultisampleArray: case Target::CubeArray: t->layers = t->depth; t->depth = 1; break;
    case Target::Cube: t->layers = 6; t->depth = 1; break;
    case Target::T3D: break;
    default: t->depth = 1; break;
    }
    t->storage = pool ? PoolOf(pool)->address + offset : 0;
    t->info = GetFormat(t->format);
    std::memcpy(&t->stride, b + 0x58, 4);
    if (t->target == Target::Buffer) Fatal("buffer textures are not implemented yet");
    const u64 scratch = ArenaAlloc(0x50);
    // The driver's layout helper computes GOB dimensions and exact layer strides.
    // Its output is independent of the console's GPU service objects.
    u64 sdk_base = 0;
    for (const auto& module : Modules()) if (module.name == "sdk") sdk_base = module.base;
    CallGuest(c, sdk_base + 0x66b190, {scratch, t->object});
    const auto* layout = reinterpret_cast<const u8*>(scratch);
    u64 packed;
    std::memcpy(&packed, layout, 8); t->layer_stride = packed >> 16;
    std::memcpy(&packed, layout + 0x10, 8); t->storage_size = packed >> 16;
    t->block_width = layout[0x28];
    for (u32 level = 0; level < t->levels; ++level) {
        const u64 view_bits = ((u64)level << 32) | (1ULL << 40) | (1ULL << 56);
        std::memcpy(reinterpret_cast<void*>(scratch + 0x30), &view_bits, 8);
        const u64 mip_offset = CallGuest(c, DriverFunction("nvnTextureBuilderGetViewOffset"), {t->object, scratch + 0x30});
        const u32 w = std::max(t->width >> level, 1u), h = std::max(t->height >> level, 1u);
        const u32 d = t->target == Target::T3D ? std::max(t->depth >> level, 1u) : 1;
        u32 bh = layout[0x29], bd = layout[0x2a];
        const u32 blocks_h = (h + t->info.block_h - 1) / t->info.block_h;
        if (t->levels > 1) {
            while (bh && blocks_h <= (8u << (bh - 1))) --bh;
            while (bd && d <= (1u << (bd - 1))) --bd;
        }
        const u64 row_bytes = (u64)((w + t->info.block_w - 1) / t->info.block_w) * t->info.bytes;
        const u64 gob_alignment = 64ULL << t->block_width;
        const u64 mip_size = t->flags & 0x50 ? (u64)t->stride * blocks_h * d :
            ((row_bytes + gob_alignment - 1) & ~(gob_alignment - 1)) *
            ((blocks_h + (8u << bh) - 1) & ~((8u << bh) - 1)) * ((d + (1u << bd) - 1) & ~((1u << bd) - 1));
        t->mips.push_back({w, h, d, bh, bd, mip_offset, mip_size});
    }
    // Resource archives initialize many textures that are never used. Allocate
    // Vulkan storage when a transfer, sample or render actually needs the image.
    AdoptCachedImage(*t);
    InstallTexture(t->object,t);
    Ret(c, 1);
}
NATIVE("nvnTextureFinalize") {
    nvn::binding_generation.fetch_add(1, std::memory_order_relaxed);
    FinalizeTexture(c->x[0]);
}
NATIVE("nvnTextureSetDebugLabel") {}
static void ForwardToBuilder(Cpu* c, const char* getter) {
    c->x[0] = CallGuest(c, nvn::DriverFunction(getter), {c->x[0], c->x[1], c->x[2], c->x[3], c->x[4]});
}
NATIVE("nvnTextureGetWidth") { ForwardToBuilder(c, "nvnTextureBuilderGetWidth"); }
NATIVE("nvnTextureGetHeight") { ForwardToBuilder(c, "nvnTextureBuilderGetHeight"); }
NATIVE("nvnTextureGetDepth") { ForwardToBuilder(c, "nvnTextureBuilderGetDepth"); }
NATIVE("nvnTextureGetLevels") { ForwardToBuilder(c, "nvnTextureBuilderGetLevels"); }
NATIVE("nvnTextureGetFormat") { ForwardToBuilder(c, "nvnTextureBuilderGetFormat"); }
NATIVE("nvnTextureGetFlags") { ForwardToBuilder(c, "nvnTextureBuilderGetFlags"); }
NATIVE("nvnTextureGetTarget") { ForwardToBuilder(c, "nvnTextureBuilderGetTarget"); }
NATIVE("nvnTextureGetSamples") { ForwardToBuilder(c, "nvnTextureBuilderGetSamples"); }
NATIVE("nvnTextureGetSwizzle") { ForwardToBuilder(c, "nvnTextureBuilderGetSwizzle"); }
NATIVE("nvnTextureGetDepthStencilMode") { ForwardToBuilder(c, "nvnTextureBuilderGetDepthStencilMode"); }
NATIVE("nvnTextureGetStride") { ForwardToBuilder(c, "nvnTextureBuilderGetStride"); }
NATIVE("nvnTextureGetMemoryOffset") { ForwardToBuilder(c, "nvnTextureBuilderGetMemoryOffset"); }
NATIVE("nvnTextureGetMemoryPool") { ForwardToBuilder(c, "nvnTextureBuilderGetMemoryPool"); }
NATIVE("nvnTextureGetStorageSize") { ForwardToBuilder(c, "nvnTextureBuilderGetStorageSize"); }
NATIVE("nvnTextureGetStorageClass") { ForwardToBuilder(c, "nvnTextureBuilderGetStorageClass"); }
NATIVE("nvnTextureGetRawStorageClass") { ForwardToBuilder(c, "nvnTextureBuilderGetRawStorageClass"); }
NATIVE("nvnTextureGetViewOffset") { ForwardToBuilder(c, "nvnTextureBuilderGetViewOffset"); }
NATIVE("nvnTextureGetSparseTileLayout") { ForwardToBuilder(c, "nvnTextureBuilderGetSparseTileLayout"); }
NATIVE("nvnTextureGetTextureAddress") { Ret(c, HostTexture(c->x[0])->storage); }
NATIVE("nvnTextureGetDebugID") { Ret(c, c->x[0] & 0xffffffff); }
NATIVE("nvnTextureCompare") { Ret(c, std::memcmp(Ptr(c, 0), Ptr(c, 1), TextureBuilderSize) == 0); }
NATIVE("nvnTextureFlushTexels") {}
NATIVE("nvnTextureInvalidateTexels") {}
