#include "../graphics_settings.h"
// NVN textures: the object keeps the texture builder bytes (so builder getters serve as texture
// getters) and the host texture pointer at +TextureBuilderSize.

#include "../../native.h"
#include "memory.h"
#include <algorithm>
#include <cstring>
#include <list>
#include <deque>
#include <map>
#include <set>
#include <array>
#include <tuple>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
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
// (OOT_IMAGE_TRACE: live images by kind)
std::mutex live_kinds_mutex; std::map<std::tuple<u32,u32,u32,bool,bool>,std::pair<s64,s64>> live_kinds;
void NoteLive(const Texture& t,s64 sign){ static const bool on=std::getenv("OOT_IMAGE_TRACE")!=nullptr; if(!on)return;
    std::lock_guard l{live_kinds_mutex}; auto& k=live_kinds[{t.width,t.height,t.format,(t.usage&(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))!=0,t.decoded_astc}];
    k.first+=sign; k.second+=sign*(s64)t.allocation_size; }
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

// ---- image memory in shared blocks ---------------------------------------------------------------
// A vkAllocateMemory of its own for every image cost the executor ~15% of its time while new scenery came
// into view (driver ioctls, up to ~1 ms each). Images up to 4 MiB share 64 MiB device-local blocks (all
// optimal tiling: no granularity conflicts); larger ones keep their own allocation. Empty blocks beyond one
// spare are freed. OOT_DEDICATED_IMAGES=1: every image its own allocation.
namespace {
struct ImageBlock { VkDeviceMemory memory = VK_NULL_HANDLE; u32 type = 0; u64 size = 0, used = 0; std::map<u64, u64> free; };
std::mutex blocks_mutex;
std::vector<ImageBlock*> image_blocks;
constexpr u64 BlockSize = 64ull << 20, BlockMax = 4ull << 20;
bool TakeFrom(ImageBlock& b, const VkMemoryRequirements& req, u64* offset) {
    for (auto it = b.free.begin(); it != b.free.end(); ++it) {
        const u64 start = it->first, end = it->first + it->second;
        const u64 at = (start + req.alignment - 1) / req.alignment * req.alignment;
        if (at + req.size > end) continue;
        b.free.erase(it);
        if (at > start) b.free[start] = at - start;
        if (at + req.size < end) b.free[at + req.size] = end - (at + req.size);
        b.used += req.size; *offset = at;
        return true;
    }
    return false;
}
bool SubAllocate(const VkMemoryRequirements& req, u32 type, Texture& t) {
    static const bool off = std::getenv("OOT_DEDICATED_IMAGES") != nullptr;
    if (off || req.size > BlockMax || !(req.memoryTypeBits >> type & 1)) return false;
    std::lock_guard lock{blocks_mutex};
    for (ImageBlock* b : image_blocks)
        if (b->type == type && b->size - b->used >= req.size && TakeFrom(*b, req, &t.memory_offset)) { t.memory = b->memory; t.block = b; return true; }
    auto* b = new ImageBlock;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = BlockSize; ai.memoryTypeIndex = type;
    if (vkAllocateMemory(Vk().device, &ai, nullptr, &b->memory) != VK_SUCCESS) { delete b; return false; }
    b->type = type; b->size = BlockSize; b->free[0] = BlockSize;
    image_blocks.push_back(b);
    if (!TakeFrom(*b, req, &t.memory_offset)) return false; // (cannot happen: req.size <= BlockMax)
    t.memory = b->memory; t.block = b;
    return true;
}
void SubFree(Texture& t) {
    std::lock_guard lock{blocks_mutex};
    auto* b = static_cast<ImageBlock*>(t.block);
    VkMemoryRequirements req; vkGetImageMemoryRequirements(Vk().device, t.image, &req); // (the image still exists here)
    u64 start = t.memory_offset, size = req.size;
    b->used -= size;
    auto next = b->free.lower_bound(start);
    if (next != b->free.end() && next->first == start + size) { size += next->second; next = b->free.erase(next); }
    if (next != b->free.begin()) { auto prev = std::prev(next); if (prev->first + prev->second == start) { start = prev->first; size += prev->second; b->free.erase(prev); } }
    b->free[start] = size;
    t.block = nullptr; t.memory = VK_NULL_HANDLE;
    if (b->used == 0) {
        u32 empty = 0; for (ImageBlock* o : image_blocks) empty += o->used == 0;
        if (empty > 1) {
            vkFreeMemory(Vk().device, b->memory, nullptr);
            image_blocks.erase(std::find(image_blocks.begin(), image_blocks.end(), b));
            delete b;
        }
    }
}
} // namespace

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
    texture.image=VK_NULL_HANDLE;texture.memory=VK_NULL_HANDLE;texture.view=VK_NULL_HANDLE;texture.block=nullptr;
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
    // (not between two commands of a list the executor runs: later commands of it may still name this texture)
    struct Patient { Patient() { BatchMutex().lock_patient(); } ~Patient() { BatchMutex().unlock(); } } batch_lock;
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

// Images of attachment textures (own memory) that go away are kept for the next texture of the same description:
// OoT makes ~2 new render targets a frame, and the driver's vkCreateImage + vkAllocateMemory + bind cost the executor
// ~2 ms of a frame. A kept image is as good as a new one (contents undefined, layout UNDEFINED to the first use).
// OOT_NO_SPARE_IMAGES=1: destroyed at once, as before.
using SpareKey = std::tuple<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32>;
// Per description, newest last; the newest is taken, and over the budget the oldest of all goes. (Taken oldest
// first, the images a description has too many of, more retired than needed, never aged: they filled the budget,
// and descriptions needed every few frames were dropped before their turn, so a third of a frame's new render
// targets, 8 of 24 in Castle Town, were made by the driver again: 3 ms of the executor's frame.) Spares nothing
// took for 1800 frames go.
struct SpareImage { VkImage image; VkDeviceMemory memory; u64 size; u64 frame; u64 sequence; };
struct SpareKind { std::deque<SpareImage> images; u64 bytes = 0; };
std::mutex spare_mutex;
std::map<SpareKey, SpareKind> spare_images;
u64 spare_bytes = 0;
u64 SpareBudget() {
    static const u64 budget = [] {
        if (const char* e = std::getenv("OOT_SPARE_IMAGE_MB")) return std::strtoull(e, nullptr, 10) << 20;
        auto& vk = Vk(); u64 local = 0;
        for (u32 h = 0; h < vk.memory_properties.memoryHeapCount; ++h) if (vk.memory_properties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local = std::max<u64>(local, vk.memory_properties.memoryHeaps[h].size);
        const bool small = vk.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || local < (6ull << 30);
        return small ? 128ull << 20 : 384ull << 20; }();
    return budget;
}
std::atomic<u64> stat_spare_hits{0}, stat_images_created{0}, stat_targets_created{0}, stat_spare_kept{0}, stat_spare_dropped{0}, stat_spare_refused{0};
std::mutex kind_stats_mutex; std::map<SpareKey, std::array<u64, 4>> kind_stats; // (OOT_IMAGE_TRACE) created, from spares, kept, dropped
void KindStat(const SpareKey& k, int what) { static const bool on = std::getenv("OOT_IMAGE_TRACE") != nullptr; if (!on) return; std::lock_guard l{kind_stats_mutex}; ++kind_stats[k][what]; }
SpareKey SpareKeyOf(const VkImageCreateInfo& ii) {
    return {(u32)ii.imageType, (u32)ii.format, ii.extent.width, ii.extent.height, ii.extent.depth, ii.mipLevels, ii.arrayLayers,
            (u32)ii.samples, (u32)ii.usage, (u32)ii.flags};
}
bool TakeSpareImage(const VkImageCreateInfo& ii, Texture& t) {
    static const bool off = std::getenv("OOT_NO_SPARE_IMAGES") != nullptr;
    if (off) return false;
    std::lock_guard lock{spare_mutex};
    auto it = spare_images.find(SpareKeyOf(ii));
    if (it == spare_images.end()) return false;
    const SpareImage s = it->second.images.back(); it->second.images.pop_back(); // (the newest: the oldest age out)
    t.image = s.image; t.memory = s.memory; t.allocation_size = s.size; t.memory_offset = 0;
    KindStat(it->first, 1);
    spare_bytes -= s.size; it->second.bytes -= s.size;
    if (it->second.images.empty()) spare_images.erase(it);
    return true;
}
// The memory of render targets whose image goes is kept for the next one needing about as much: OoT's shadow maps
// change size every frame (817, 818, 819... 1791, 1875 square), so no spare image of the same description is ever
// there, and vkAllocateMemory (not vkCreateImage) was the cost: ~13% of the executor's time in Castle Town. Memory
// of its own per image, as before (NVIDIA compresses attachments only so), just recycled. OOT_NO_SPARE_MEMORY=1: off.
struct SpareMemory { VkDeviceMemory memory; u32 type; u64 frame; };
std::multimap<u64, SpareMemory> spare_memory; // by size; spare_mutex held
u64 spare_memory_bytes = 0;
std::unordered_map<VkDeviceMemory, u32> attachment_memory_types; // allocations of render targets: their memory type
std::atomic<u64> stat_memory_reused{0};
// Destroying images and freeing their memory costs the driver about as much as making them, and ran on the executor
// (in the actions of retired batches): a thread of its own does it. OOT_SYNC_DESTROY=1: in place, as before.
void DestroyOffThread(std::function<void()> destroy) {
    static const bool sync = std::getenv("OOT_SYNC_DESTROY") != nullptr;
    if (sync) { destroy(); return; }
    static std::mutex m; static std::condition_variable cv; static std::vector<std::function<void()>> queue;
    static std::once_flag once;
    std::call_once(once, [] { std::thread([] {
        rt::SetThreadName("vk-destroy");
        for (;;) {
            std::vector<std::function<void()>> work;
            { std::unique_lock lock{m}; cv.wait(lock, [] { return !queue.empty(); }); work.swap(queue); }
            for (auto& f : work) f();
        }
    }).detach(); });
    { std::lock_guard lock{m}; queue.push_back(std::move(destroy)); }
    cv.notify_one();
}
void FreeSpareMemoryLocked(std::multimap<u64, SpareMemory>::iterator it) {
    DestroyOffThread([memory = it->second.memory] { vkFreeMemory(Vk().device, memory, nullptr); });
    attachment_memory_types.erase(it->second.memory);
    spare_memory_bytes -= it->first; spare_memory.erase(it);
}
// (spare_mutex held) false: not memory of a render target, or not kept; the caller frees it.
bool KeepSpareMemory(VkDeviceMemory memory, u64 size) {
    static const bool off = std::getenv("OOT_NO_SPARE_MEMORY") != nullptr;
    auto type = attachment_memory_types.find(memory);
    if (off || type == attachment_memory_types.end()) { if (type != attachment_memory_types.end()) attachment_memory_types.erase(type); return false; }
    spare_memory.emplace(size, SpareMemory{memory, type->second, CurrentFrameNumber()});
    spare_memory_bytes += size;
    while (spare_memory_bytes > SpareBudget() && !spare_memory.empty()) { // (the oldest goes)
        auto oldest = spare_memory.begin();
        for (auto it = spare_memory.begin(); it != spare_memory.end(); ++it) if (it->second.frame < oldest->second.frame) oldest = it;
        FreeSpareMemoryLocked(oldest);
    }
    return true;
}
// Memory of type `type` holding at least `size`, not much more: taken from the spares.
bool TakeSpareMemory(u64 size, u32 type, VkDeviceMemory* memory, u64* got) {
    std::lock_guard lock{spare_mutex};
    // (the shadow maps' sizes wander between frames: on GPUs of their own memory up to twice the size is taken)
    const bool roomy = SpareBudget() > (128ull << 20);
    const u64 limit = roomy ? size * 2 + (1ull << 20) : size + std::max<u64>(size / 4, 1ull << 20);
    for (auto it = spare_memory.lower_bound(size); it != spare_memory.end() && it->first <= limit; ++it) {
        if (it->second.type != type) continue;
        *memory = it->second.memory; *got = it->first;
        spare_memory_bytes -= it->first; spare_memory.erase(it);
        ++stat_memory_reused;
        return true;
    }
    return false;
}
void NoteAttachmentMemory(VkDeviceMemory memory, u32 type) { std::lock_guard lock{spare_mutex}; attachment_memory_types[memory] = type; }
void DestroySpare(const SpareImage& s) { // (spare_mutex held)
    const bool keep = KeepSpareMemory(s.memory, s.size);
    DestroyOffThread([image = s.image, memory = keep ? VK_NULL_HANDLE : s.memory] {
        vkDestroyImage(Vk().device, image, nullptr);
        if (memory) vkFreeMemory(Vk().device, memory, nullptr); });
}
void DropSpareImages() { // (low on memory)
    std::lock_guard lock{spare_mutex};
    for (auto& [k, kind] : spare_images) for (auto& s : kind.images) { vkDestroyImage(Vk().device, s.image, nullptr); attachment_memory_types.erase(s.memory); vkFreeMemory(Vk().device, s.memory, nullptr); }
    spare_images.clear(); spare_bytes = 0;
    while (!spare_memory.empty()) FreeSpareMemoryLocked(spare_memory.begin());
}
// (spare_mutex held) The oldest spare of a description goes.
void DropOldestSpare(std::map<SpareKey, SpareKind>::iterator kind) {
    const SpareImage s = kind->second.images.front(); kind->second.images.pop_front();
    KindStat(kind->first, 3);
    spare_bytes -= s.size; kind->second.bytes -= s.size; DestroySpare(s); ++stat_spare_dropped;
    if (kind->second.images.empty()) spare_images.erase(kind);
}
void AgeSpareImages() { // (now and then: what nothing took for 1800 frames)
    std::lock_guard lock{spare_mutex};
    const u64 now = CurrentFrameNumber();
    for (auto it = spare_images.begin(); it != spare_images.end();) {
        auto kind = it++;
        while (now - kind->second.images.front().frame > 1800) { const bool last = kind->second.images.size() == 1; DropOldestSpare(kind); if (last) break; }
    }
    for (auto it = spare_memory.begin(); it != spare_memory.end();) { auto m = it++; if (now - m->second.frame > 1800) FreeSpareMemoryLocked(m); }
}
// (the image is no longer used by the GPU) true: kept.
bool KeepSpareImage(const Texture& t) {
    static const bool off = std::getenv("OOT_NO_SPARE_IMAGES") != nullptr;
    if (off || !t.image || !t.memory || t.block || t.allocation_size > SpareBudget() / 2) { ++stat_spare_refused; return false; }
    if (!(t.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) { ++stat_spare_refused; return false; }
    ++stat_spare_kept;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = t.target == Target::T3D ? VK_IMAGE_TYPE_3D : (t.target == Target::T1D || t.target == Target::T1DArray) ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
    ii.format = t.info.vk; ii.extent = {t.width, t.height, t.target == Target::T3D ? t.depth : 1};
    ii.mipLevels = t.levels; ii.arrayLayers = t.target == Target::T3D ? 1 : t.layers; ii.samples = (VkSampleCountFlagBits)t.samples; ii.usage = t.usage;
    ii.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; // (as CreateImage)
    if ((t.target == Target::Cube || t.target == Target::CubeArray) && t.layers % 6 == 0) ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (t.target == Target::T3D) ii.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    if (t.info.block_w > 1 && !t.decoded_astc) ii.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    std::lock_guard lock{spare_mutex};
    KindStat(SpareKeyOf(ii), 2);
    auto& kind = spare_images[SpareKeyOf(ii)];
    static u64 sequence = 0;
    kind.images.push_back({t.image, t.memory, t.allocation_size, CurrentFrameNumber(), ++sequence});
    kind.bytes += t.allocation_size; spare_bytes += t.allocation_size;
    while (spare_bytes > SpareBudget() && !spare_images.empty()) {
        auto oldest = spare_images.begin();
        for (auto k = spare_images.begin(); k != spare_images.end(); ++k) if (k->second.images.front().sequence < oldest->second.images.front().sequence) oldest = k;
        DropOldestSpare(oldest);
    }
    return true;
}

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
        // (also on GPUs sharing system memory, such as the Steam Deck's, or with under 6 GB of their own, as
        // Hookshot does since 4378a24: the threshold of Low unless the user chose Quality)
        static const bool small_memory=[&]{
            VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(vk.physical,&p);
            u64 local=0;for(u32 h=0;h<vk.memory_properties.memoryHeapCount;++h)if(vk.memory_properties.memoryHeaps[h].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)local=std::max<u64>(local,vk.memory_properties.memoryHeaps[h].size);
            const bool small=p.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU||local<(6ull<<30);
            if(small)RT_LOG("textures: shared or small video memory: ASTC textures from 256 KiB kept as BC3");
            return small;}();
        const bool low=gfx::Current().texture_memory==1||small_memory;
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
    static const bool trace = std::getenv("OOT_IMAGE_TRACE") != nullptr;
    if (trace) { static std::mutex m; static std::map<std::tuple<u32,u32,u32,u32,u32,u32>, u64> made; static u64 n = 0; std::lock_guard l{m};
        ++made[{t.width, t.height, t.format, t.levels, t.layers, (u32)!!(ii.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))}];
        if (++n % 500 == 0) { for (auto& [k, c] : made) if (c > 4) RT_LOG("images made: %ux%u fmt %u levels %u layers %u attachment %u: %llu", std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k), std::get<4>(k), std::get<5>(k), (unsigned long long)c); made.clear(); RT_LOG("(500 images made)"); } }
    if (TakeSpareImage(ii, t)) { ++stat_spare_hits; image_live_bytes += t.allocation_size; if (t.decoded_astc) image_astc_bytes += t.allocation_size; NoteLive(t, 1); return; }
    KindStat(SpareKeyOf(ii), 0);
    ++stat_images_created; if (ii.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) ++stat_targets_created;
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &t.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(vk.device, t.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // (only images that cannot be render targets: NVIDIA compresses attachments in memory of their own; shared
    // blocks made the frame's GPU time 20% longer)
    if(!(ii.usage&(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))&&SubAllocate(req,ai.memoryTypeIndex,t)){
        t.allocation_size=req.size;image_live_bytes+=req.size;if(t.decoded_astc)image_astc_bytes+=req.size;NoteLive(t,1);
        VK_CHECK(vkBindImageMemory(vk.device,t.image,t.memory,t.memory_offset));
        return;
    }
    const bool attachment=(ii.usage&(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))!=0;
    if(attachment){u64 got=0;
        if(TakeSpareMemory(req.size,ai.memoryTypeIndex,&t.memory,&got)){
            {std::lock_guard lock{spare_mutex};attachment_memory_types[t.memory]=ai.memoryTypeIndex;}
            t.allocation_size=got;image_live_bytes+=got;if(t.decoded_astc)image_astc_bytes+=got;NoteLive(t,1);
            VK_CHECK(vkBindImageMemory(vk.device,t.image,t.memory,0));
            return;
        }}
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
        DropSpareImages();
        allocated=vkAllocateMemory(vk.device,&ai,nullptr,&t.memory);
        if(allocated!=VK_SUCCESS){ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,0);allocated=vkAllocateMemory(vk.device,&ai,nullptr,&t.memory);
            if(allocated==VK_SUCCESS)RT_LOG("texture placed in non-device-local memory");}
    }
    if(allocated!=VK_SUCCESS)RT_LOG("texture allocation failed: %ux%ux%u format %u, requested %llu MiB; live images %llu MiB, cached %llu MiB",t.width,t.height,t.depth,t.format,
        (unsigned long long)(req.size>>20),(unsigned long long)(image_live_bytes.load()>>20),(unsigned long long)(image_cached_bytes.load()>>20));
    VK_CHECK(allocated);
    if(attachment)NoteAttachmentMemory(t.memory,ai.memoryTypeIndex);
    t.allocation_size=req.size;image_live_bytes+=req.size;if(t.decoded_astc)image_astc_bytes+=req.size;NoteLive(t,1);
    VK_CHECK(vkBindImageMemory(vk.device, t.image, t.memory, 0));
}

} // namespace
bool StealLiveImage(Texture& t); void NoteHolder(Texture& t); void ReportImageAges(); void ReuseOverlapping(Texture& t, bool take);
void EnsureImage(Texture& t) {
    // A finalized texture still used by recorded work keeps its image until that batch. One no shared owner
    // holds any more (a stale reference to an object on its way out, destroyed with its image) is not
    // retired again.
    std::shared_ptr<Texture> self=t.weak_from_this().lock();
    if(t.owner_finalized&&!self){
        static std::atomic<u32> told{0};
        if(told++<5)RT_LOG("texture %#llx used after its last owner released it",(unsigned long long)t.object);
    }
    t.used_frame=CurrentFrameNumber();
    if(!t.image){
        AdoptCachedImage(t); // a descriptor replacement may have released it since initialization
        if(!t.image)StealLiveImage(t);
        {static const bool take=std::getenv("OOT_IMAGE_REUSE_TAKE")!=nullptr; // (taking them made the GPU time longer: re-uploads)
         if(!t.image&&take)ReuseOverlapping(t,true);}
        const bool created=!t.image;
        if(created){const auto t0=std::chrono::steady_clock::now();CreateImage(t);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            if(ms>20){static std::atomic<u32> shown{0};if(shown++<40)RT_LOG("image %ux%u fmt %u levels %u layers %u (%.0f MiB) at %#llx made in %.0f ms",t.width,t.height,t.format,t.levels,t.layers,
                t.allocation_size/1048576.0,(unsigned long long)t.storage,ms);}}
        NoteHolder(t);
        if(created)ReuseOverlapping(t,false); // (only a new image brings a new occupant of that memory)
    }
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
void ForgetImageHolder(Texture* t);
namespace {
using ImageKey = std::tuple<u64, u32, u32, u32, u32, u32, u32, u32, u32>;
ImageKey KeyOf(const Texture& t) {
    return {t.storage, t.format, t.width, t.height, t.depth, t.levels, t.layers, t.samples, (u32)t.target};
}
std::mutex cache_mutex;
std::list<ImageKey> cache_order;
// Small images the GPU drew (4 MiB or less) have an LRU and budget of their own: OoT draws some once (terrain tiles'
// normal maps when a chunk comes in), finalizes them, and later makes a texture over that memory again and samples it
// without drawing it. The render targets cycling through the main cache evicted them, and they came back from guest
// memory as garbage: streaks over the ground in square patches after a while of play (Hookshot #593/763a693).
// OOT_SMALL_IMAGE_CACHE_MB=0: one cache, as before.
std::list<ImageKey> small_order;
u64 small_cached_bytes = 0;
const u64 SmallImageMax = 4ull << 20;
const u64 SmallCacheBudget = std::getenv("OOT_SMALL_IMAGE_CACHE_MB") ? std::strtoull(std::getenv("OOT_SMALL_IMAGE_CACHE_MB"), nullptr, 10) << 20 : 512ull << 20;
constexpr size_t SmallCacheEntries = 8192;
struct CachedImage {
    std::unique_ptr<Texture> texture;
    std::list<ImageKey>::iterator position;
    bool small = false;
};
std::map<ImageKey,CachedImage> image_cache;

void DestroyLater(Texture* t) {
    ForgetImageHolder(t); // (on its way out: no other object takes its image now)
    AfterFlush([t] {
        std::vector<VkImageView> views;
        if (t->view) views.push_back(t->view);
        for (auto& [key, view] : t->views) views.push_back(view);
        t->view = VK_NULL_HANDLE; t->views.clear();
        if (t->block || t->memory) NoteLive(*t,-1);
        if (t->block) {SubFree(*t);image_live_bytes-=t->allocation_size;if(t->decoded_astc)image_astc_bytes-=t->allocation_size;} // (before the image goes: it reads its size)
        if (t->image && !t->block && t->memory && KeepSpareImage(*t)) {
            image_live_bytes-=t->allocation_size;if(t->decoded_astc)image_astc_bytes-=t->allocation_size; delete t;
            if (!views.empty()) DestroyOffThread([views = std::move(views)] { for (VkImageView v : views) vkDestroyImageView(Vk().device, v, nullptr); });
            return; }
        VkDeviceMemory free_memory = VK_NULL_HANDLE;
        if (t->memory && !t->block) {bool kept; {std::lock_guard lock{spare_mutex}; kept = KeepSpareMemory(t->memory, t->allocation_size);}
            if (!kept) free_memory = t->memory; image_live_bytes-=t->allocation_size;if(t->decoded_astc)image_astc_bytes-=t->allocation_size;}
        DestroyOffThread([views = std::move(views), image = t->image, free_memory] {
            auto& vk = Vk();
            for (VkImageView v : views) vkDestroyImageView(vk.device, v, nullptr);
            if (image) vkDestroyImage(vk.device, image, nullptr);
            if (free_memory) vkFreeMemory(vk.device, free_memory, nullptr); });
        delete t;
    });
}
} // namespace

// (cache_mutex held)
void EvictOldest(bool small = false) {
    auto& order = small ? small_order : cache_order;
    auto old = image_cache.find(order.front());
    if (old != image_cache.end()) { image_cached_bytes-=old->second.texture->allocation_size;if(small)small_cached_bytes-=old->second.texture->allocation_size;
        DestroyLater(old->second.texture.release()); image_cache.erase(old); ++image_cache_evicted; }
    order.pop_front();
}
std::atomic<u64> image_cache_overlap{0};
void AdoptCachedImage(Texture& t) {
    std::lock_guard lock{cache_mutex};
    auto it = image_cache.find(KeyOf(t));
    bool contents = true;
    if (it == image_cache.end()) {
        // A finalized image of the same description over memory t now uses: that memory was reused (OoT's transient
        // render targets move a little every frame), so the image is free; its contents are not t's.
        // OOT_OVERLAP_ADOPT=1 (no gain measured).
        static const bool off = std::getenv("OOT_OVERLAP_ADOPT") == nullptr; // (no gain measured: opt-in)
        if (off || !t.storage_size) return;
        for (auto c = image_cache.begin(); c != image_cache.end(); ++c) {
            const Texture& o = *c->second.texture;
            if (o.format != t.format || o.width != t.width || o.height != t.height || o.depth != t.depth || o.levels != t.levels ||
                o.layers != t.layers || o.samples != t.samples || o.target != t.target || !o.storage_size) continue;
            if (o.storage >= t.storage + t.storage_size || t.storage >= o.storage + o.storage_size) continue;
            it = c; contents = false; ++image_cache_overlap; break;
        }
        if (it == image_cache.end()) return;
    }
    Texture& cached = *it->second.texture;
    t.image = cached.image; t.memory = cached.memory; t.view = cached.view; t.views = std::move(cached.views);
    t.block = cached.block; t.memory_offset = cached.memory_offset; cached.block = nullptr;
    t.allocation_size=cached.allocation_size;image_cached_bytes-=cached.allocation_size;if(it->second.small)small_cached_bytes-=cached.allocation_size;
    t.layout = cached.layout; t.usage = cached.usage; t.gpu_written = contents && cached.gpu_written; t.uploaded = contents && cached.uploaded;
    t.decoded_astc = cached.decoded_astc; t.bc3 = cached.bc3; t.decoded_rgb9e5 = cached.decoded_rgb9e5; t.info = cached.info;
    cached.image = VK_NULL_HANDLE; cached.memory = VK_NULL_HANDLE; cached.view = VK_NULL_HANDLE; cached.views.clear();
    (it->second.small ? small_order : cache_order).erase(it->second.position);
    image_cache.erase(it);
    ++image_cache_adopted;
}

void RetireTexture(Texture* t) {
    ForgetImageHolder(t); // (cached or destroyed: either way no longer a holder)
    if (!t->image || !t->gpu_written) { DestroyLater(t); return; }
    std::lock_guard lock{cache_mutex};
    const ImageKey key = KeyOf(*t);
    auto it=image_cache.find(key);
    if(it!=image_cache.end()) { // (the older one goes; this one takes its place in its list)
        image_cached_bytes-=it->second.texture->allocation_size;if(it->second.small)small_cached_bytes-=it->second.texture->allocation_size;
        DestroyLater(it->second.texture.release());
        (it->second.small ? small_order : cache_order).erase(it->second.position);
        image_cache.erase(it);
    }
    const bool small=SmallCacheBudget&&t->allocation_size<=SmallImageMax;
    auto& order=small?small_order:cache_order;
    order.push_back(key);
    image_cache.emplace(key,CachedImage{std::unique_ptr<Texture>(t),std::prev(order.end()),small});
    image_cached_bytes+=t->allocation_size;if(small)small_cached_bytes+=t->allocation_size;
    t->object = 0;
    while (!cache_order.empty()&&(cache_order.size() > ImageCacheEntries||image_cached_bytes-small_cached_bytes>ImageCacheBudget)) EvictOldest(false);
    while (!small_order.empty()&&(small_order.size() > SmallCacheEntries||small_cached_bytes>SmallCacheBudget)) EvictOldest(true);
}
// ---- one image per memory and description, as on the console ----------------------------------------
// OoT makes new texture objects over the same memory every frame (its render targets) while older objects
// over it are still referenced (descriptors, unfinalized objects). Each had an image of its own: ~12 new
// images a frame (driver allocations on the executor, ~15% of its time) and gigabytes of copies (155 of
// one 960x540 target). On the console they are one memory: here the image goes to whichever object over
// it is used, with its contents and layout. OOT_NO_IMAGE_SHARING=1: an image per object.
namespace {
std::mutex holders_mutex;
std::map<ImageKey, Texture*> holders; // the object holding the image of that memory and description
std::atomic<u64> images_shared{0}, images_reused{0}, images_released{0};
}
std::set<Texture*> with_images; // objects holding images (OOT_IMAGE_TRACE: their ages)
// (the same by description, for ReuseOverlapping: only objects of t's description are candidates)
using DescKey = std::tuple<u32, u32, u32, u32, u32, u32, u32, u32>;
DescKey DescOf(const Texture& t) { return {t.format, t.width, t.height, t.depth, t.levels, t.layers, t.samples, (u32)t.target}; }
std::map<DescKey, std::set<Texture*>> by_desc;
void Unlist(Texture* t) { // (holders_mutex held)
    if (!with_images.erase(t)) return;
    auto it = by_desc.find(DescOf(*t));
    if (it != by_desc.end()) { it->second.erase(t); if (it->second.empty()) by_desc.erase(it); }
}
void NoteHolder(Texture& t) {
    if (!t.image) return;
    std::lock_guard lock{holders_mutex};
    holders[KeyOf(t)] = &t;
    if (with_images.insert(&t).second) by_desc[DescOf(t)].insert(&t);
}
void ForgetImageHolder(Texture* t) {
    std::lock_guard lock{holders_mutex};
    Unlist(t);
    auto it = holders.find(KeyOf(*t));
    if (it != holders.end() && it->second == t) holders.erase(it);
}
// Images of objects over memory a newer object of the same description now uses, unused for a while
// (OoT's transient render targets move a little every frame: 131 copies of one 960x540 target were kept,
// 4.9 MB each). take: the first one becomes t's image (its contents are not t's: as a new image);
// otherwise they are destroyed. OOT_NO_IMAGE_REUSE=1: kept.
void ReuseOverlapping(Texture& t, bool take) {
    static const bool off = std::getenv("OOT_NO_IMAGE_REUSE") != nullptr;
    if (off || !t.storage || !t.storage_size) return;
    const u64 now = CurrentFrameNumber();
    std::vector<Texture*> dead;
    {
        std::lock_guard lock{holders_mutex};
        auto candidates = by_desc.find(DescOf(t));
        if (candidates == by_desc.end()) return;
        for (Texture* h : candidates->second) {
            // (taken after 10 frames unused, destroyed after 60: objects touched every few frames keep theirs)
            static const u64 take_after = std::getenv("OOT_IMAGE_TAKE_FRAMES") ? std::strtoull(std::getenv("OOT_IMAGE_TAKE_FRAMES"), nullptr, 10) : 60;
            if (h == &t || !h->image || now - h->used_frame < (take ? take_after : 60u) || !h->storage_size) continue;
            if (h->storage >= t.storage + t.storage_size || t.storage >= h->storage + h->storage_size) continue;
            if (h->format != t.format || h->width != t.width || h->height != t.height || h->depth != t.depth || h->levels != t.levels ||
                h->layers != t.layers || h->samples != t.samples || h->target != t.target) continue;
            if (take) {
                if (h->depth_stencil_mode != t.depth_stencil_mode || std::memcmp(h->swizzle, t.swizzle, sizeof t.swizzle)) continue;
                t.image = h->image; t.memory = h->memory; t.block = h->block; t.memory_offset = h->memory_offset; t.allocation_size = h->allocation_size;
                t.view = h->view; t.views = std::move(h->views); t.layout = h->layout; t.usage = h->usage;
                t.decoded_astc = h->decoded_astc; t.bc3 = h->bc3; t.decoded_rgb9e5 = h->decoded_rgb9e5; t.info = h->info;
                t.uploaded = t.gpu_written = false; // (a new image as far as its contents go)
                h->image = VK_NULL_HANDLE; h->memory = VK_NULL_HANDLE; h->block = nullptr; h->view = VK_NULL_HANDLE; h->views.clear(); h->allocation_size = 0;
                h->layout = VK_IMAGE_LAYOUT_UNDEFINED; h->uploaded = h->gpu_written = false;
                auto it = holders.find(KeyOf(*h)); if (it != holders.end() && it->second == h) holders.erase(it);
                Unlist(h);
                ++binding_generation; ++images_reused;
                return;
            }
            dead.push_back(h);
        }
        for (Texture* h : dead) {
            auto it = holders.find(KeyOf(*h)); if (it != holders.end() && it->second == h) holders.erase(it);
            Unlist(h);
        }
    }
    for (Texture* h : dead) { // (the image goes after the GPU is done with it; the object stays, without one)
        auto* image = new Texture(*h);
        h->image = VK_NULL_HANDLE; h->memory = VK_NULL_HANDLE; h->block = nullptr; h->view = VK_NULL_HANDLE; h->views.clear(); h->allocation_size = 0;
        h->layout = VK_IMAGE_LAYOUT_UNDEFINED; h->uploaded = h->gpu_written = false;
        image->object = 0;
        DestroyLater(image);
        ++images_released;
    }
    if (!dead.empty()) ++binding_generation;
}
void ReportImageAges() {
    std::lock_guard lock{holders_mutex};
    const u64 now = CurrentFrameNumber();
    std::map<std::tuple<u32,u32,u32>, std::array<u64,4>> ages; // (<60, <600, older frames; bytes)
    for (Texture* t : with_images) if (t->image) {
        auto& a = ages[{t->width, t->height, t->format}];
        const u64 age = now - t->used_frame;
        ++a[age < 2 ? 0 : age < 10 ? 1 : 2]; a[3] += t->allocation_size;
    }
    std::vector<std::pair<u64, std::string>> top;
    for (auto& [k, a] : ages) { char b[160]; std::snprintf(b, sizeof b, "%ux%u fmt %u: used this or the last frame %llu, 2-9 frames ago %llu, earlier %llu (%.0f MiB)", std::get<0>(k), std::get<1>(k), std::get<2>(k),
        (unsigned long long)a[0], (unsigned long long)a[1], (unsigned long long)a[2], a[3] / 1048576.0); top.push_back({a[3], b}); }
    std::sort(top.rbegin(), top.rend());
    for (size_t i = 0; i < top.size() && i < 10; ++i) RT_LOG("  held %s", top[i].second.c_str());
    std::vector<std::pair<u64, u64>> spans; // (960x540 format 41: where they are)
    for (Texture* t : with_images) if (t->image && t->width == 960 && t->height == 540 && t->format == 41) spans.push_back({t->storage, t->storage_size});
    std::sort(spans.begin(), spans.end());
    std::string list; u64 overlapping = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (i && spans[i].first < spans[i - 1].first + spans[i - 1].second) ++overlapping;
        if (i < 24) { char b[48]; std::snprintf(b, sizeof b, " %#llx+%#llx", (unsigned long long)spans[i].first, (unsigned long long)spans[i].second); list += b; }
    }
    RT_LOG("  960x540/41 storage (%zu, %llu overlapping the previous):%s", spans.size(), (unsigned long long)overlapping, list.c_str());
}
bool StealLiveImage(Texture& t) {
    static const bool off = std::getenv("OOT_NO_IMAGE_SHARING") != nullptr;
    if (off || !t.storage) return false;
    std::lock_guard lock{holders_mutex};
    auto it = holders.find(KeyOf(t));
    if (it == holders.end()) return false;
    Texture& h = *it->second;
    if (&h == &t || !h.image || h.depth_stencil_mode != t.depth_stencil_mode || std::memcmp(h.swizzle, t.swizzle, sizeof t.swizzle)) return false;
    t.image = h.image; t.memory = h.memory; t.block = h.block; t.memory_offset = h.memory_offset; t.allocation_size = h.allocation_size;
    t.view = h.view; t.views = std::move(h.views);
    t.layout = h.layout; t.usage = h.usage; t.gpu_written = h.gpu_written; t.uploaded = h.uploaded; t.upload_frame = h.upload_frame;
    t.upload_sequence = h.upload_sequence; t.checked_sequence = h.checked_sequence; t.content_hash = h.content_hash; t.hash_frame = h.hash_frame;
    t.decoded_astc = h.decoded_astc; t.bc3 = h.bc3; t.decoded_rgb9e5 = h.decoded_rgb9e5; t.info = h.info;
    h.image = VK_NULL_HANDLE; h.memory = VK_NULL_HANDLE; h.block = nullptr; h.view = VK_NULL_HANDLE; h.views.clear(); h.allocation_size = 0;
    h.layout = VK_IMAGE_LAYOUT_UNDEFINED; h.uploaded = h.gpu_written = false;
    it->second = &t;
    ++binding_generation; // (descriptors cached with the other object's views are made again)
    ++images_shared;
    return true;
}
Texture::~Texture() {
    std::lock_guard lock{holders_mutex};
    Unlist(this);
    auto it = holders.find(KeyOf(*this));
    if (it != holders.end() && it->second == this) holders.erase(it);
}
void TrimImageCache(u64 keep_bytes) {
    std::lock_guard lock{cache_mutex};
    while (!cache_order.empty() && image_cached_bytes > keep_bytes) EvictOldest(false);
    while (!small_order.empty() && image_cached_bytes > keep_bytes) EvictOldest(true);
}
} // namespace nvn
namespace nvn {
void ReportImageCache() {
    std::lock_guard lock{cache_mutex};
    RT_LOG("image cache: %llu adopted (%llu over reused memory), %llu shared, %llu reused, %llu released, %llu evicted; %zu cached (%.0f MiB), %.0f MiB live (%.0f MiB decoded ASTC)",
           (unsigned long long)image_cache_adopted.exchange(0), (unsigned long long)image_cache_overlap.exchange(0), (unsigned long long)images_shared.exchange(0), (unsigned long long)images_reused.exchange(0),
           (unsigned long long)images_released.exchange(0), (unsigned long long)image_cache_evicted.exchange(0),
           image_cache.size(), image_cached_bytes / 1048576.0, image_live_bytes / 1048576.0, image_astc_bytes / 1048576.0);
    AgeSpareImages();
    RT_LOG("render target memory: %.1f per frame from spares, %.0f MiB spare", stat_memory_reused.exchange(0) / 300.0, spare_memory_bytes / 1048576.0);
    RT_LOG("images per frame: %.1f created (%.1f render targets), %.1f from spares; spares: %.1f kept, %.1f dropped over the budget, %.1f not kept; %.0f MiB spare",
           stat_images_created.exchange(0) / 300.0, stat_targets_created.exchange(0) / 300.0, stat_spare_hits.exchange(0) / 300.0,
           stat_spare_kept.exchange(0) / 300.0, stat_spare_dropped.exchange(0) / 300.0, stat_spare_refused.exchange(0) / 300.0, spare_bytes / 1048576.0);
    if (std::getenv("OOT_IMAGE_TRACE")) {
        std::lock_guard l{kind_stats_mutex};
        for (auto& [k, v] : kind_stats) if (v[0] + v[3] > 20) RT_LOG("  kind %ux%ux%u vkfmt %u levels %u layers %u usage %#x: %llu created, %llu from spares, %llu kept, %llu dropped", std::get<2>(k), std::get<3>(k), std::get<4>(k),
            std::get<1>(k), std::get<5>(k), std::get<6>(k), std::get<8>(k), (unsigned long long)v[0], (unsigned long long)v[1], (unsigned long long)v[2], (unsigned long long)v[3]);
        kind_stats.clear();
        std::lock_guard sl{spare_mutex};
        for (auto& [k, kind] : spare_images) RT_LOG("  spare %ux%ux%u vkfmt %u levels %u layers %u: %zu images, %.0f MiB, oldest %llu frames, newest %llu frames", std::get<2>(k), std::get<3>(k), std::get<4>(k), std::get<1>(k), std::get<5>(k), std::get<6>(k),
            kind.images.size(), kind.bytes / 1048576.0, (unsigned long long)(CurrentFrameNumber() - kind.images.front().frame), (unsigned long long)(CurrentFrameNumber() - kind.images.back().frame));
    }
    if (std::getenv("OOT_IMAGE_TRACE")) {
        std::lock_guard l{live_kinds_mutex}; std::vector<std::pair<s64,std::string>> top;
        for (auto& [k,v] : live_kinds) if (v.first) { char b[160]; std::snprintf(b,sizeof b,"%ux%u fmt %u%s%s: %lld images, %.0f MiB",std::get<0>(k),std::get<1>(k),std::get<2>(k),std::get<3>(k)?" attachment":"",std::get<4>(k)?" astc":"",(long long)v.first,v.second/1048576.0); top.push_back({v.second,b}); }
        std::sort(top.rbegin(),top.rend());
        for (size_t i=0;i<top.size()&&i<14;++i) RT_LOG("  live %s",top[i].second.c_str());
        ReportImageAges();
    }
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
