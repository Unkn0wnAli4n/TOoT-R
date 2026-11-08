// Host textures for NVN textures.
#pragma once
#include <atomic>

#include "objects.h"
#include "vk.h"

namespace nvn {
u64 CurrentFrameNumber(); // render.cpp
// Bumped by every texture/sampler registration and texture object change (descriptor caching).
inline std::atomic<u64> binding_generation{0};

enum class Target : u32 { T1D, T2D, T3D, T1DArray, T2DArray, T2DMultisample, T2DMultisampleArray, Rectangle, Cube, CubeArray, Buffer };

struct FormatInfo {
    VkFormat vk = VK_FORMAT_UNDEFINED;
    u32 block_w = 1, block_h = 1, bytes = 4;
    bool depth = false, stencil = false;
};
FormatInfo GetFormat(u32 nvn_format);

struct Texture : std::enable_shared_from_this<Texture> {
    Texture() = default;
    u64 object = 0; // guest NVNtexture
    u32 format = 0;
    Target target = Target::T2D;
    u32 width = 1, height = 1, depth = 1, levels = 1, layers = 1, samples = 1;
    u32 flags = 0;
    u32 swizzle[4]{2,3,4,5}, depth_stencil_mode=0;
    u64 storage = 0; // guest address of the texture memory
    FormatInfo info;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    u64 allocation_size=0;
    bool owner_finalized=false,gpu_retirement_pending=false;
    VkImageView view = VK_NULL_HANDLE; // full view with the texture's own format and swizzle
    std::vector<std::pair<TextureView, VkImageView>> views;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    bool gpu_written = false; // rendered to: contents live on the host GPU
    bool uploaded = false;
    u64 upload_frame = 0; // (OOT_REUPLOAD_TEXTURES diagnostic)
    u32 upload_sequence = 0, checked_sequence = 0; // CPU-write tracking (ArmTextureMemory)
    bool decoded_astc = false;
    bool bc3 = false; // decoded ASTC kept as BC3 (re-encoded on the GPU: a quarter of RGBA8's memory)
    bool decoded_rgb9e5 = false; // RGB9E5 (no host storage-image support) kept as RGBA16F
    struct Mip {
        u32 width, height, depth, block_height, block_depth;
        u64 offset, storage_size;
    };
    std::vector<Mip> mips;
    u64 storage_size = 0, layer_stride = 0;
    u32 block_width = 0, stride = 0;
};

Texture* HostTexture(u64 object);
void InstallTexture(u64 object,Texture* texture); // transfers ownership to the guest object
std::shared_ptr<Texture> TextureReference(u64 object);
void FinalizeTexture(u64 object);
VkImageAspectFlags AspectOf(const Texture& t);
// Records a layout transition of the whole texture.
void Transition(VkCommandBuffer cmd, Texture& t, VkImageLayout layout);
// Texture uploads from guest memory since the last report (diagnostics).
struct UploadStats { u64 count = 0, bytes = 0; };
extern UploadStats upload_stats;
void EnsureImage(Texture& t);
void EnsureUploaded(Texture& t);
void AdoptCachedImage(Texture& t);
void ReportImageCache(); // (diagnostics, every 300 frames)
void TrimImageCache(u64 keep_bytes); // releases cached GPU-written images beyond keep_bytes
void RetireTexture(Texture* t); // takes ownership; destruction waits for pending GPU work
struct TextureCopy;
std::vector<u8> ReadTexels(Texture& t);
void CopyTextureToGuest(Texture& t, const TextureCopy& region, u64 address, u64 row_stride, u64 image_stride);
void CopyTexture(Texture& source, const TextureCopy& source_region,
                 Texture& destination, const TextureCopy& destination_region, u32 flags);
VkImageView ImageView(Texture& t, const TextureView& view = {});
VkImageView AttachmentView(Texture& t, const TextureView& view = {});
// The byte offset within one block-linear mip, using the Switch 2 GOB permutation.
u64 TexelStorageOffset(const Texture& t, u32 level, u32 x_byte, u32 y_block, u32 z);

} // namespace nvn
