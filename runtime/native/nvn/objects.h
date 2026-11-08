// Host state of NVN objects.
#pragma once

#include <vector>
#include <memory>

#include "vk.h"

namespace nvn {

constexpr size_t BufferBuilderSize = 0x28;
constexpr size_t SamplerBuilderSize = 0x30;
constexpr size_t TextureBuilderSize = 0x68;

// NVNtextureView fields the renderer uses (0 = take the texture's own value).
struct TextureView {
    bool present = false;
    u32 levels_base = 0, levels_count = 0;
    u32 layers_base = 0, layers_count = 0;
    u32 format = 0;
    u32 swizzle[4] = {};
    u32 depth_stencil_mode = 0;
    u32 target = 0;
};
TextureView ReadView(u64 view_object);

struct Texture;
struct TextureBinding {
    u64 texture = 0; // guest NVNtexture
    TextureView view;
    bool image = false;
    std::shared_ptr<Texture> host; // descriptor snapshot survives guest object finalization/reuse
};

struct TexturePool {
    std::vector<TextureBinding> entries;
};

struct SamplerEntry {
    u8 state[SamplerBuilderSize] = {};
    bool valid = false;
    VkSampler sampler = VK_NULL_HANDLE;
    uint64_t settings_generation = 0; // (gfx::Generation() it was made for)
};

struct SamplerPool {
    std::vector<SamplerEntry> entries;
};

TexturePool* HostTexturePool(u64 object);
SamplerPool* HostSamplerPool(u64 object);

} // namespace nvn
