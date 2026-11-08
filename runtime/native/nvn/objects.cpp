// NVN buffers, samplers and texture/sampler pools.
//
// Objects created from a builder keep a copy of the builder bytes at their start, so the SDK
// driver's builder getters double as object getters; host state follows the copy.

#include <vector>

#include "../../native.h"
#include "memory.h"
#include "batch.h"
#include "objects.h"
#include "texture.h"

using namespace rt;
using namespace nvn;

namespace nvn {

u64 DriverFunction(const char* name);

TexturePool* HostTexturePool(u64 object) { return *reinterpret_cast<TexturePool**>(object + 0x18); }
SamplerPool* HostSamplerPool(u64 object) { return *reinterpret_cast<SamplerPool**>(object + 0x18); }

} // namespace nvn

namespace {

// Forwards an object getter to the driver's getter for the builder whose bytes the object holds.
void CallBuilderGetter(Cpu* c, const char* builder_getter) { c->x[0] = CallGuest(c, DriverFunction(builder_getter), {c->x[0], c->x[1], c->x[2], c->x[3]}); }

} // namespace

// ---- buffers: NVNbufferBuilder +0x10 memory pool, +0x18 offset, +0x20 size --------------------
NATIVE("nvnBufferInitialize") {
    std::memcpy(Ptr(c, 0), Ptr(c, 1), BufferBuilderSize);
    Ret(c, 1);
}
NATIVE("nvnBufferFinalize") {}
NATIVE("nvnBufferSetDebugLabel") {}
NATIVE("nvnBufferGetAddress") {
    const u8* b = Ptr<u8>(c, 0);
    u64 pool, offset;
    std::memcpy(&pool, b + 0x10, 8);
    std::memcpy(&offset, b + 0x18, 8);
    Ret(c, PoolOf(pool)->address + offset);
}
NATIVE("nvnBufferMap") {
    const u8* b = Ptr<u8>(c, 0);
    u64 pool, offset;
    std::memcpy(&pool, b + 0x10, 8);
    std::memcpy(&offset, b + 0x18, 8);
    Ret(c, PoolOf(pool)->address + offset);
}
NATIVE("nvnBufferGetMemoryPool") { CallBuilderGetter(c, "nvnBufferBuilderGetMemoryPool"); }
NATIVE("nvnBufferGetMemoryOffset") { CallBuilderGetter(c, "nvnBufferBuilderGetMemoryOffset"); }
NATIVE("nvnBufferGetSize") { CallBuilderGetter(c, "nvnBufferBuilderGetSize"); }
NATIVE("nvnBufferFlushMappedRange") {}
NATIVE("nvnBufferInvalidateMappedRange") { // (buffer, offset, size): the CPU is about to read GPU results there
    const u8* b = Ptr<u8>(c, 0);
    u64 pool, offset;
    std::memcpy(&pool, b + 0x10, 8);
    std::memcpy(&offset, b + 0x18, 8);
    const u64 at = c->x[1], size = c->x[2];
    nvn::WaitForWrites(PoolOf(pool)->address + offset + at, size);
}
NATIVE("nvnBufferGetDebugID") { Ret(c, c->x[0] & 0xffffffff); }

// ---- samplers -----------------------------------------------------------------------------------
NATIVE("nvnSamplerInitialize") {
    std::memcpy(Ptr(c, 0), Ptr(c, 1), SamplerBuilderSize);
    *reinterpret_cast<VkSampler*>(c->x[0] + SamplerBuilderSize) = VK_NULL_HANDLE;
    Ret(c, 1);
}
NATIVE("nvnSamplerFinalize") {}
NATIVE("nvnSamplerSetDebugLabel") {}
NATIVE("nvnSamplerGetMinMagFilter") { CallBuilderGetter(c, "nvnSamplerBuilderGetMinMagFilter"); }
NATIVE("nvnSamplerGetWrapMode") { CallBuilderGetter(c, "nvnSamplerBuilderGetWrapMode"); }
NATIVE("nvnSamplerGetLodClamp") { CallBuilderGetter(c, "nvnSamplerBuilderGetLodClamp"); }
NATIVE("nvnSamplerGetLodBias") { CallBuilderGetter(c, "nvnSamplerBuilderGetLodBias"); }
NATIVE("nvnSamplerGetCompare") { CallBuilderGetter(c, "nvnSamplerBuilderGetCompare"); }
NATIVE("nvnSamplerGetBorderColor") { CallBuilderGetter(c, "nvnSamplerBuilderGetBorderColor"); }
NATIVE("nvnSamplerGetBorderColori") { CallBuilderGetter(c, "nvnSamplerBuilderGetBorderColori"); }
NATIVE("nvnSamplerGetBorderColorui") { CallBuilderGetter(c, "nvnSamplerBuilderGetBorderColorui"); }
NATIVE("nvnSamplerGetMaxAnisotropy") { CallBuilderGetter(c, "nvnSamplerBuilderGetMaxAnisotropy"); }
NATIVE("nvnSamplerGetReductionFilter") { CallBuilderGetter(c, "nvnSamplerBuilderGetReductionFilter"); }
NATIVE("nvnSamplerCompare") { Ret(c, std::memcmp(Ptr(c, 0), Ptr(c, 1), SamplerBuilderSize) == 0); }
NATIVE("nvnSamplerGetDebugID") { Ret(c, c->x[0] & 0xffffffff); }

// ---- texture and sampler pools: object +0 memory pool, +8 offset, +0x10 count, +0x18 host -------
NATIVE("nvnTexturePoolInitialize") {
    u8* o = Ptr<u8>(c, 0);
    std::memcpy(o, &c->x[1], 8);
    std::memcpy(o + 8, &c->x[2], 8);
    const u32 count = (u32)c->x[3];
    std::memcpy(o + 0x10, &count, 4);
    auto* pool = new TexturePool{};
    pool->entries.resize(count);
    *reinterpret_cast<TexturePool**>(o + 0x18) = pool;
    Ret(c, 1);
}
NATIVE("nvnTexturePoolFinalize") { delete HostTexturePool(c->x[0]); }
NATIVE("nvnTexturePoolSetDebugLabel") {}
NATIVE("nvnTexturePoolGetMemoryPool") { Ret(c, *Ptr<u64>(c, 0)); }
NATIVE("nvnTexturePoolGetMemoryOffset") { Ret(c, Ptr<u64>(c, 0)[1]); }
NATIVE("nvnTexturePoolGetSize") { Ret(c, *reinterpret_cast<u32*>(c->x[0] + 0x10)); }
NATIVE("nvnTexturePoolRegisterTexture") {
    nvn::binding_generation.fetch_add(1, std::memory_order_relaxed);
    TexturePool* pool = HostTexturePool(c->x[0]);
    const u32 id = (u32)c->x[1];
    if (id >= pool->entries.size()) Fatal("texture id %u outside pool of %zu", id, pool->entries.size());
    pool->entries[id] = TextureBinding{c->x[2], c->x[3] ? ReadView(c->x[3]) : TextureView{}, false,TextureReference(c->x[2])};
}
NATIVE("nvnTexturePoolRegisterImage") {
    nvn::binding_generation.fetch_add(1, std::memory_order_relaxed);
    TexturePool* pool = HostTexturePool(c->x[0]);
    const u32 id = (u32)c->x[1];
    if (id >= pool->entries.size()) Fatal("image id %u outside pool of %zu", id, pool->entries.size());
    pool->entries[id] = TextureBinding{c->x[2], c->x[3] ? ReadView(c->x[3]) : TextureView{}, true,TextureReference(c->x[2])};
}
NATIVE("nvnSamplerPoolInitialize") {
    u8* o = Ptr<u8>(c, 0);
    std::memcpy(o, &c->x[1], 8);
    std::memcpy(o + 8, &c->x[2], 8);
    const u32 count = (u32)c->x[3];
    std::memcpy(o + 0x10, &count, 4);
    auto* pool = new SamplerPool{};
    pool->entries.resize(count);
    *reinterpret_cast<SamplerPool**>(o + 0x18) = pool;
    Ret(c, 1);
}
NATIVE("nvnSamplerPoolFinalize") { delete HostSamplerPool(c->x[0]); }
NATIVE("nvnSamplerPoolSetDebugLabel") {}
NATIVE("nvnSamplerPoolGetMemoryPool") { Ret(c, *Ptr<u64>(c, 0)); }
NATIVE("nvnSamplerPoolGetMemoryOffset") { Ret(c, Ptr<u64>(c, 0)[1]); }
NATIVE("nvnSamplerPoolGetSize") { Ret(c, *reinterpret_cast<u32*>(c->x[0] + 0x10)); }
NATIVE("nvnSamplerPoolRegisterSampler") {
    nvn::binding_generation.fetch_add(1, std::memory_order_relaxed);
    SamplerPool* pool = HostSamplerPool(c->x[0]);
    const u32 id = (u32)c->x[1];
    if (id >= pool->entries.size()) Fatal("sampler id %u outside pool of %zu", id, pool->entries.size());
    std::memcpy(pool->entries[id].state, Ptr(c, 2), SamplerBuilderSize);
    pool->entries[id].valid = true;
    pool->entries[id].sampler = VK_NULL_HANDLE;
}
NATIVE("nvnSamplerPoolRegisterSamplerBuilder") {
    nvn::binding_generation.fetch_add(1, std::memory_order_relaxed);
    SamplerPool* pool = HostSamplerPool(c->x[0]);
    const u32 id = (u32)c->x[1];
    if (id >= pool->entries.size()) Fatal("sampler id %u outside pool of %zu", id, pool->entries.size());
    std::memcpy(pool->entries[id].state, Ptr(c, 2), SamplerBuilderSize);
    pool->entries[id].valid = true;
    pool->entries[id].sampler = VK_NULL_HANDLE;
}

namespace nvn {

// NVNtextureView: u64 at +0 = layer base (0-15), layer count (16-31), level base (32-39), level count
// (40-47), target (48-55), set flags (56 levels, 57 layers, 58 format, 59 swizzle, 60 depth-stencil
// mode, 61 target), depth-stencil mode (62-63); +8 format; +0xc swizzle bytes.
TextureView ReadView(u64 object) {
    TextureView v;
    u64 bits;
    std::memcpy(&bits, reinterpret_cast<void*>(object), 8);
    v.present = true;
    if (bits >> 56 & 1) { v.levels_base = (bits >> 32) & 0xff; v.levels_count = (bits >> 40) & 0xff; }
    if (bits >> 57 & 1) { v.layers_base = bits & 0xffff; v.layers_count = (bits >> 16) & 0xffff; }
    if (bits >> 58 & 1) std::memcpy(&v.format, reinterpret_cast<void*>(object + 8), 4);
    if (bits >> 59 & 1)
        for (int i = 0; i < 4; ++i) v.swizzle[i] = reinterpret_cast<const u8*>(object + 0xc)[i];
    if (bits >> 60 & 1) v.depth_stencil_mode = (u32)(bits >> 62) + 1; // stored +1 so 0 = unset
    if (bits >> 61 & 1) v.target = (u32)((bits >> 48) & 0xff) + 1;
    return v;
}

} // namespace nvn
