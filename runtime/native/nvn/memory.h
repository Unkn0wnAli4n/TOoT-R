// NVN memory pools on Vulkan. A pool's storage is guest memory; its GPU address equals its CPU
// address. Each pool imports its pages into Vulkan and exposes them as one buffer, so data the game
// writes is read by the GPU in place.
#pragma once

#include <map>
#include <memory>
#include <shared_mutex>

#include "vk.h"

namespace nvn {

enum PoolFlags : u32 {
    PoolCpuNoAccess = 0x1,
    PoolCpuUncached = 0x2,
    PoolCpuCached = 0x4,
    PoolGpuNoAccess = 0x8,
    PoolGpuUncached = 0x10,
    PoolGpuCached = 0x20,
    PoolShaderCode = 0x40,
    PoolCompressible = 0x80,
    PoolPhysical = 0x100,
    PoolVirtual = 0x200,
};

struct Pool;
// Guest pages imported into Vulkan, shared by every pool inside them (see ImportPool).
struct HostImport {
    ~HostImport();
    u64 base = 0, size = 0;
    bool backed = false; // (GPU memory mapped over the guest range instead of an import of it: memory.cpp, AMD)
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
};
struct VirtualMapping {
    u64 size = 0, physical_offset = 0;
    std::shared_ptr<Pool> physical;
};

struct Pool {
    ~Pool();
    u64 address = 0; // guest storage (= GPU address)
    u64 size = 0;
    u32 flags = 0;
    u64 import_base = 0; // page-aligned start of the imported range
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE; // the imported guest pages (a shared import's buffer)
    u64 buffer_offset = 0;            // where import_base lies in `buffer`
    std::shared_ptr<HostImport> host_import;
    // GPU-cached pools also get a copy in device memory (same layout as `buffer`), which all GPU
    // work uses: CPU writes are uploaded at submission and GPU writes copied back (see Submit).
    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    VkBuffer device_buffer = VK_NULL_HANDLE;
    u64 import_size = 0;
    bool copy_failed = false; // no device copy (budget or allocation)
    std::map<u64, VirtualMapping> mappings; // virtual offset -> physical storage
};

struct BufferRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    Pool* pool = nullptr;
    u64 cpu_address = 0, available = 0;
};

// Finds the pool containing a GPU address (nullptr if none).
BufferRange LookupAddress(u64 address);
// The same, for binding the range to GPU work: may first give its pool a device-memory copy
// (recording a copy into the open batch).
BufferRange GpuLookupAddress(u64 address);
u64 PoolGeneration(); // changes when pools or mappings change
// Device address of the guest-address translation table used by shaders' global memory accesses.
// update: rewrite it first if pools changed (waits for work that may still read it); only draws
// and dispatches whose shaders access global memory need that.
u64 GlobalAddressTable(bool update = true);
// Resolves GPU virtual storage to its imported physical pool. Never dereference a
// virtual GPU address directly: its host reservation deliberately has no CPU access.
void* ResolveAddress(u64 address, u64 size);
Pool* PoolOf(u64 guest_pool_object);

// Device-memory copies of pools (Pool::device_buffer). Called by Submit, with the batch lock held:
// records into `uploads` the copies of guest pages the CPU wrote since the last submission
// (to run before the batch), and into `cmd` (at the batch end) the copies of the batch's GPU writes
// back to guest memory. Returns a lock that must be held until the batch was submitted.
// Storage writes are copied back by `writeback` (a submission after the batch) when given: their pages go
// to `pending`, for MarkWritebackPending once submitted.
std::shared_lock<std::shared_mutex> RecordPoolTransfers(VkCommandBuffer uploads, bool* any_uploads, VkCommandBuffer cmd,
    const std::map<u64, u64>& gpu_writes, const std::map<u64, u64>& gpu_reports,
    VkCommandBuffer writeback = VK_NULL_HANDLE, std::vector<std::pair<u64, u64>>* pending = nullptr);
// Pages of `ranges` are inaccessible until the write-back copy `serial` has completed (WritebackDone).
void MarkWritebackPending(const std::vector<std::pair<u64, u64>>& ranges, u64 serial);
void WritebackDone(const std::vector<std::pair<u64, u64>>& ranges, u64 serial);
// Before the host kernel writes guest memory (file reads): such writes cannot be tracked by faults.
void PrepareHostWrite(u64 address, u64 size);
// CPU-write tracking of uploaded texture memory: ArmTextureMemory write-protects the range and
// returns the current write sequence; TextureMemoryWritten tells whether the CPU wrote any of it
// (at 64 KiB granularity) after that sequence. TextureWriteSequence changes with every such write.
u32 ArmTextureMemory(u64 address, u64 size);
// Whether imported guest pages may be write-protected to find CPU writes. Not with AMD's drivers: amdgpu checks
// every imported (userptr) page at each submission, and one read-only page gets the submission rejected and the
// device lost (measured on the Steam Deck). Textures are then checked by content (texture_data.cpp).
bool PageProtectionUsable();
// GPU results kept in a pool's device copy (batch.cpp): inaccessible to the CPU until copied back.
void ProtectDeviceAhead(u64 start, u64 end);
void SettleDeviceAhead(u64 start, u64 end);
bool TextureMemoryWritten(u64 address, u64 size, u32 since);
u32 TextureWriteSequence();
void ReportPoolStats();
// Diagnostic (OOT_READ_WATCH): log CPU accesses to pages of `ranges` until UnwatchPages.
void WatchGpuWrites(const std::map<u64, u64>& ranges, std::vector<u64>* pages);
void UnwatchPages(const std::vector<u64>& pages);

} // namespace nvn
