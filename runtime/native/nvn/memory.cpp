#include <unordered_map>
#include <sched.h>
#include <deque>
#include <bit>
#include <unordered_set>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <pthread.h>
#include <ucontext.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <unistd.h>
#endif
#include <vector>
#include <mutex>
#include <map>
#include <set>

#include "../../logfile.h"
#include "../../native.h"
#include "trace_compat.h"
#include "memory.h"
#include "batch.h"

using namespace rt;

namespace nvn {
extern std::atomic<u64> imports_live; // (live shared imports)
namespace {

std::shared_mutex pools_mutex;
std::map<u64, std::shared_ptr<Pool>> pools_by_address; // storage start -> pool
u64 next_virtual_address = 0x3000000000ULL;
bool table_dirty = true; // pools changed since the address table was written
// (an entry changed or went, not only added: work already submitted may still read it, so the table is written only after
// a flush. A pool that only came - imported at its first GPU use, or new - is nowhere in that work, and its entries are
// written in place at once: a load adds hundreds, and each flush ran into the limit of batches in flight, a third of a
// second of waits in one save load. OOT_TABLE_FLUSH_ALWAYS=1: a flush for every change, as before.)
bool table_changed = true;
std::atomic<u64> pool_generation{1}; // bumped with every pool/mapping change (LookupAddress cache)

// Guest NVNmemoryPool objects hold the host pool pointer in their first word.
Pool*& HostPool(u64 object) { return *reinterpret_cast<Pool**>(object); }

// ---- importing guest pages ----------------------------------------------------------------------
// Each import is a Vulkan memory allocation, and drivers allow few (NVIDIA: 4096). The movie player
// creates thousands of 4 KiB pools and keeps them: small pools are imported when the GPU first looks
// them up (LookupAddressUncached), and a pool inside a live import uses it at an offset. An import is
// released with the last pool using it.
std::mutex imports_mutex;
std::map<u64, std::weak_ptr<HostImport>> imports; // base -> import
// ---- pools in GPU memory (AMD) -------------------------------------------------------------------
// amdgpu checks every page of every imported (userptr) range at each submission: ~11 ms per GiB on the Steam
// Deck, and this game imports ~5 GiB (30-50 ms a submission). Pools of 1 MiB and more are moved into exported
// GPU memory instead, mapped over the guest range (the contents are copied in; pointers stay valid): the
// GPU uses it directly, nothing is checked per submission (0.03 ms measured). Smaller pools are imported.
// When the last pool goes, the contents go back into ordinary memory. OOT_IMPORT_POOLS=1: imports only.
bool BackPools();
std::shared_ptr<HostImport> CreateBacked(u64 base, u64 size);
// NVIDIA's Windows driver takes hold of an import's pages when the GPU first reaches them, and loses the device if a page
// is read-only at that moment (found with a test program on a Tesla T4, driver 616.92: import, protect, first copy =
// device lost; a page the GPU had reached while writable may be protected afterwards). The game's pages are protected
// to find CPU writes (texture memory, the pools with device copies), often before the GPU has used them: that was
// the crash at start-up on Windows. So the GPU reads every page of a new import once, while it is still writable.
// On by default on Windows; OOT_PRIME_IMPORTS=1 / 0 elsewhere or to turn it off.
bool ImportsPrimed() { // (whether new imports get that first read: asked before any is made)
    const char* e = std::getenv("OOT_PRIME_IMPORTS");
#ifdef _WIN32
    const bool on = !e || std::atoi(e) != 0;
#else
    const bool on = e && std::atoi(e) != 0;
#endif
    return on && Vk().copy_queue != Vk().queue;
}
void PrimeImport(VkBuffer buffer, u64 size) {
    static const bool on = [] { const char* e = std::getenv("OOT_PRIME_IMPORTS");
#ifdef _WIN32
        return !e || std::atoi(e) != 0;
#else
        return e && std::atoi(e) != 0;
#endif
    }();
    if (!on || !size) return;
    auto& vk = Vk();
    // (on the second queue: imports are made with the pool locks held, and the main queue's lock is held by threads
    // that wait for those - the presenter for one. A device with a single queue goes without.)
    if (vk.copy_queue == vk.queue) { static bool told = false; if (!told) { told = true; RT_LOG("imports: no second queue to read them on first"); } return; }
    constexpr u64 Chunk = 8u << 20;
    static std::mutex mutex; std::lock_guard lock{mutex};
    static VkCommandPool pool = VK_NULL_HANDLE; static VkBuffer scratch = VK_NULL_HANDLE; static VkFence fence = VK_NULL_HANDLE;
    if (!pool) {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = vk.queue_family; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(vk.device, &pi, nullptr, &pool));
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = Chunk; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &scratch));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk.device, scratch, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory memory; VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &memory)); VK_CHECK(vkBindBufferMemory(vk.device, scratch, memory, 0));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK_CHECK(vkCreateFence(vk.device, &fi, nullptr, &fence));
    }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd; VK_CHECK(vkAllocateCommandBuffers(vk.device, &ai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    for (u64 at = 0; at < size; at += Chunk) { const VkBufferCopy region{at, 0, std::min(Chunk, size - at)}; vkCmdCopyBuffer(cmd, buffer, scratch, 1, &region); }
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    { std::lock_guard copy{vk.copy_mutex}; VK_CHECK(vkQueueSubmit(vk.copy_queue, 1, &si, fence)); }
    VK_CHECK(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, UINT64_MAX)); VK_CHECK(vkResetFences(vk.device, 1, &fence));
    vkFreeCommandBuffers(vk.device, pool, 1, &cmd);
    static std::atomic<u64> total{0}; static std::atomic<u32> told{0};
    total += size; if (told++ == 0) RT_LOG("imports: each is read once by the GPU while its pages are writable");
}
std::shared_ptr<HostImport> CreateImport(u64 base, u64 size) {
    auto& vk = Vk();
    // (an import of pages GPU memory is mapped over would get every submission rejected)
    for (auto& [b, w] : imports) if (auto i = w.lock(); i && i->backed && i->base < base + size && base < i->base + i->size) {
        RT_LOG("import %#llx+%#llx overlaps GPU memory at %#llx+%#llx: not imported", (unsigned long long)base, (unsigned long long)size,
               (unsigned long long)i->base, (unsigned long long)i->size);
        return nullptr;
    }
    rt::EnsureCommitted(reinterpret_cast<void*>(base), size); // (Windows: the driver cannot import pages not yet committed)
    static auto get_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetDeviceProcAddr(vk.device, "vkGetMemoryHostPointerPropertiesEXT"));
    VkMemoryHostPointerPropertiesEXT props{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (const VkResult r = get_props(vk.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, reinterpret_cast<void*>(base), &props); r != VK_SUCCESS) {
        RT_LOG("import %#llx+%#llx: host pointer properties failed (%d)", (unsigned long long)base, (unsigned long long)size, (int)r); return nullptr; }
    auto imp = std::make_shared<HostImport>();
    imp->base = base; imp->size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    bi.pNext = &ext;
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &imp->buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk.device, imp->buffer, &req);
    VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import.pHostPointer = reinterpret_cast<void*>(base);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    flags.pNext = &import;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags;
    ai.allocationSize = size;
    ai.memoryTypeIndex = FindMemoryType(props.memoryTypeBits & req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (const VkResult r = vkAllocateMemory(vk.device, &ai, nullptr, &imp->memory); r != VK_SUCCESS) {
        RT_LOG("import %#llx+%#llx: allocation failed (%d)", (unsigned long long)base, (unsigned long long)size, (int)r);
#ifdef __linux__
        if (FILE* maps = std::fopen("/proc/self/maps", "r")) { // (which host mappings the range spans)
            char line[256];
            while (std::fgets(line, sizeof line, maps)) {
                unsigned long long a, b;
                if (std::sscanf(line, "%llx-%llx", &a, &b) == 2 && a < base + size && b > base) RT_LOG("  mapping %s", line);
            }
            std::fclose(maps);
        }
#endif
        vkDestroyBuffer(vk.device, imp->buffer, nullptr); imp->buffer = VK_NULL_HANDLE; return nullptr; }
    VK_CHECK(vkBindBufferMemory(vk.device, imp->buffer, imp->memory, 0));
    PrimeImport(imp->buffer, size);
    return imp;
}
namespace track { bool UnprotectForImport(u64 base, u64 size); }
bool BackPools() {
#ifdef __linux__
    static const bool on = [] {
        const bool v = (Vk().placed_maps || Vk().dma_buf) && !PageProtectionUsable() && !std::getenv("OOT_IMPORT_POOLS");
        if (v) RT_LOG("memory pools: in GPU memory mapped at their addresses (%s)", Vk().placed_maps ? "placed maps: every pool, nothing checked per submission" : "exported memory, 1 MiB and larger pools");
        return v;
    }();
    return on;
#else
    return false;
#endif
}
std::shared_ptr<HostImport> CreatePlaced(u64 base, u64 size);
// VK_EXT_map_memory_placed: ordinary (not exported) GPU memory, mapped by the driver at the pool's guest address.
// Such memory stays valid in the GPU's address space; exported memory is listed in every submission (12 ms a
// submission with OoT's ~4000 pools on the Deck), userptr imports are checked page by page (18-40 ms).
std::shared_ptr<HostImport> CreatePlaced(u64 base, u64 size) {
    auto& vk = Vk();
    static auto map2 = reinterpret_cast<PFN_vkMapMemory2KHR>(vkGetDeviceProcAddr(vk.device, "vkMapMemory2KHR"));
    auto imp = std::make_shared<HostImport>();
    imp->base = base; imp->size = size; imp->backed = true; imp->placed = true;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (vkCreateBuffer(vk.device, &bi, nullptr, &imp->buffer) != VK_SUCCESS) { imp->backed = false; return nullptr; }
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk.device, imp->buffer, &req);
    u32 type = ~0u; // (CPU-cached and coherent, not device-local where such a type exists)
    for (u32 pass = 0; pass < 2 && type == ~0u; ++pass)
        for (u32 t = 0; t < vk.memory_properties.memoryTypeCount; ++t) {
            const auto f = vk.memory_properties.memoryTypes[t].propertyFlags;
            const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | (pass == 0 ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
            if ((req.memoryTypeBits >> t & 1) && (f & want) == want && !(f & 0xc0)) { type = t; break; }
        }
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags; ai.allocationSize = std::max<u64>(req.size, size); ai.memoryTypeIndex = type;
    void* staging = nullptr;
    if (type == ~0u || vkAllocateMemory(vk.device, &ai, nullptr, &imp->memory) != VK_SUCCESS ||
        vkBindBufferMemory(vk.device, imp->buffer, imp->memory, 0) != VK_SUCCESS ||
        vkMapMemory(vk.device, imp->memory, 0, VK_WHOLE_SIZE, 0, &staging) != VK_SUCCESS) {
        RT_LOG("pool %#llx+%#llx: GPU memory unavailable, imported instead", (unsigned long long)base, (unsigned long long)size);
        imp->backed = imp->placed = false; // (the destructor frees what was made)
        return nullptr;
    }
    // The contents so far (the memory keeps them), then the memory in place of the guest pages.
    std::memcpy(staging, reinterpret_cast<const void*>(base), size);
    vkUnmapMemory(vk.device, imp->memory);
    VkMemoryMapPlacedInfoEXT at{VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT};
    at.pPlacedAddress = reinterpret_cast<void*>(base);
    VkMemoryMapInfoKHR mi{VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR};
    mi.pNext = &at; mi.flags = VK_MEMORY_MAP_PLACED_BIT_EXT; mi.memory = imp->memory; mi.offset = 0; mi.size = VK_WHOLE_SIZE;
    void* mapped = nullptr;
    if (map2(vk.device, &mi, &mapped) != VK_SUCCESS || mapped != reinterpret_cast<void*>(base))
        Fatal("cannot map GPU memory at pool %#llx+%#llx", (unsigned long long)base, (unsigned long long)size);
    static std::atomic<u64> total{0}; total += size;
    static std::atomic<u32> shown{0};
    if (shown++ < 6) RT_LOG("pool %#llx+%#llx in GPU memory at its address (%.0f MiB so far)", (unsigned long long)base, (unsigned long long)size, total / 1048576.0);
    return imp;
}
std::shared_ptr<HostImport> CreateBacked(u64 base, u64 size) {
    if (Vk().placed_maps) return CreatePlaced(base, size);
#ifdef __linux__
    auto& vk = Vk();
    static auto get_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(vk.device, "vkGetMemoryFdKHR"));
    auto imp = std::make_shared<HostImport>();
    imp->base = base; imp->size = size; imp->backed = true;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &ext; bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (vkCreateBuffer(vk.device, &bi, nullptr, &imp->buffer) != VK_SUCCESS) { imp->backed = false; return nullptr; }
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk.device, imp->buffer, &req);
    // (CPU-cached and coherent: the game reads what the GPU wrote; not device-local where a cached type exists)
    u32 type = ~0u;
    for (u32 pass = 0; pass < 2 && type == ~0u; ++pass)
        for (u32 t = 0; t < vk.memory_properties.memoryTypeCount; ++t) {
            const auto f = vk.memory_properties.memoryTypes[t].propertyFlags;
            const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | (pass == 0 ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
            if ((req.memoryTypeBits >> t & 1) && (f & want) == want && !(f & 0xc0)) { type = t; break; } // (0xc0: AMD device-coherent/uncached)
        }
    VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.pNext = &export_info; flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags; ai.allocationSize = req.size; ai.memoryTypeIndex = type;
    int fd = -1;
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    void* staging = MAP_FAILED;
    if (type == ~0u || vkAllocateMemory(vk.device, &ai, nullptr, &imp->memory) != VK_SUCCESS ||
        vkBindBufferMemory(vk.device, imp->buffer, imp->memory, 0) != VK_SUCCESS ||
        (gi.memory = imp->memory, gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, get_fd(vk.device, &gi, &fd)) != VK_SUCCESS ||
        (staging = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) {
        RT_LOG("pool %#llx+%#llx: GPU memory unavailable, imported instead", (unsigned long long)base, (unsigned long long)size);
        if (fd >= 0) close(fd);
        imp->backed = false; // (the destructor frees what was made)
        return nullptr;
    }
    // The contents so far, then the GPU memory in place of the guest pages.
    std::memcpy(staging, reinterpret_cast<const void*>(base), size);
    munmap(staging, size);
    if (mmap(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED)
        Fatal("cannot map GPU memory over pool %#llx+%#llx", (unsigned long long)base, (unsigned long long)size);
    close(fd);
    static std::atomic<u64> total{0};
    total += size;
    static std::atomic<u32> shown{0};
    if (shown++ < 12) RT_LOG("pool %#llx+%#llx in GPU memory (%.0f MiB so far)", (unsigned long long)base, (unsigned long long)size, total / 1048576.0);
    return imp;
#else
    (void)base; (void)size; return nullptr;
#endif
}
void ImportPool(Pool& p) {
    auto& vk = Vk();
    const u64 align = vk.host_import_alignment;
    p.import_base = p.address & ~(align - 1);
    const u64 end = (p.address + p.size + align - 1) & ~(align - 1);
    p.import_size = end - p.import_base;
    std::lock_guard lock{imports_mutex};
    // A live import holding the whole range?
    auto it = imports.upper_bound(p.import_base);
    if (it != imports.begin()) {
        --it;
        if (auto imp = it->second.lock(); imp && imp->base <= p.import_base && imp->base + imp->size >= end) {
            p.host_import = imp; p.buffer = imp->buffer; p.buffer_offset = p.import_base - imp->base; return; }
    }
    // A new import of exactly the range. (Imports must not overlap: the driver refuses pages already
    // imported. Larger shared chunks were tried: a later pool across a chunk's end could not be imported.)
    // (OOT_BACK_MIN_KB: the smallest pool put in GPU memory. Small pools imported were ~3800 userptr objects on the Deck,
    // and amdgpu checks each one at every submission: 18 ms a submission)
    // (placed maps cost nothing per submission: every pool; exported memory is listed in every submission: large ones only)
    static const u64 back_min = std::getenv("OOT_BACK_MIN_KB") ? std::strtoull(std::getenv("OOT_BACK_MIN_KB"), nullptr, 10) << 10 : Vk().placed_maps ? 4096 : (1u << 20);
    if (BackPools() && p.import_size >= back_min) {
        bool overlap = false;
        for (auto& [b, w] : imports) if (auto i = w.lock(); i && i->base < end && p.import_base < i->base + i->size) overlap = true;
        // (write-protected texture pages in the range: writable and counted as written, before their contents are copied
        // and again after the GPU memory took their place - it is mapped writable whatever was armed meanwhile)
        if (!overlap) track::UnprotectForImport(p.import_base, p.import_size);
        if (!overlap) if (auto backed = CreateBacked(p.import_base, p.import_size)) {
            track::UnprotectForImport(p.import_base, p.import_size);
            if (std::getenv("OOT_ADDRESS_REPORT")) RT_LOG("pool %#llx+%#llx: GPU memory buffer %#llx", (unsigned long long)p.import_base, (unsigned long long)p.import_size, (unsigned long long)(u64)backed->buffer);
            ++imports_live;
            imports[backed->base] = backed;
            p.host_import = backed; p.buffer = backed->buffer; p.buffer_offset = 0;
            return;
        }
    }
    // Write-protected pages (CPU-write tracking) cannot be imported, and the NVIDIA driver then also refuses
    // every later import of the same size, wherever it is: the pages are made writable first.
    track::UnprotectForImport(p.import_base, p.import_size);
    std::shared_ptr<HostImport> imp = CreateImport(p.import_base, p.import_size);
    if (!imp) { // (a size that failed before: one page more, when that page is free)
        const auto next = imports.lower_bound(end);
        const bool free = next == imports.end() || next->first >= end + align || next->second.expired();
        if (free) {
            track::UnprotectForImport(end, align);
            if ((imp = CreateImport(p.import_base, p.import_size + align))) RT_LOG("import %#llx+%#llx: imported one page larger",
                (unsigned long long)p.import_base, (unsigned long long)p.import_size);
        }
    }
    if (!imp) { std::string live; for (auto& [b, w] : imports) if (auto i = w.lock()) { char t[64]; std::snprintf(t, sizeof t, " %#llx+%#llx", (unsigned long long)i->base, (unsigned long long)i->size); live += t; } RT_LOG("live imports:%s", live.c_str()); }
    if (!imp) Fatal("cannot import guest memory %#llx+%#llx into Vulkan (%llu imports live)", (unsigned long long)p.import_base,
                    (unsigned long long)p.import_size, (unsigned long long)imports_live.load());
    ++imports_live;
    if (std::getenv("OOT_ADDRESS_REPORT")) RT_LOG("pool %#llx+%#llx: imported as buffer %#llx", (unsigned long long)p.import_base, (unsigned long long)p.import_size, (unsigned long long)(u64)imp->buffer);
    for (auto e = imports.begin(); e != imports.end();) e = e->second.expired() ? imports.erase(e) : std::next(e); // (forget released ones)
    imports[imp->base] = imp;
    p.host_import = imp; p.buffer = imp->buffer; p.buffer_offset = p.import_base - imp->base;
}


// ---- device-memory copies of GPU-cached pools -------------------------------------------------
// The GPU reads imported guest pages across the bus; GPU-cached pools get a device-memory copy.
// CPU writes are found with write protection: the first write to a protected page saves its
// contents (the "shadow") and unprotects it. At submission, each dirty page is protected again
// and the bytes that differ from its shadow are copied to the device copy before the batch runs
// (bytes the GPU wrote and the CPU did not touch are left alone). GPU writes of the batch are
// copied back to guest memory at its end, so CPU readers see them after the batch completed.
namespace track {
constexpr u64 PageSize = 4096;
enum : u8 { Untracked = 0, Protected = 1, Dirty = 2 };
constexpr u64 LeafPages = 1ull << 18; // one leaf per GiB of address space
std::atomic<u8*> leaves[1u << 18];    // 48-bit addresses
u8* Leaf(u64 page, bool create) {
    const u64 top = page >> 30;
    if (top >= (1u << 18)) return nullptr;
    u8* leaf = leaves[top].load(std::memory_order_acquire);
    if (!leaf && create) {
        u8* fresh = static_cast<u8*>(mmap(nullptr, LeafPages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (fresh == MAP_FAILED) Fatal("cannot allocate page tracking");
        if (leaves[top].compare_exchange_strong(leaf, fresh)) leaf = fresh;
        else munmap(fresh, LeafPages);
    }
    return leaf;
}
u8* State(u64 page, bool create = false) { u8* leaf = Leaf(page, create); return leaf ? &leaf[(page >> 12) & (LeafPages - 1)] : nullptr; }

// Held briefly by the fault handler and by submission (which never writes guest pages).
std::atomic_flag lock = ATOMIC_FLAG_INIT;
struct Guard { Guard() { while (lock.test_and_set(std::memory_order_acquire)) __builtin_ia32_pause(); } ~Guard() { lock.clear(std::memory_order_release); } };

constexpr u32 NoShadow = ~0u;
struct Entry { u64 page; u32 shadow; };
constexpr u32 MaxDirty = 1u << 20;
Entry* dirty = nullptr; u32 dirty_count = 0; bool overflow = false;
constexpr u32 ShadowSlots = 32768; // 128 MiB
u8* shadows = nullptr; u32* free_slots = nullptr; u32 free_count = 0;

std::atomic<u64> faults{0}, uploaded{0}, written_back{0}, upload_regions{0};

void Init() {
    static std::once_flag once;
    std::call_once(once, [] {
        dirty = static_cast<Entry*>(mmap(nullptr, MaxDirty * sizeof(Entry), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
        shadows = static_cast<u8*>(mmap(nullptr, (u64)ShadowSlots * PageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
        free_slots = static_cast<u32*>(mmap(nullptr, ShadowSlots * 4, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (dirty == MAP_FAILED || shadows == MAP_FAILED || free_slots == MAP_FAILED) Fatal("cannot allocate page tracking");
        for (u32 i = 0; i < ShadowSlots; ++i) free_slots[i] = ShadowSlots - 1 - i;
        free_count = ShadowSlots;
    });
}
// (lock held)
void Push(u64 page, u32 shadow) {
    if (dirty_count == MaxDirty) { overflow = true; if (shadow != NoShadow) free_slots[free_count++] = shadow; return; }
    dirty[dirty_count++] = {page, shadow};
}
// (lock held) Pages written in many captures: left writable (CapturePoolWrites). On by default on Windows, where a write
// fault is expensive; OOT_HOT_PAGES=1 / 0 elsewhere or to turn it off.
std::unordered_map<u64, u32> heat, hot; u64 hot_generation = 0;
bool HotPagesOn() {
    static const bool on = [] { const char* e = std::getenv("OOT_HOT_PAGES");
#ifdef _WIN32
        return !e || std::atoi(e) != 0;
#else
        return e && std::atoi(e) != 0;
#endif
    }();
    return on;
}
// ---- CPU writes to uploaded texture memory ------------------------------------------------------
// Pages of CPU-uploaded textures are write-protected ("armed") before their texels are read. The
// first CPU write afterwards disarms the page and stamps its 64 KiB chunk with a new sequence
// number; a texture whose chunks carry a number newer than its upload is uploaded again on its
// next use (the game fills text labels, item icons and other menu images after creating them).
enum : u8 { NeverArmed = 0, Armed = 1, Disarmed = 2 };
constexpr u64 ChunkShift = 16;
std::atomic<u8*> armed_leaves[1u << 18];
std::atomic<u32*> chunk_leaves[1u << 18];
std::atomic<u32> write_sequence{1};
template <class T> T* TableEntry(std::atomic<T*>* table, u64 address, u64 shift, bool create) {
    const u64 top = address >> 30; if (top >= (1u << 18)) return nullptr;
    constexpr u64 Bytes = 1ull << 30;
    const u64 count = Bytes >> shift;
    T* leaf = table[top].load(std::memory_order_acquire);
    if (!leaf && create) {
        T* fresh = static_cast<T*>(mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (fresh == MAP_FAILED) Fatal("cannot allocate page tracking");
        if (table[top].compare_exchange_strong(leaf, fresh)) leaf = fresh; else munmap(fresh, count * sizeof(T));
    }
    return leaf ? &leaf[(address & (Bytes - 1)) >> shift] : nullptr;
}
u8* ArmState(u64 page, bool create = false) { return TableEntry(armed_leaves, page, 12, create); }
std::atomic<u64*> back_leaves[1u << 18]; // (see "GPU writes on their way back")
std::atomic<u64> pending_faults{0}, pending_wait_ns{0};
u64* BackSerial(u64 page, bool create = false) { return TableEntry(back_leaves, page, 12, create); }
void SettleRange(u64 address, u64 size);
u32* ChunkSequence(u64 address, bool create = false) { return TableEntry(chunk_leaves, address, ChunkShift, create); }
std::atomic<u64> texture_write_faults{0};
// (lock held) Records a CPU write to an armed page and disarms it (its protection is the caller's).
void Disarm(u64 page) {
    u8* armed = ArmState(page);
    if (!armed || *armed != Armed) return;
    *armed = Disarmed;
    if (u32* chunk = ChunkSequence(page)) __atomic_store_n(chunk, write_sequence.fetch_add(1, std::memory_order_relaxed) + 1, __ATOMIC_RELEASE);
    texture_write_faults.fetch_add(1, std::memory_order_relaxed);
}
// (lock held) Makes a page writable for the CPU.
void MakeWritable(u64 page) { Disarm(page); mprotect(reinterpret_cast<void*>(page), PageSize, PROT_READ | PROT_WRITE); }

// (lock held) Makes a page writable and queues it for upload.
void MarkDirty(u64 page, u8& state, bool save) {
    u32 slot = NoShadow;
    if (save && free_count) { slot = free_slots[--free_count]; std::memcpy(shadows + (u64)slot * PageSize, reinterpret_cast<void*>(page), PageSize); }
    MakeWritable(page);
    state = Dirty;
    Push(page, slot);
}

// Diagnostic (OOT_READ_WATCH): pages GPU work of in-flight batches writes are inaccessible until
// the batch completes; CPU accesses before then are logged (they read results not yet there).
std::atomic<u8*> watch_leaves[1u << 18];
u8* Watch(u64 page, bool create) {
    const u64 top = page >> 30; if (top >= (1u << 18)) return nullptr;
    u8* leaf = watch_leaves[top].load(std::memory_order_acquire);
    if (!leaf && create) {
        u8* fresh = static_cast<u8*>(mmap(nullptr, LeafPages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (watch_leaves[top].compare_exchange_strong(leaf, fresh)) leaf = fresh; else munmap(fresh, LeafPages);
    }
    return leaf ? &leaf[(page >> 12) & (LeafPages - 1)] : nullptr;
}
void RestoreProtection(u64 page) { // (lock held)
    u8* state = State(page);
    mprotect(reinterpret_cast<void*>(page), PageSize, state && *state == Protected ? PROT_READ : PROT_READ | PROT_WRITE);
}

// ---- GPU writes on their way back to guest memory ------------------------------------------------
// A batch's storage writes are copied back to guest memory by a submission of their own, so the batch
// (and the syncs the game waits on) completes without waiting for the copy, which overlaps the next
// frame's recording. Until its copy has completed a page is inaccessible (PROT_NONE): a CPU access
// waits for that copy (a timeline semaphore; no locks), then proceeds. Per page: 0 never pending,
// 1 settled, else 1 + the serial of the copy bringing its contents.
int SettledProtection(u64 page) { // (lock held) the protection page tracking wants for a page
    if (u8* state = State(page); state && *state == Protected) return PROT_READ;
    if (u8* armed = ArmState(page); armed && *armed == Armed) return PROT_READ;
    return PROT_READ | PROT_WRITE;
}
// (lock held) Pages of [first, end) no longer pending get their tracked protection (in runs).
void SettleRun(u64 first, u64 end) {
    u64 run = 0; int prot = -1;
    const auto flush = [&](u64 to) { if (prot >= 0) mprotect(reinterpret_cast<void*>(run), to - run, prot); prot = -1; };
    for (u64 page = first; page < end; page += PageSize) {
        const int want = SettledProtection(page);
        if (want != prot) { flush(page); run = page; prot = want; }
    }
    flush(end);
}
// Write-protection faults of tracked pages; false: not ours (a real crash).
// GPU results not yet copied back (batch.cpp, device_ahead): their pages are inaccessible; a CPU access copies
// the binding back (and waits for it), then proceeds. Page-aligned outward; start -> end.
std::mutex ahead_mutex;
std::map<u64, u64> ahead_pages;
std::atomic<int> settling{0}; // (SettleDeviceAhead between its two steps: HandleFault)
bool AheadAt(u64 address, u64* start, u64* end) {
    std::lock_guard lock{ahead_mutex};
    auto it = ahead_pages.upper_bound(address);
    if (it == ahead_pages.begin()) return false;
    --it;
    if (it->second <= address) return false;
    *start = it->first; *end = it->second;
    return true;
}
bool HandleFault(u64 address, u64 host_pc) {
    const u64 page = address & ~(PageSize - 1);
    if (u64 start, end; AheadAt(address, &start, &end)) {
        static std::atomic<u32> shown{0};
        if (shown++ < 200) { char name[32] = "?"; pthread_getname_np(pthread_self(), name, sizeof name);
            Dl_info info{}; dladdr(reinterpret_cast<void*>(host_pc), &info);
            RT_LOG("CPU reads GPU results at %#llx (pages %#llx-%#llx, host pc +%#llx, thread %s): copying them back",
                   (unsigned long long)address, (unsigned long long)start, (unsigned long long)end, (unsigned long long)(host_pc - (u64)info.dli_fbase), name); }
        nvn::WaitForWrites(page, PageSize);     // (copies that page back; it is settled on the way)
        nvn::SettleDeviceAhead(page, page + PageSize); // (in case another thread recalled it meanwhile)
        return true;
    }
    if (u64* back = BackSerial(page); back && __atomic_load_n(back, __ATOMIC_ACQUIRE)) {
        for (;;) {
            const u64 value = __atomic_load_n(back, __ATOMIC_ACQUIRE);
            if (value > 1) {
                const auto t0 = std::chrono::steady_clock::now();
                WaitWriteback(value - 1);
                pending_faults.fetch_add(1, std::memory_order_relaxed);
                pending_wait_ns.fetch_add((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
                static const bool trace = std::getenv("OOT_PENDING_TRACE") != nullptr;
                if (trace) { static std::atomic<u32> shown{0}; if (shown++ < 40) RT_LOG("CPU touched a page on its way back: %#llx (host pc %#llx)", (unsigned long long)address, (unsigned long long)host_pc); }
            }
            Guard guard;
            if (*back != value) continue; // (pending again for a newer copy, or settled meanwhile)
            *back = 1;
            mprotect(reinterpret_cast<void*>(page), PageSize, SettledProtection(page));
            return true;
        }
    }
    if (u8* watch = Watch(page, false); watch && *watch) {
        Guard guard;
        if (*watch) {
            *watch = 0; RestoreProtection(page);
            static std::atomic<u32> shown{0};
            if (shown++ < 60) RT_LOG("CPU touched GPU-pending %#llx (host pc %#llx)", (unsigned long long)address, (unsigned long long)host_pc);
        }
        return true;
    }
    if (Watch(page, false) && std::getenv("OOT_READ_WATCH")) return true; // (restored by another thread meanwhile)
    if (u8* armed = ArmState(page); armed && *armed != NeverArmed) {
        Guard guard;
        if (*armed != NeverArmed) {
            u8* state = State(page);
            if (state && *state == Protected) { MarkDirty(page, *state, true); faults.fetch_add(1, std::memory_order_relaxed); }
            else MakeWritable(page);
            return true; // (disarmed by another thread meanwhile: retry the access)
        }
    }
    if (u8* state = State(page)) {
        Guard guard;
        if (*state == Protected) { MarkDirty(page, *state, true); faults.fetch_add(1, std::memory_order_relaxed); return true; }
        if (*state == Dirty) return true; // made writable by another thread meanwhile
    }
    // A page between two states: SettleDeviceAhead takes a run out of the ahead map first and gives the pages their
    // protection back a moment later, under the tracking lock. A thread touching such a page in between found it in
    // no table and still inaccessible - reported as a crash (a game thread writing at 0x1035d60a6a after 47,000
    // frames, 2026-11-17; once in 17 loads before). While a settle is under way the access is tried again; so is
    // any fault's first repeat on a thread, after a pause, so that a real bad access is still reported, twice late.
    if (settling.load(std::memory_order_acquire)) { while (settling.load(std::memory_order_acquire)) sched_yield(); return true; }
    { static thread_local u64 last_page = 0; static thread_local int repeats = 0;
      if (last_page != page) { last_page = page; repeats = 0; }
      if (repeats++ < 2) { sched_yield(); { Guard guard; } return true; } }
    return false;
}
#ifdef _WIN32
LONG CALLBACK OnException(EXCEPTION_POINTERS* e) {
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->NumberParameters < 2)
        return EXCEPTION_CONTINUE_SEARCH;
    const u64 address = (u64)e->ExceptionRecord->ExceptionInformation[1];
    if (rt::CommitOnFault(reinterpret_cast<void*>(address), e->ExceptionRecord->ExceptionInformation[0] == 1)) return EXCEPTION_CONTINUE_EXECUTION; // (heap not yet touched: platform.h)
    if (HandleFault(address, (u64)e->ContextRecord->Rip)) return EXCEPTION_CONTINUE_EXECUTION;
    char line[160];
    const int n = std::snprintf(line, sizeof line, "[rt] access violation at %#llx, host pc %#llx\n",
                                (unsigned long long)address, (unsigned long long)e->ContextRecord->Rip);
    std::fwrite(line, 1, (size_t)n, stderr);
    {   // (where, as offsets in tootr.exe: what a log sent with a report is read by)
        const u64 base = (u64)GetModuleHandleW(nullptr);
        void* frames[24]; const USHORT count = CaptureStackBackTrace(0, 24, frames, nullptr);
        std::fprintf(stderr, "[rt]   tootr.exe at %#llx; pc +%#llx; stack:", (unsigned long long)base, (unsigned long long)((u64)e->ContextRecord->Rip - base));
        for (USHORT i = 0; i < count; ++i) std::fprintf(stderr, " +%#llx", (unsigned long long)((u64)frames[i] - base));
        std::fprintf(stderr, "\n");
        // (and by name, with tootr.sym beside the executable: only the frames inside it)
        std::fprintf(stderr, "[rt]   in:");
        const auto named = [&](u64 pc) { unsigned long long into = 0; const char* s = pc >= base && pc - base < (1ull << 31) ? rt::SymbolAt(pc - base, &into) : ""; if (*s) std::fprintf(stderr, " %s+%#llx <-", s, into); };
        named((u64)e->ContextRecord->Rip);
        for (USHORT i = 0; i < count; ++i) named((u64)frames[i]);
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
    rt::ShowFatal("The game stopped on an error it could not recover from (an access violation).");
    return EXCEPTION_CONTINUE_SEARCH;
}
void InstallHandler() {
    static std::once_flag once;
    std::call_once(once, [] {
        Init();
        if (!AddVectoredExceptionHandler(1, OnException)) Fatal("cannot install the page tracking handler");
    });
}
#else
struct sigaction previous{};
void OnFault(int signal, siginfo_t* info, void* context) {
    if (HandleFault(reinterpret_cast<u64>(info->si_addr), (u64)static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP])) return;
    // Not a tracked page: report it (a crash), then hand it on.
    {
        char line[160];
        const int n = std::snprintf(line, sizeof line, "[rt] SIGSEGV at %#llx (code %d), host pc %#llx; host backtrace:\n",
            (unsigned long long)reinterpret_cast<u64>(info->si_addr), info->si_code,
            (unsigned long long)static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]);
        (void)!write(2, line, n);
        {   // a call through a bad pointer: the caller's return address is at the stack top
            const auto& g = static_cast<ucontext_t*>(context)->uc_mcontext.gregs;
            char name[32] = "?"; pthread_getname_np(pthread_self(), name, sizeof name);
            const u64 sp = (u64)g[REG_RSP]; u64 ret = 0; std::memcpy(&ret, reinterpret_cast<void*>(sp), 8);
            const int m = std::snprintf(line, sizeof line, "[rt]   thread %s, return address at stack top %#llx\n", name, (unsigned long long)ret);
            (void)!write(2, line, m);
            void* r = reinterpret_cast<void*>(ret); backtrace_symbols_fd(&r, 1, 2);
        }
        void* frames[32]; const int count = backtrace(frames, 32); backtrace_symbols_fd(frames, count, 2);
    }
    if (previous.sa_flags & SA_SIGINFO) { if (previous.sa_sigaction) { previous.sa_sigaction(signal, info, context); return; } }
    else if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN) { previous.sa_handler(signal); return; }
    sigaction(SIGSEGV, &previous, nullptr); // fault again with the previous disposition
}
void InstallHandler() {
    static std::once_flag once;
    std::call_once(once, [] {
        Init();
        struct sigaction action{};
        action.sa_sigaction = OnFault;
        action.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        sigaction(SIGSEGV, &action, &previous);
    });
}
#endif

// Pages of [address, address + size), with every page queued for a full upload.
void Track(u64 address, u64 size) {
    InstallHandler();
    SettleRange(address, size);
    const u64 first = address & ~(PageSize - 1), end = (address + size + PageSize - 1) & ~(PageSize - 1);
    for (u64 page = first; page < end; page += PageSize) State(page, true);
    Guard guard;
    for (u64 page = first; page < end; page += PageSize) {
        u8& state = *State(page);
        if (state == Protected) MakeWritable(page);
        state = Dirty;
        Push(page, NoShadow);
    }
}
// Pages of [address, address + size) whose current contents the GPU copy will have.
void Protect(u64 address, u64 size) {
    InstallHandler();
    SettleRange(address, size);
    const u64 first = address & ~(PageSize - 1), end = (address + size + PageSize - 1) & ~(PageSize - 1);
    for (u64 page = first; page < end; page += PageSize) State(page, true);
    Guard guard;
    for (u64 page = first; page < end; page += PageSize) {
        u8& state = *State(page);
        if (state != Untracked) continue; // (a page shared with another copied pool)
        mprotect(reinterpret_cast<void*>(page), PageSize, PROT_READ);
        state = Protected;
    }
}
// Before page protection in [address, address + size) changes for other reasons: pending pages wait
// for their copy and are settled (so nothing makes stale contents accessible).
void SettleRange(u64 address, u64 size) {
    if (!size) return;
    const u64 first = address & ~(PageSize - 1), end = (address + size + PageSize - 1) & ~(PageSize - 1);
    for (;;) {
        u64 newest = 0;
        for (u64 page = first; page < end; page += PageSize)
            if (u64* back = BackSerial(page)) newest = std::max(newest, __atomic_load_n(back, __ATOMIC_ACQUIRE));
        if (newest <= 1) return;
        WaitWriteback(newest - 1);
        Guard guard;
        bool newer = false;
        for (u64 page = first; page < end; page += PageSize) {
            u64* back = BackSerial(page);
            if (!back || *back <= 1) continue;
            if (*back > newest) { newer = true; continue; }
            *back = 1;
            mprotect(reinterpret_cast<void*>(page), PageSize, SettledProtection(page));
        }
        if (!newer) return;
    }
}
// Drivers import only writable host pages. Pages write-protected for tracking become writable and
// count as written (a full upload; textures on them are uploaded again): nothing written while they
// are writable is missed. They are protected again at the next submission. Returns whether any were.
bool UnprotectForImport(u64 base, u64 size) {
    Init();
    SettleRange(base, size);
    bool any = false;
    Guard guard;
    for (u64 page = base & ~(PageSize - 1); page < base + size; page += PageSize) {
        if (u8* state = State(page); state && *state == Protected) { MarkDirty(page, *state, false); any = true; }
        else if (u8* armed = ArmState(page); armed && *armed == Armed) { MakeWritable(page); any = true; }
    }
    if (any) mprotect(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE);
    return any;
}
} // namespace track

// (not on GPUs sharing system memory, such as the Steam Deck's: their "device memory" is the same RAM, so a
// mirror only doubles the memory used. OOT_MIRROR_POOLS=1 mirrors there too; OOT_HOST_POOLS=1 never mirrors.)
bool MirrorEnabled() {
    static const bool enabled = [] {
        if (std::getenv("OOT_HOST_POOLS") || !PageProtectionUsable()) return false; // (mirrors find CPU writes by protection)
        if (std::getenv("OOT_MIRROR_POOLS")) return true;
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(Vk().physical, &p);
        const bool integrated = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if (integrated) RT_LOG("memory pools: shared system memory, used in place (no device copies)");
        return !integrated;
    }();
    return enabled;
}
std::atomic<u64> mirrored_bytes{0};

bool CopyEligible(const Pool& p) {
    static const bool uncached = std::getenv("OOT_MIRROR_UNCACHED") != nullptr;
    // (shader-code pools hold data too: OoT keeps a light-culling counter there that ~130k threads add to each frame;
    // read across the bus that was 2.7 ms of GPU time. OOT_NO_MIRROR_SHADER_POOLS=1: left in place)
    static const bool shader_pools = std::getenv("OOT_NO_MIRROR_SHADER_POOLS") == nullptr;
    // (physical pools behind virtual ones, though marked GPU-no-access, are what the GPU reads and writes through
    // those mappings: OoT's per-tile light lists, 1 MB written by ~130k threads, were 2 ms a frame across the bus.
    // OOT_NO_MIRROR_PHYSICAL=1: left in place)
    static const bool physical = std::getenv("OOT_NO_MIRROR_PHYSICAL") == nullptr;
    return MirrorEnabled() && !p.copy_failed && (uncached || (p.flags & PoolGpuCached) || (physical && (p.flags & PoolPhysical))) && p.buffer &&
           !(p.flags & ((uncached ? 0 : PoolCpuUncached) | (shader_pools ? 0 : PoolShaderCode) | (physical ? 0 : PoolGpuNoAccess) | PoolVirtual));
}
// Gives a pool its device-memory copy (on its first use as a GPU buffer: pools holding only
// texture data never get one). Earlier work of the open batch may have used the imported pages;
// a copy recorded here carries their contents over.
bool CreateDeviceCopy(Pool& p) {
    // (a sixth of the card's own memory (a twelfth under 6 GB), at most 3 GiB: the fixed 3 GiB let a 4 GB card give 1 GiB to these copies
    // while its textures no longer fitted. Pools past the budget are used where they are, as on shared-memory GPUs.
    // OOT_POOL_COPY_MB=<n>: the budget.)
    static const u64 Budget = []() -> u64 {
        if (const char* e = std::getenv("OOT_POOL_COPY_MB")) return std::strtoull(e, nullptr, 10) << 20;
        auto& vk = Vk(); u64 local = 0;
        for (u32 h = 0; h < vk.memory_properties.memoryHeapCount; ++h) if (vk.memory_properties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local = std::max<u64>(local, vk.memory_properties.memoryHeaps[h].size);
        local -= std::min(local, test_vram_hog_bytes); // (OOT_TEST_VRAM_HOG_MB: as the smaller card it stands for)
        // (under 6 GB a twelfth: on a 4 GB card the sixth, 680 MiB, left its render targets no room after an area change,
        // and those went to the shared memory - Hyrule Field at 59-80 fps, GPU 12-15 ms; with 340 MiB 91-113 fps, GPU 7-9 ms,
        // with the settings of a 4 GB laptop. 200 MiB was slower again: GPU 10-11 ms.)
        const u64 budget = std::min<u64>(3ull << 30, local < (6ull << 30) ? local / 12 : local / 6);
        RT_LOG("memory pools: up to %llu MiB of device copies (video memory %llu MiB)", (unsigned long long)(budget >> 20), (unsigned long long)(local >> 20));
        return budget; }();
    if (mirrored_bytes + p.import_size > Budget) { p.copy_failed = true; return false; }
    auto& vk = Vk();
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = p.import_size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk.device, buffer, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(vk.device, &ai, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(vk.device, buffer, nullptr);
        p.copy_failed = true;
        return false;
    }
    VK_CHECK(vkBindBufferMemory(vk.device, buffer, memory, 0));
    track::Protect(p.address, p.size); // later CPU writes are uploaded at submission
    const VkBufferCopy whole{p.buffer_offset, 0, p.import_size};
    Immediate([&](VkCommandBuffer cmd) { vkCmdCopyBuffer(cmd, p.buffer, buffer, 1, &whole); });
    {
        std::unique_lock lock{pools_mutex};
        p.device_memory = memory;
        p.device_buffer = buffer;
        table_dirty = true; table_changed = true; ++pool_generation;
    }
    mirrored_bytes += p.import_size;
    return true;
}
VkBuffer GpuBuffer(const Pool& p) { return p.device_buffer ? p.device_buffer : p.buffer; }

} // namespace
// Pages of [start, end) except `keep` (page addresses the CPU reads: copied back eagerly, left accessible).
void ProtectDeviceAhead(u64 start, u64 end, const std::set<u64>* keep) {
    static const bool off = std::getenv("OOT_AHEAD_NO_PROTECT") != nullptr;
    if (off || !MirrorEnabled()) return;
    track::InstallHandler();
    const u64 first = start & ~(track::PageSize - 1), last = (end + track::PageSize - 1) & ~(track::PageSize - 1);
    std::vector<std::pair<u64, u64>> runs;
    {
        std::lock_guard lock{track::ahead_mutex};
        // (page by page: those not kept and not already in a protected run; runs of them are added)
        const auto covered = [&](u64 page) {
            auto it = track::ahead_pages.upper_bound(page);
            return it != track::ahead_pages.begin() && std::prev(it)->second > page;
        };
        u64 run = 0;
        const auto close = [&](u64 to) { if (run && to > run) { track::ahead_pages[run] = to; runs.push_back({run, to}); } run = 0; };
        for (u64 page = first; page < last; page += track::PageSize) {
            if ((keep && keep->count(page)) || covered(page)) close(page);
            else if (!run) run = page;
        }
        close(last);
    }
    if (runs.empty()) return;
    track::Guard guard;
    for (auto [a, b] : runs) mprotect(reinterpret_cast<void*>(a), b - a, PROT_NONE);
}
// The pages of [start, end) become accessible again (the rest of a protected run stays protected).
void SettleDeviceAhead(u64 start, u64 end) {
    const u64 first = start & ~(track::PageSize - 1), last = (end + track::PageSize - 1) & ~(track::PageSize - 1);
    std::vector<std::pair<u64, u64>> settle;
    struct Settling { Settling() { track::settling.fetch_add(1, std::memory_order_acq_rel); } ~Settling() { track::settling.fetch_sub(1, std::memory_order_acq_rel); } } in_progress;
    {
        std::lock_guard lock{track::ahead_mutex};
        auto it = track::ahead_pages.upper_bound(first);
        if (it != track::ahead_pages.begin()) --it;
        while (it != track::ahead_pages.end() && it->first < last) {
            const u64 a = it->first, b = it->second;
            if (b <= first) { ++it; continue; }
            it = track::ahead_pages.erase(it);
            if (a < first) track::ahead_pages[a] = first;
            if (b > last) track::ahead_pages[last] = b;
            settle.push_back({std::max(a, first), std::min(b, last)});
        }
    }
    if (settle.empty()) return;
    track::Guard guard;
    for (auto [a, b] : settle) track::SettleRun(a, b);
}

std::atomic<u64> imports_live{0};
HostImport::~HostImport() {
    auto& vk = Vk();
    { std::lock_guard lock{vk.queue_mutex}; VK_CHECK(vkQueueWaitIdle(vk.queue)); } // (presentation may still be queued)
    if (vk.copy_queue != vk.queue) { std::lock_guard copy{vk.copy_mutex}; VK_CHECK(vkQueueWaitIdle(vk.copy_queue)); } // (write-back copies)
    if (placed) { // the contents back into ordinary memory at the same addresses (the driver leaves the range reserved)
        void* keep = std::malloc(size);
        if (!keep) Fatal("cannot keep a pool's contents");
        std::memcpy(keep, reinterpret_cast<void*>(base), size);
        static auto unmap2 = reinterpret_cast<PFN_vkUnmapMemory2KHR>(vkGetDeviceProcAddr(vk.device, "vkUnmapMemory2KHR"));
        VkMemoryUnmapInfoKHR ui{VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR}; ui.memory = memory; ui.flags = VK_MEMORY_UNMAP_RESERVE_BIT_EXT;
        unmap2(vk.device, &ui);
#ifdef __linux__
        if (mmap(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0) == MAP_FAILED)
            Fatal("cannot give a pool's memory back");
        madvise(reinterpret_cast<void*>(base), size, MADV_NOHUGEPAGE);
#else
        Fatal("placed pool maps are Linux-only here");
#endif
        std::memcpy(reinterpret_cast<void*>(base), keep, size);
        std::free(keep);
        backed = false;
    }
#ifdef __linux__
    if (backed) { // the contents back into ordinary memory at the same addresses
        track::UnprotectForImport(base, size); // (the plain memory that takes its place is writable: nothing stays armed)
        void* keep = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (keep == MAP_FAILED) Fatal("cannot keep a pool's contents");
        std::memcpy(keep, reinterpret_cast<void*>(base), size);
        if (mmap(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0) == MAP_FAILED)
            Fatal("cannot give a pool's memory back");
        madvise(reinterpret_cast<void*>(base), size, MADV_NOHUGEPAGE);
        std::memcpy(reinterpret_cast<void*>(base), keep, size);
        munmap(keep, size);
    }
#endif
    if (buffer) vkDestroyBuffer(vk.device, buffer, nullptr);
    if (memory) vkFreeMemory(vk.device, memory, nullptr);
    --imports_live;
}

Pool::~Pool() {
    if (flags & PoolVirtual) {
        munmap(reinterpret_cast<void*>(address), (size + 0xffff) & ~0xffffULL);
        return;
    }
    if (host_import) buffer = VK_NULL_HANDLE; // (the shared import is released with its last pool)
    if (!buffer && !memory && !device_buffer) return;
    auto& vk = Vk();
    if (device_buffer) {
        // The memory returns to the game: leave its pages writable (pages shared with another
        // pool get a full upload to it).
        const u64 first = address & ~(track::PageSize - 1), end = (address + size + track::PageSize - 1) & ~(track::PageSize - 1);
        track::SettleRange(address, size);
        track::Guard guard;
        for (u64 page = first; page < end; page += track::PageSize) {
            u8& state = *track::State(page);
            if (state == track::Protected) track::MarkDirty(page, state, false);
        }
    }
    // Presentation may still be queued after releasing this lock.
    std::lock_guard lock{vk.queue_mutex};
    VK_CHECK(vkQueueWaitIdle(vk.queue));
    if (vk.copy_queue != vk.queue) { std::lock_guard copy{vk.copy_mutex}; VK_CHECK(vkQueueWaitIdle(vk.copy_queue)); } // (write-back copies)
    if (buffer) vkDestroyBuffer(vk.device, buffer, nullptr);
    if (memory) vkFreeMemory(vk.device, memory, nullptr);
    if (device_buffer) { vkDestroyBuffer(vk.device, device_buffer, nullptr); mirrored_bytes -= import_size; }
    if (device_memory) vkFreeMemory(vk.device, device_memory, nullptr);
}

// Guest-address translation table for shaders' untracked global memory accesses (see the shader
// compiler's global_address_table): u64 count, then {start, end, device address} per imported pool.
bool GlobalAddressTableCurrent() { std::shared_lock lock{pools_mutex}; return !table_dirty; }
#ifdef OOT_SHADER_LONGSHOT
// With Longshot's translator the table is a page table instead: one u64 per 4 KiB guest page, the page's device
// address or 0, for the two windows of guest GPU addresses its programs look up (spirv_emit_context.cpp
// kGuestPagesBase...). Its address is pushed to the programs (GuestPageTableAddress), not written into them.
constexpr u64 PagesBaseA = 0x1000000000ull, PagesBytesA = 40ull << 30, PagesBaseB = 0x7f00000000ull, PagesBytesB = 4ull << 30;
constexpr u64 PageTableEntries = (PagesBytesA + PagesBytesB) >> 12;
std::atomic<u64> page_table_address{0};
#endif
u64 GuestPageTableAddress() {
#ifdef OOT_SHADER_LONGSHOT
    return page_table_address.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}
u64 GlobalAddressTable(bool update) {
    std::lock_guard batch_lock{BatchMutex()};
#ifdef OOT_SHADER_LONGSHOT
    constexpr u64 Capacity = (PageTableEntries * 8) / 24 + 1; // (the buffer's size below: 8 + Capacity * 24 bytes)
    struct Filled { u64 index, count, device; bool operator==(const Filled&) const = default; };
    static std::vector<Filled> filled; // what the table holds, sorted by index
#else
    constexpr u64 Capacity = 16384;
#endif
    static VkBuffer buffer = VK_NULL_HANDLE;
    static u64* mapped = nullptr;
    static u64 device = 0;
    auto& vk = Vk();
    if (!buffer) {
        // Programs reading global memory embed this table's GPU address: at another address every run, their
        // pipelines missed the driver's cache every run (270-660 ms each, mid-game). The address of the first run
        // is kept (capture/replay) and asked for again; if the driver cannot give it, a new one is kept.
        const std::string replay_path = rt::DataDir() + "/address_table.bin";
        struct Replay { u64 buffer = 0, memory = 0; } replay{};
        if (vk.address_replay) if (FILE* f = std::fopen(replay_path.c_str(), "rb")) { if (std::fread(&replay, sizeof replay, 1, f) != 1) replay = {}; std::fclose(f); }
        VkDeviceMemory memory = VK_NULL_HANDLE;
        const auto make = [&](const Replay& want) -> bool {
            VkBufferOpaqueCaptureAddressCreateInfo bo{VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO};
            bo.opaqueCaptureAddress = want.buffer;
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.pNext = vk.address_replay ? &bo : nullptr;
            bi.flags = vk.address_replay ? VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT : 0;
            bi.size = 8 + Capacity * 24;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            if (VkResult r = vkCreateBuffer(vk.device, &bi, nullptr, &buffer); r != VK_SUCCESS) {
                RT_LOG("guest address table: buffer at %#llx refused (%d)", (unsigned long long)want.buffer, (int)r); buffer = VK_NULL_HANDLE; return false; }
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(vk.device, buffer, &req);
            VkMemoryOpaqueCaptureAddressAllocateInfo mo{VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO};
            mo.opaqueCaptureAddress = want.memory;
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.pNext = vk.address_replay ? &mo : nullptr;
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | (vk.address_replay ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT : 0);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &flags;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkResult r = vkAllocateMemory(vk.device, &ai, nullptr, &memory);
            if (r == VK_SUCCESS) r = vkBindBufferMemory(vk.device, buffer, memory, 0);
            if (r != VK_SUCCESS) {
                RT_LOG("guest address table: memory at %#llx refused (%d)", (unsigned long long)want.memory, (int)r);
                if (memory) vkFreeMemory(vk.device, memory, nullptr);
                vkDestroyBuffer(vk.device, buffer, nullptr); buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE;
                return false;
            }
            return true;
        };
        const bool replayed = (replay.buffer || replay.memory) && make(replay); // (NVIDIA: the memory carries it, the buffer reports 0)
        if (!replayed && !make({})) Fatal("cannot make the guest address table");
        if (vk.address_replay && !replayed) { // (kept for the next run)
            VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bi.buffer = buffer;
            VkDeviceMemoryOpaqueCaptureAddressInfo mi{VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO}; mi.memory = memory;
            Replay now{vkGetBufferOpaqueCaptureAddress(vk.device, &bi), vkGetDeviceMemoryOpaqueCaptureAddress(vk.device, &mi)};
            if (FILE* f = std::fopen(replay_path.c_str(), "wb")) { std::fwrite(&now, sizeof now, 1, f); std::fclose(f); }
        }
        RT_LOG("guest address table: %s GPU address", replayed ? "the same" : vk.address_replay ? "a new (kept)" : "this run's");
        { VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; ai.buffer = buffer;
          RT_LOG("guest address table at GPU address %#llx", (unsigned long long)vkGetBufferDeviceAddress(vk.device, &ai)); }
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = 8 + Capacity * 24; // (the mapping below)
        VK_CHECK(vkMapMemory(vk.device, memory, 0, bi.size, 0, reinterpret_cast<void**>(&mapped)));
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer;
        device = vkGetBufferDeviceAddress(vk.device, &info);
#ifdef OOT_SHADER_LONGSHOT
        std::memset(mapped, 0, PageTableEntries * 8);
        page_table_address.store(device, std::memory_order_relaxed);
#else
        mapped[0] = 0;
#endif
    }
    if (!update) return device; // (the address only: shader compilation)
    std::unique_lock lock{pools_mutex};
    if (table_dirty) {
        // Previously recorded shaders still reference this table until their
        // batch completes. Do not replace their mappings while they are pending.
        static const bool always = std::getenv("OOT_TABLE_FLUSH_ALWAYS") != nullptr;
#ifdef OOT_SHADER_LONGSHOT
        const bool flush = always || table_changed;
#else
        const bool flush = true; // (rewritten whole there)
#endif
        if (flush) { lock.unlock(); Flush(); lock.lock(); }
        table_changed = false;
        { static const bool stats = std::getenv("OOT_TABLE_STATS") != nullptr; static u64 rebuilds = 0; // (diagnostic: each one flushes)
          if (stats && ++rebuilds % 64 == 0) RT_LOG("guest address table: %llu rebuilds", (unsigned long long)rebuilds); }
        u64 count = 0;
        const auto device_of = [&](const Pool& physical) {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = GpuBuffer(physical);
            return vkGetBufferDeviceAddress(vk.device, &info) + (physical.device_buffer ? 0 : physical.buffer_offset) + (physical.address - physical.import_base);
        };
#ifdef OOT_SHADER_LONGSHOT
        // The table is changed in place and only where it differs: work already submitted is still reading it (the
        // flush above submits, it does not wait). It was cleared and written again whole, megabytes of entries at
        // every pool change, and what the GPU ran meanwhile read zeros through it: for that frame the culling, the fog
        // and the occlusion counts were wrong (no haze, the white light in doorways).
        std::vector<Filled> now;
        u64 outside = 0;
        const auto add = [&](u64 start, u64 size, u64 device) {
            u64 first = (start + 0xfff) & ~0xfffull; const u64 last = (start + size) & ~0xfffull; // (whole pages)
            if (first >= last) return;
            u64 index;
            if (first >= PagesBaseA && last <= PagesBaseA + PagesBytesA) index = (first - PagesBaseA) >> 12;
            else if (first >= PagesBaseB && last <= PagesBaseB + PagesBytesB) index = (PagesBytesA >> 12) + ((first - PagesBaseB) >> 12);
            else { ++outside; return; }
            const u64 n = (last - first) >> 12; device += first - start;
            now.push_back({index, n, device}); ++count;
        };
#else
        const auto add = [&](u64 start, u64 size, u64 device) {
            // Entries stay sorted and non-overlapping (shaders binary-search them); contiguous
            // ranges with contiguous device memory merge.
            if (count) {
                u64* last = mapped + 1 + (count - 1) * 3;
                if (last[1] == start && last[2] + (last[1] - last[0]) == device) { last[1] = start + size; return; }
            }
            if (count == Capacity) return;
            u64* entry = mapped + 1 + count * 3;
            entry[0] = start; entry[1] = start + size; entry[2] = device;
            ++count;
        };
#endif
        for (const auto& [start, pool] : pools_by_address) {
            if (pool->flags & PoolVirtual) {
                // Each mapped range of a virtual pool points into its physical pool.
                for (const auto& [offset, mapping] : pool->mappings)
                    if (mapping.physical && mapping.physical->buffer)
                        add(pool->address + offset, mapping.size, device_of(*mapping.physical) + mapping.physical_offset);
                continue;
            }
            if (pool->buffer) add(pool->address, pool->size, device_of(*pool));
        }
#ifdef OOT_SHADER_LONGSHOT
        std::sort(now.begin(), now.end(), [](const Filled& a, const Filled& b) { return a.index < b.index; });
        {   // entries no range holds any more are cleared, ranges that are new or moved are written
            u64 cleared = 0, written = 0; size_t k = 0;
            for (const Filled& old : filled) {
                if (std::binary_search(now.begin(), now.end(), old, [](const Filled& a, const Filled& b) { return a.index < b.index; }) &&
                    *std::lower_bound(now.begin(), now.end(), old, [](const Filled& a, const Filled& b) { return a.index < b.index; }) == old) continue;
                u64 at = old.index; const u64 end = old.index + old.count;
                while (k < now.size() && now[k].index + now[k].count <= at) ++k;
                for (size_t j = k; at < end; ++j) {
                    const u64 stop = j < now.size() ? std::min(end, std::max(at, now[j].index)) : end;
                    if (stop > at) { std::memset(mapped + at, 0, (stop - at) * 8); cleared += stop - at; }
                    if (j >= now.size()) break;
                    at = std::max(at, now[j].index + now[j].count);
                }
            }
            for (const Filled& range : now) {
                const auto was = std::lower_bound(filled.begin(), filled.end(), range, [](const Filled& a, const Filled& b) { return a.index < b.index; });
                if (was != filled.end() && *was == range) continue;
                for (u64 i = 0; i < range.count; ++i) mapped[range.index + i] = range.device + (i << 12);
                written += range.count;
            }
            static const bool stats = std::getenv("OOT_TABLE_STATS") != nullptr;
            if (stats && (cleared || written)) RT_LOG("guest page table: %llu entries cleared, %llu written (%zu ranges)", (unsigned long long)cleared, (unsigned long long)written, now.size());
            filled = std::move(now);
        }
        { static u64 reported = 0; if (outside && outside != reported) { reported = outside;
            RT_LOG("guest page table: %llu pool ranges lie outside its windows (their pointers read 0)", (unsigned long long)outside); } }
#else
        mapped[0] = count;
#endif
        table_dirty = false;
    }
    return device;
}

// The contiguous run of guest GPU memory around an address: its pool, or the mapped range of a virtual pool.
bool GuestRegionAround(u64 address, u64* start, u64* end) {
    std::shared_lock lock{pools_mutex};
    auto it = pools_by_address.upper_bound(address);
    if (it == pools_by_address.begin()) return false;
    --it;
    const Pool* p = it->second.get();
    if (address - p->address >= p->size) return false;
    if (!(p->flags & PoolVirtual)) { *start = p->address; *end = p->address + p->size; return true; }
    const u64 offset = address - p->address;
    auto mapping = p->mappings.upper_bound(offset);
    if (mapping == p->mappings.begin()) return false;
    --mapping;
    if (offset - mapping->first >= mapping->second.size || !mapping->second.physical) return false;
    *start = p->address + mapping->first; *end = *start + mapping->second.size;
    return true;
}
BufferRange LookupAddressUncached(u64 address, bool for_gpu = true);
BufferRange LookupAddress(u64 address) {
    // Contiguous ranges resolved by this thread, direct-mapped by 4 KiB page; valid until pools or
    // mappings change.
    struct Cached { u64 generation = 0, lo = 0, hi = 0; BufferRange at_lo; };
    static thread_local Cached cached[1024];
    const u64 generation = pool_generation.load(std::memory_order_acquire);
    Cached& c = cached[(address >> 12) & 1023];
    if (c.generation == generation && address >= c.lo && address < c.hi) {
        const u64 delta = address - c.lo;
        return {c.at_lo.buffer, c.at_lo.offset + delta, c.at_lo.pool, c.at_lo.cpu_address + delta, c.at_lo.available - delta};
    }
    const BufferRange found = LookupAddressUncached(address, true);
    if (found.pool) c = {generation, address, address + found.available, found};
    return found;
}

u64 PoolGeneration() { return pool_generation.load(std::memory_order_acquire); }
BufferRange GpuLookupAddress(u64 address) {
    const BufferRange range = LookupAddress(address);
    if (range.pool && !range.pool->device_buffer && CopyEligible(*range.pool)) {
        if (nvn::resolve_worker) { RESOLVE_BAIL(); return range; } // (the device copy is the executor's to make)
        if (CreateDeviceCopy(*range.pool)) return LookupAddress(address);
    }
    return range;
}

BufferRange LookupAddressUncached(u64 address, bool for_gpu) {
    std::shared_lock lock{pools_mutex};
    auto it = pools_by_address.upper_bound(address);
    if (it == pools_by_address.begin()) return {};
    --it;
    Pool* p = it->second.get();
    if (address - p->address >= p->size) return {};
    u64 offset = address - p->address, available = p->size - offset;
    if (p->flags & PoolVirtual) {
        auto mapping = p->mappings.upper_bound(offset);
        if (mapping == p->mappings.begin()) return {};
        --mapping;
        const u64 delta = offset - mapping->first;
        if (delta >= mapping->second.size) return {};
        available = std::min(available, mapping->second.size - delta);
        offset = mapping->second.physical_offset + delta;
        p = mapping->second.physical.get();
    }
    if (for_gpu && !p->buffer && !(p->flags & PoolVirtual)) { // (a small pool's first GPU use: import it now)
        Pool* pool = p;
        lock.unlock();
        {
            std::unique_lock exclusive{pools_mutex};
            if (!pool->buffer) { ImportPool(*pool); table_dirty = true; ++pool_generation; } // (added: see table_changed)
        }
        return LookupAddressUncached(address, true);
    }
    return {GpuBuffer(*p), (p->device_buffer ? 0 : p->buffer_offset) + p->address - p->import_base + offset, p, p->address + offset, available};
}

// For reading or writing the memory on the CPU only (texture texels): the pool is found but not brought to the GPU. A
// pool the GPU never binds stays plain memory - on a GPU that keeps pools in its own memory (the Steam Deck) every
// pool looked up was moved there, the game's gigabytes of texture data with them, and the first area ran the GPU out of
// memory ("Not enough memory for command submission", 2026-11-16). The range's buffer may be null.
BufferRange LookupAddressCpu(u64 address) { return LookupAddressUncached(address, false); }

void* ResolveAddress(u64 address, u64 size) {
    const auto range = LookupAddress(address);
    if (!range.pool || size > range.available)
        Fatal("GPU address %#llx+%#llx is not backed by contiguous pool storage", (unsigned long long)address,
              (unsigned long long)size);
    return reinterpret_cast<void*>(range.cpu_address);
}

Pool* PoolOf(u64 object) { return HostPool(object); }

struct MappingRequest {
    u64 physical_pool, physical_offset, virtual_offset, size;
    u32 storage_class, padding;
};
static_assert(sizeof(MappingRequest) == 0x28);

bool MapVirtual(Pool& pool, s32 count, const MappingRequest* requests) {
    if (!(pool.flags & PoolVirtual) || count < 0 || (count && !requests)) return false;
    // Releasing the last physical reference can take the queue lock. Keep it
    // alive until pools_mutex is released: rendering takes these locks in reverse.
    std::vector<std::shared_ptr<Pool>> retired;
    std::unique_lock lock{pools_mutex};
    // Validate the entire batch before replacing any mappings.
    for (s32 i = 0; i < count; ++i) {
        const auto& r = requests[i];
        if (!r.size || ((r.physical_offset | r.virtual_offset | r.size) & 0xffff) ||
            r.virtual_offset > pool.size || r.size > pool.size - r.virtual_offset) return false;
        if (r.physical_pool) {
            Pool* source = HostPool(r.physical_pool);
            if (!source || (source->flags & PoolVirtual) || r.physical_offset > source->size ||
                r.size > source->size - r.physical_offset || !pools_by_address.contains(source->address)) return false;
        }
    }
    for (s32 i = 0; i < count; ++i) {
        const auto& r = requests[i];
        const u64 end = r.virtual_offset + r.size;
        for (auto it = pool.mappings.begin(); it != pool.mappings.end();) {
            const u64 base = it->first, old_end = base + it->second.size;
            if (old_end <= r.virtual_offset || base >= end) { ++it; continue; }
            const VirtualMapping old = it->second;
            retired.push_back(old.physical);
            it = pool.mappings.erase(it);
            if (base < r.virtual_offset)
                pool.mappings.emplace(base, VirtualMapping{r.virtual_offset - base, old.physical_offset, old.physical});
            if (old_end > end) {
                pool.mappings.emplace(end, VirtualMapping{old_end - end, old.physical_offset + end - base, old.physical});
                break;
            }
        }
        if (r.physical_pool) {
            Pool* source = HostPool(r.physical_pool);
            pool.mappings[r.virtual_offset] = {r.size, r.physical_offset, pools_by_address.at(source->address)};
        }
    }
    table_dirty = true; table_changed = true; ++pool_generation;
    return true;
}

// ---- CPU writes captured when the game submits ------------------------------------------------------------------
// The device copies of the pools were brought up to date by GPU copies out of the game's live memory, recorded at
// each batch submission and run whenever the GPU got there. The executor translates a frame milliseconds after the
// game submitted it, and by then the game is writing the next frame into the same memory (per-object matrices,
// instance tables, the frame's constants): a depth pre-pass and the pass shading with depth EQUAL, in different
// batches, read different frames' data. Mesh-drawn objects (doors, vines, ornaments) lost their shading pass to it,
// leaving the pre-pass's hole in the picture, which bloom blew up to white. As NX2navi's uniform snapshot and
// Longshot's captures: what the CPU wrote is copied when the game submits commands (CapturePoolWrites, on the game's
// thread), and reaches the device copies in the batch that begins that submission's commands (ApplyPoolCapture).
// OOT_LIVE_UPLOADS=1: from live memory at each batch, as before.
namespace {
struct CaptureCopy { std::shared_ptr<Pool> pool; VkBufferCopy region; };
struct Staging { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; u8* mapped = nullptr; u64 size = 0; };
struct PoolCapture { Staging staging; std::vector<CaptureCopy> copies; };
std::mutex capture_mutex;
std::vector<Staging> spare_staging;                 // (capture_mutex)
std::deque<std::unique_ptr<PoolCapture>> applied;   // (capture_mutex) for the next batch's uploads
bool LiveUploads() { static const bool on = std::getenv("OOT_LIVE_UPLOADS") != nullptr; return on; }
Staging TakeStaging(u64 bytes) {
    const u64 size = std::max<u64>(std::bit_ceil(bytes), 64u << 10);
    { std::lock_guard lock{capture_mutex};
      for (size_t i = 0; i < spare_staging.size(); ++i) if (spare_staging[i].size == size) { Staging s = spare_staging[i]; spare_staging.erase(spare_staging.begin() + i); return s; } }
    auto& vk = Vk(); Staging s; s.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &s.buffer));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk.device, s.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &s.memory));
    VK_CHECK(vkBindBufferMemory(vk.device, s.buffer, s.memory, 0));
    VK_CHECK(vkMapMemory(vk.device, s.memory, 0, size, 0, reinterpret_cast<void**>(&s.mapped)));
    return s;
}
}
void* CapturePoolWrites() {
    if (LiveUploads() || !track::dirty) return nullptr;
    std::shared_lock lock{pools_mutex};
    std::vector<track::Entry> entries;
    {
        track::Guard guard;
        if (!track::dirty_count) return nullptr;
        entries.assign(track::dirty, track::dirty + track::dirty_count);
        track::dirty_count = 0;
        if (++track::hot_generation % 600 == 0) track::heat.clear(); // (how often each page was written: in windows of 600 captures)
        // Protect again first: later writes fault and are queued with a fresh shadow.
        for (const auto& e : entries) {
            u8& state = *track::State(e.page);
            if (state != track::Dirty) continue;
            if (u64* back = track::BackSerial(e.page); back && *back > 1) { state = track::Protected; continue; } // (stays inaccessible until its copy)
            if (track::HotPagesOn()) {
                // A page the game writes all the time (a movie's pictures: 800 pages a frame) stays writable: its contents
                // are compared with a fresh shadow at each capture instead. On Windows each write fault cost ~1 ms there,
                // and a movie after a long session ran at 1 fps (issue #21). One unchanged for 1200 captures is protected again.
                auto hot = track::hot.find(e.page);
                if (hot == track::hot.end() && ++track::heat[e.page] >= 8 && track::hot.size() < 8192) hot = track::hot.emplace(e.page, 0u).first;
                if (hot != track::hot.end() && hot->second < 1200) {
                    u32 slot = track::NoShadow;
                    if (track::free_count) { slot = track::free_slots[--track::free_count]; std::memcpy(track::shadows + (u64)slot * track::PageSize, reinterpret_cast<void*>(e.page), track::PageSize); }
                    track::Push(e.page, slot); // (still Dirty and writable: captured again next time)
                    continue;
                }
                if (hot != track::hot.end()) track::hot.erase(hot);
            }
            state = track::Protected;
            mprotect(reinterpret_cast<void*>(e.page), track::PageSize, PROT_READ);
        }
    }
    struct Range { std::shared_ptr<Pool> pool; u64 from, to; };
    std::vector<Range> ranges; std::vector<u32> freed; u64 bytes = 0;
    for (const auto& e : entries) {
        bool covered = false; const u64 bytes_before = bytes;
        auto it = pools_by_address.upper_bound(e.page + track::PageSize - 1);
        while (it != pools_by_address.begin()) {
            --it;
            const std::shared_ptr<Pool>& p = it->second;
            if (p->address + p->size <= e.page) break;
            if (!p->device_buffer) continue;
            covered = true;
            const u64 lo = std::max(e.page, p->address), hi = std::min(e.page + track::PageSize, p->address + p->size);
            const auto add = [&](u64 a, u64 b) {
                if (!ranges.empty() && ranges.back().pool == p && ranges.back().to == a) ranges.back().to = b; else ranges.push_back({p, a, b});
                bytes += b - a; };
            if (e.shadow == track::NoShadow) { add(lo, hi); continue; }
            // Only the 64-byte granules that changed.
            const u8* now = reinterpret_cast<const u8*>(e.page);
            const u8* before = track::shadows + (u64)e.shadow * track::PageSize;
            u64 run = 0; bool open = false;
            for (u64 g = lo & ~63ull; g < hi; g += 64) {
                const bool changed = std::memcmp(now + (g - e.page), before + (g - e.page), 64) != 0;
                if (changed && !open) { run = std::max(g, lo); open = true; }
                else if (!changed && open) { add(run, g); open = false; }
            }
            if (open) add(run, hi);
            if (track::HotPagesOn()) { track::Guard guard; if (auto hot = track::hot.find(e.page); hot != track::hot.end()) hot->second = bytes == bytes_before ? hot->second + 1 : 0; }
        }
        if (!covered) { // the pool went away: leave the page to the game
            track::Guard guard;
            u8& state = *track::State(e.page);
            if (state == track::Protected) { track::MakeWritable(e.page); state = track::Untracked; }
            if (state == track::Dirty && track::hot.erase(e.page)) state = track::Untracked; // (a hot page: dropped from the list next time)
        }
        if (e.shadow != track::NoShadow) freed.push_back(e.shadow);
    }
    {
        track::Guard guard;
        for (u32 slot : freed) track::free_slots[track::free_count++] = slot;
        if (track::overflow) RT_LOG("page tracking overflow: some CPU writes may not reach the GPU");
        track::overflow = false;
    }
    if (ranges.empty()) return nullptr;
    auto capture = std::make_unique<PoolCapture>();
    capture->staging = TakeStaging(bytes);
    u64 at = 0;
    for (const auto& r : ranges) {
        std::memcpy(capture->staging.mapped + at, reinterpret_cast<const void*>(r.from), r.to - r.from);
        capture->copies.push_back({r.pool, {at, r.from - r.pool->import_base, r.to - r.from}});
        at += r.to - r.from;
    }
    track::uploaded += bytes; track::upload_regions += ranges.size();
    return capture.release();
}
void ApplyPoolCapture(void* capture) {
    if (!capture) return;
    std::lock_guard lock{capture_mutex};
    applied.emplace_back(static_cast<PoolCapture*>(capture));
}
bool PoolCapturesWaiting() { std::lock_guard lock{capture_mutex}; return !applied.empty(); }
std::shared_lock<std::shared_mutex> RecordPoolTransfers(VkCommandBuffer uploads, bool* any_uploads, VkCommandBuffer cmd,
                                                         const std::map<u64, u64>& gpu_writes, const std::map<u64, u64>& gpu_reports,
                                                         VkCommandBuffer writeback, std::vector<std::pair<u64, u64>>* pending) {
    std::shared_lock lock{pools_mutex};
    *any_uploads = false;
    if (!LiveUploads()) { // the captures applied since the last batch, in their order
        std::deque<std::unique_ptr<PoolCapture>> captures;
        { std::lock_guard captured{capture_mutex}; captures.swap(applied); }
        bool first = true;
        for (auto& capture : captures) {
            std::map<Pool*, std::vector<VkBufferCopy>> copies;
            for (const auto& c : capture->copies) if (c.pool->device_buffer) copies[c.pool.get()].push_back(c.region);
            if (!copies.empty() && first) { // after all earlier GPU work (it may still use the old contents)
                first = false;
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(uploads, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            } else if (!copies.empty()) { // (a later capture may write the same bytes again: in order)
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(uploads, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            }
            for (auto& [p, list] : copies) { vkCmdCopyBuffer(uploads, capture->staging.buffer, p->device_buffer, (u32)list.size(), list.data()); *any_uploads = true; }
            // (the staging memory and the pools it names stay until the batch is done)
            nvn::AfterFlush([held = std::shared_ptr<PoolCapture>(std::move(capture))] { std::lock_guard captured{capture_mutex}; spare_staging.push_back(held->staging); });
        }
    }
    if ((LiveUploads() ? !track::dirty : true) && gpu_writes.empty() && gpu_reports.empty()) return lock;
    // Mirrored pools overlapping [start, end), in address order.
    const auto each_pool = [&](u64 start, u64 end, auto&& visit) {
        auto it = pools_by_address.upper_bound(end - 1);
        std::vector<Pool*> found;
        while (it != pools_by_address.begin()) {
            --it;
            Pool* p = it->second.get();
            if (p->address + p->size <= start) break;
            if (p->device_buffer) found.push_back(p);
        }
        for (auto i = found.rbegin(); i != found.rend(); ++i) visit(**i);
    };
    // CPU writes since the last submission.
    if (track::dirty && LiveUploads()) {
        std::vector<track::Entry> entries;
        {
            track::Guard guard;
            entries.assign(track::dirty, track::dirty + track::dirty_count);
            track::dirty_count = 0;
            // Protect again first: later writes fault and are queued with a fresh shadow.
            for (const auto& e : entries) {
                u8& state = *track::State(e.page);
                if (state != track::Dirty) continue;
                state = track::Protected;
                if (u64* back = track::BackSerial(e.page); back && *back > 1) continue; // (stays inaccessible until its copy)
                mprotect(reinterpret_cast<void*>(e.page), track::PageSize, PROT_READ);
            }
        }
        std::map<Pool*, std::vector<VkBufferCopy>> copies;
        std::vector<u32> freed;
        u64 bytes = 0;
        for (const auto& e : entries) {
            bool covered = false;
            each_pool(e.page, e.page + track::PageSize, [&](Pool& p) {
                covered = true;
                const u64 lo = std::max(e.page, p.address), hi = std::min(e.page + track::PageSize, p.address + p.size);
                const auto copy = [&](u64 a, u64 b) {
                    auto& list = copies[&p];
                    const u64 offset = a - p.import_base;
                    if (!list.empty() && list.back().dstOffset + list.back().size == offset) list.back().size += b - a;
                    else list.push_back({p.buffer_offset + offset, offset, b - a});
                    bytes += b - a;
                };
                if (e.shadow == track::NoShadow) { copy(lo, hi); return; }
                // Only the 64-byte granules that changed.
                const u8* now = reinterpret_cast<const u8*>(e.page);
                const u8* before = track::shadows + (u64)e.shadow * track::PageSize;
                u64 run = 0; bool open = false;
                for (u64 g = lo & ~63ull; g < hi; g += 64) {
                    const bool changed = std::memcmp(now + (g - e.page), before + (g - e.page), 64) != 0;
                    if (changed && !open) { run = std::max(g, lo); open = true; }
                    else if (!changed && open) { copy(run, g); open = false; }
                }
                if (open) copy(run, hi);
            });
            if (!covered) { // the pool went away: leave the page to the game
                track::Guard guard;
                u8& state = *track::State(e.page);
                if (state == track::Protected) { track::MakeWritable(e.page); state = track::Untracked; }
            }
            if (e.shadow != track::NoShadow) freed.push_back(e.shadow);
        }
        {
            track::Guard guard;
            for (u32 slot : freed) track::free_slots[track::free_count++] = slot;
            if (track::overflow) RT_LOG("page tracking overflow: some CPU writes may not reach the GPU");
            track::overflow = false;
        }
        if (!copies.empty()) { // after all earlier GPU work (it may still use the old contents)
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(uploads, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        for (auto& [p, list] : copies) {
            vkCmdCopyBuffer(uploads, p->buffer, p->device_buffer, (u32)list.size(), list.data());
            track::upload_regions += list.size();
        }
        track::uploaded += bytes;
        *any_uploads = !copies.empty();
    }
    // GPU writes of the batch, back to guest memory: storage writes by `writeback` (its own submission,
    // after the batch; their pages are made pending), reports (small) at the batch end.
    static const bool eager = std::getenv("OOT_LAZY_WRITEBACK") == nullptr || !writeback; // (lazy: no measured gain yet)
    u64 bytes = 0;
    const auto add = [&](const std::map<u64, u64>& ranges, std::map<Pool*, std::vector<VkBufferCopy>>& copies, bool lazy) {
        // OOT_WRITEBACK_MAX=<bytes> (experiment): larger GPU-written ranges are not copied back.
        static const u64 max_range = std::getenv("OOT_WRITEBACK_MAX") ? std::strtoull(std::getenv("OOT_WRITEBACK_MAX"), nullptr, 0) : ~0ull;
        for (const auto& [start, end] : ranges)
            if (end - start <= max_range)
            each_pool(start, end, [&](Pool& p) {
                const u64 lo = std::max(start, p.address), hi = std::min(end, p.address + p.size);
                if (lo >= hi) return;
                copies[&p].push_back({lo - p.import_base, p.buffer_offset + lo - p.import_base, hi - lo});
                bytes += hi - lo;
                if (lazy) {
                    const u64 a = lo & ~(track::PageSize - 1), b = (hi + track::PageSize - 1) & ~(track::PageSize - 1);
                    if (!pending->empty() && pending->back().second >= a) pending->back().second = std::max(pending->back().second, b);
                    else pending->push_back({a, b});
                }
                static const bool trace = std::getenv("OOT_WRITEBACK_TRACE") != nullptr;
                if (trace) { static std::map<std::pair<u64, u64>, u64> sizes; static u64 n = 0; sizes[{lo, hi - lo}]++;
                    if (++n % 2000 == 0) { for (auto& [k, c] : sizes) if (k.second * c > (4u << 20)) RT_LOG("write-back %#llx+%#llx: %llu", (unsigned long long)k.first, (unsigned long long)k.second, (unsigned long long)c); sizes.clear(); } }
            });
    };
    const auto record = [&](VkCommandBuffer into, std::map<Pool*, std::vector<VkBufferCopy>>& copies) {
        if (copies.empty()) return;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(into, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        for (auto& [p, list] : copies) vkCmdCopyBuffer(into, p->device_buffer, p->buffer, (u32)list.size(), list.data());
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(into, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    };
    std::map<Pool*, std::vector<VkBufferCopy>> at_end, after;
    if (eager) { add(gpu_writes, at_end, false); add(gpu_reports, at_end, false); }
    else { add(gpu_writes, after, true); add(gpu_reports, at_end, false); }
    record(cmd, at_end);
    if (!after.empty()) record(writeback, after);
    track::written_back += bytes;
    return lock;
}

void WatchGpuWrites(const std::map<u64, u64>& ranges, std::vector<u64>* pages) {
    static const bool enabled = std::getenv("OOT_READ_WATCH") != nullptr;
    if (!enabled) return;
    track::InstallHandler();
    for (const auto& [start, end] : ranges)
        if (end - start != 16) // (reports: known)
        for (u64 page = start & ~(track::PageSize - 1); page < end; page += track::PageSize) {
            if (!LookupAddress(page).pool && !LookupAddress(std::max(page, start)).pool) continue;
            track::State(page, true); u8* w = track::Watch(page, true);
            track::Guard guard;
            if ((*w)++ == 0) mprotect(reinterpret_cast<void*>(page), track::PageSize, PROT_NONE);
            pages->push_back(page);
        }
}
void UnwatchPages(const std::vector<u64>& pages) {
    if (pages.empty()) return;
    track::Guard guard;
    for (u64 page : pages) { u8* w = track::Watch(page, false); if (w && *w && --*w == 0) track::RestoreProtection(page); }
}

// Pages the game reads late (batch.cpp, WaitForWrites' OOT_READBACK_LATE rules): they stay readable while a copy of GPU
// results is on its way to them, so a read sees what the last finished copy left, as on the console's shared memory.
namespace { std::mutex late_mutex; std::unordered_set<u64> late_pages; std::atomic<bool> any_late{false}; }
void NoteLateReadRange(u64 address, u64 size) {
    std::lock_guard lock{late_mutex};
    for (u64 page = address & ~(track::PageSize - 1); page < address + size; page += track::PageSize) late_pages.insert(page);
    any_late.store(true, std::memory_order_release);
}
void MarkWritebackPending(const std::vector<std::pair<u64, u64>>& ranges, u64 serial) {
    if (ranges.empty()) return;
    track::InstallHandler();
    if (any_late.load(std::memory_order_acquire)) { // (the ranges without the late pages)
        std::vector<std::pair<u64, u64>> kept;
        { std::lock_guard lock{late_mutex};
          for (const auto& [a, b] : ranges) { u64 run = a;
              for (u64 page = a; page < b; page += track::PageSize) if (late_pages.count(page)) { if (run < page) kept.push_back({run, page}); run = page + track::PageSize; }
              if (run < b) kept.push_back({run, b}); } }
        if (kept != ranges) { static std::atomic<bool> inside{false}; // (once more, with what is left; no late page in it)
            if (kept.empty()) return;
            for (const auto& [a, b] : kept) for (u64 page = a; page < b; page += track::PageSize << 18) track::BackSerial(page, true);
            for (const auto& [a, b] : kept) track::BackSerial(b - track::PageSize, true);
            track::Guard guard;
            for (const auto& [a, b] : kept) {
                for (u64 page = a; page < b; page += track::PageSize) __atomic_store_n(track::BackSerial(page), serial + 1, __ATOMIC_RELEASE);
                mprotect(reinterpret_cast<void*>(a), b - a, PROT_NONE);
            }
            (void)inside; return; }
    }
    for (const auto& [a, b] : ranges) for (u64 page = a; page < b; page += track::PageSize << 18) track::BackSerial(page, true);
    for (const auto& [a, b] : ranges) track::BackSerial(b - track::PageSize, true);
    track::Guard guard;
    for (const auto& [a, b] : ranges) {
        for (u64 page = a; page < b; page += track::PageSize) __atomic_store_n(track::BackSerial(page), serial + 1, __ATOMIC_RELEASE);
        mprotect(reinterpret_cast<void*>(a), b - a, PROT_NONE);
    }
}
void WritebackDone(const std::vector<std::pair<u64, u64>>& ranges, u64 serial) {
    if (ranges.empty()) return;
    track::Guard guard;
    for (const auto& [a, b] : ranges) {
        u64 run = 0; bool open = false;
        for (u64 page = a; page < b; page += track::PageSize) {
            u64* back = track::BackSerial(page);
            const bool done = back && *back > 1 && *back <= serial + 1;
            if (done) { *back = 1; if (!open) { run = page; open = true; } }
            else if (open) { track::SettleRun(run, page); open = false; }
        }
        if (open) track::SettleRun(run, b);
    }
}

void PrepareHostWrite(u64 address, u64 size) {
    rt::EnsureCommitted(reinterpret_cast<void*>(address), size);
    if (!size || !track::dirty) return;
    track::SettleRange(address, size);
    const u64 first = address & ~(track::PageSize - 1), end = (address + size + track::PageSize - 1) & ~(track::PageSize - 1);
    track::Guard guard;
    for (u64 page = first; page < end; page += track::PageSize) {
        u8* state = track::State(page);
        if (state && *state == track::Protected) track::MarkDirty(page, *state, true);
        else if (u8* armed = track::ArmState(page); armed && *armed == track::Armed) track::MakeWritable(page);
    }
}

bool PageProtectionUsable() {
    static const bool usable = [] {
        // (OOT_NO_PAGE_PROTECTION=1: none anywhere, as on AMD)
        if (std::getenv("OOT_NO_PAGE_PROTECTION")) { RT_LOG("CPU writes: no page protection of imported memory (asked for)"); return false; }
#ifdef _WIN32
        // Windows: with texture pages left alone the game ran, but its menus showed shapes from stale buffers now and
        // then (wrong outlines flashing, a band across the screen, squares of noise): the GPU copies of the game's
        // buffers are kept current by write protection too. The GPU reads the game's memory itself there, as on AMD.
        // OOT_PAGE_PROTECTION=1: protected all the same.
        // (2026-11-18: the cause was the driver losing the device over pages that are read-only when the GPU first
        // reaches them - PrimeImport, above. With every import read once first, protection is as on Linux.
        // OOT_PAGE_PROTECTION=0: none, the slow way of the days between.)
        if (const char* e = std::getenv("OOT_PAGE_PROTECTION"); (e && std::atoi(e) == 0) || (!e && !ImportsPrimed())) { RT_LOG("CPU writes: Windows, no page protection of imported memory"); return false; }
#endif
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(Vk().physical, &p);
        const bool amd = p.vendorID == 0x1002;
        if (amd) RT_LOG("CPU writes: AMD GPU, no page protection of imported memory (textures checked by content)");
        return !amd;
    }();
    return usable;
}
// Whether CPU writes to texture memory are found by write-protecting its pages (the alternative: hashing each texture's
// texels every 16 frames, and no draws resolved ahead - a draw's textures cannot be trusted off the executor's thread).
// Imported memory cannot be protected on AMD (PageProtectionUsable), but where every pool is GPU memory mapped at its
// address or plain memory (placed maps: a Steam Deck) nothing is imported, and protecting such pages is ordinary.
// OOT_BACKED_WATCH=0: by content there, as before 2026-11-17.
bool PoolsInGpuMemory() { return BackPools() && Vk().placed_maps; }
bool TextureWatchUsable() {
    static const bool usable = [] {
        // (OOT_NO_TEXTURE_WATCH=1: by content everywhere)
        if (std::getenv("OOT_NO_TEXTURE_WATCH")) { RT_LOG("CPU writes: textures checked by content (asked for)"); return false; }
#ifdef _WIN32
        // Windows: NVIDIA's driver loses the device at the first submission after imported pages were write-protected
        // (an RTX 3050 laptop, driver 617.42: device lost as the first textures went in; none with the watch off).
        // OOT_TEXTURE_WATCH=1: protected all the same.
        if (const char* e = std::getenv("OOT_TEXTURE_WATCH"); (e && std::atoi(e) == 0) || (!e && !ImportsPrimed())) { RT_LOG("CPU writes: Windows, no page protection of texture memory (textures checked by content)"); return false; }
#endif
        if (PageProtectionUsable()) return true;
        const char* e = std::getenv("OOT_BACKED_WATCH");
        const bool v = BackPools() && Vk().placed_maps && !(e && std::atoi(e) == 0);
        if (v) RT_LOG("CPU writes: texture pages are write-protected (pools are GPU memory in place, nothing is imported)");
        return v;
    }();
    return usable;
}
u32 ArmTextureMemory(u64 address, u64 size) {
    const u32 sequence = track::write_sequence.load(std::memory_order_acquire);
    static const bool disabled = std::getenv("OOT_NO_TEXTURE_WATCH") != nullptr;
    if (disabled || !size || !TextureWatchUsable()) return sequence;
    track::InstallHandler();
    const u64 first = address & ~(track::PageSize - 1), end = (address + size + track::PageSize - 1) & ~(track::PageSize - 1);
    for (u64 page = first; page < end; page += track::PageSize) track::ArmState(page, true);
    for (u64 chunk = first >> track::ChunkShift; chunk <= (end - 1) >> track::ChunkShift; ++chunk) track::ChunkSequence(chunk << track::ChunkShift, true);
    track::SettleRange(address, size);
    track::Guard guard;
    u64 run = 0; bool open = false;
    const auto flush = [&](u64 to) { if (open) mprotect(reinterpret_cast<void*>(run), to - run, PROT_READ); open = false; };
    for (u64 page = first; page < end; page += track::PageSize) {
        u8& armed = *track::ArmState(page);
        if (armed == track::Armed) { flush(page); continue; }
        armed = track::Armed;
        if (!open) { run = page; open = true; }
    }
    flush(end);
    return sequence;
}
bool TextureMemoryWritten(u64 address, u64 size, u32 since) {
    if (!size) return false;
    const u64 first = address >> track::ChunkShift, last = (address + size - 1) >> track::ChunkShift;
    for (u64 chunk = first; chunk <= last; ++chunk) {
        const u32* sequence = track::ChunkSequence(chunk << track::ChunkShift);
        if (sequence && (s32)(__atomic_load_n(sequence, __ATOMIC_ACQUIRE) - since) > 0) return true;
    }
    return false;
}
u32 TextureWriteSequence() { return track::write_sequence.load(std::memory_order_acquire); }

std::atomic<u64> cpu_reuploads{0}; // (texture_data.cpp counts)
void ReportPoolStats() {
    {   // (each import is a userptr object on amdgpu, whose pages the kernel checks at every submission)
        std::lock_guard lock{imports_mutex}; u64 bytes = 0, n = 0, backed_bytes = 0, backed = 0;
        std::map<u64, u64> sizes; // (imports by size class, MiB)
        for (auto& [b0, w] : imports) if (auto i = w.lock()) {
            if (i->backed) { backed_bytes += i->size; ++backed; continue; }
            bytes += i->size; ++n; sizes[i->size >= (1u << 20) ? 1 : 0] += i->size; }
        RT_LOG("imports: %llu live, %.0f MiB (%.0f MiB of them 1 MiB or larger); %llu pools in GPU memory, %.0f MiB", (unsigned long long)n, bytes / 1048576.0,
               sizes[1] / 1048576.0, (unsigned long long)backed, backed_bytes / 1048576.0);
    }
    if (const u64 n = track::pending_faults.exchange(0)) RT_LOG("write-back: %llu CPU accesses waited for copies (%.2f ms per frame)",
        (unsigned long long)n, track::pending_wait_ns.exchange(0) / 1e6 / 300.0);
    if (const u64 faults = track::texture_write_faults.exchange(0)) RT_LOG("texture memory: %llu CPU-written pages, %llu re-uploads in 300 frames",
        (unsigned long long)faults, (unsigned long long)cpu_reuploads.exchange(0));
    if (!MirrorEnabled()) return;
    if (track::HotPagesOn()) { track::Guard g; RT_LOG("hot pages: %zu left writable, %zu seen", track::hot.size(), track::heat.size()); }
    RT_LOG("device pool copies: %.0f MiB; per frame %.2f MiB uploaded (%.0f regions, %.0f write faults), %.2f MiB written back",
           mirrored_bytes / 1048576.0, track::uploaded.exchange(0) / 300.0 / 1048576.0, track::upload_regions.exchange(0) / 300.0,
           track::faults.exchange(0) / 300.0, track::written_back.exchange(0) / 300.0 / 1048576.0);
}

std::string DescribeDeviceAddress(u64 address) {
    std::shared_lock lock{pools_mutex};
    auto& vk = Vk();
    for (const auto& [start, p] : pools_by_address) {
        for (VkBuffer b : {p->buffer, p->device_buffer}) {
            if (!b) continue;
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; info.buffer = b;
            const u64 base = vkGetBufferDeviceAddress(vk.device, &info);
            if (address >= base && address < base + p->import_size) {
                char text[160]; std::snprintf(text, sizeof text, "%s copy of pool %#llx (flags %#x) + %#llx (guest %#llx)", b == p->device_buffer ? "device" : "imported",
                    (unsigned long long)p->address, p->flags, (unsigned long long)(address - base), (unsigned long long)(p->import_base + address - base));
                return text;
            }
        }
    }
    return "not in a guest pool";
}
bool PoolsMirrored() { return MirrorEnabled(); }
} // namespace nvn

using namespace nvn;

// NVNmemoryPoolBuilder: +4 flags, +0x10 storage, +0x18 size.
NATIVE("nvnMemoryPoolInitialize") {
    const u64 object = c->x[0];
    const u8* builder = Ptr<u8>(c, 1);
    auto p = std::make_shared<Pool>();
    std::memcpy(&p->flags, builder + 4, 4);
    std::memcpy(&p->address, builder + 0x10, 8);
    std::memcpy(&p->size, builder + 0x18, 8);
    if (!p->size) Fatal("memory pool without a size");
    if (p->flags & PoolVirtual) {
        if (p->address) Fatal("virtual pool must not have CPU storage");
        const u64 reserved = (p->size + 0xffff) & ~0xffffULL;
        std::unique_lock lock{pools_mutex};
        if (!reserved || reserved > 0x4000000000ULL || next_virtual_address > 0x7000000000ULL - reserved)
            Fatal("GPU virtual address space exhausted");
        // (any free place will do: the game asks the pool for its address. A place taken already - by the graphics driver's
        // own memory, on Windows with NVIDIA DLSS built in: "cannot reserve GPU virtual pool" at every start of a player's
        // game - is passed over, a 64 KiB step at a time up to the region's end)
        {static const bool test=[]{if(!std::getenv("OOT_TEST_TAKE_VIRTUAL"))return false; // (test: the first places taken, as the driver did)
            return mmap(reinterpret_cast<void*>(0x3000000000ULL),0x40000000ULL,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE|MAP_NORESERVE,-1,0)!=MAP_FAILED;}();(void)test;}
        void* memory = MAP_FAILED; u32 passed = 0;
        while (next_virtual_address <= 0x7000000000ULL - reserved) {
            memory = mmap(reinterpret_cast<void*>(next_virtual_address), reserved, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
            if (memory != MAP_FAILED && reinterpret_cast<u64>(memory) == next_virtual_address) break;
            if (memory != MAP_FAILED) munmap(memory, reserved); // (placed elsewhere: not this place)
            memory = MAP_FAILED; ++passed;
            next_virtual_address += passed < 64 ? 0x10000ULL : 0x40000000ULL; // (64 KiB steps, then 1 GiB ones past a large occupant)
        }
        if (memory == MAP_FAILED) Fatal("cannot reserve GPU virtual pool");
        if (passed) RT_LOG("memory: a GPU virtual pool placed at %#llx, past %u taken places", (unsigned long long)next_virtual_address, passed);
        p->address = next_virtual_address;
        next_virtual_address += reserved;
        p->import_base = p->address;
    } else {
        if (!p->address) Fatal("memory pool without storage");
        // (small pools: imported on their first GPU use; see "importing guest pages")
        // (OOT_LAZY_POOL_IMPORT=1: small pools come to the GPU at their first GPU use; =2: every pool does. With every
        // pool brought at creation a Steam Deck had 4.9 GiB of the game's memory in GPU memory at the title, most of it
        // texture data the GPU never reads, and ran out at the first area: "Not enough memory for command submission".)
        // (the default where pools live in GPU memory, i.e. integrated GPUs: 2. There the pools the GPU never binds - most
        // of the texture data - stay ordinary memory: 5.0 -> 3.65 GiB of GPU-mapped memory on the Deck in Kokiri, where
        // turning to the forest had run the 9 GiB out. OOT_LAZY_POOL_IMPORT=0: every pool at creation, as on other GPUs.)
        static const int lazy = std::getenv("OOT_LAZY_POOL_IMPORT") ? std::max(0, std::atoi(std::getenv("OOT_LAZY_POOL_IMPORT"))) : BackPools() ? 2 : 0;
        if (lazy == 0 || (lazy == 1 && p->size > (64u << 10))) ImportPool(*p);
        else { const u64 align = Vk().host_import_alignment; p->import_base = p->address & ~(align - 1);
               p->import_size = ((p->address + p->size + align - 1) & ~(align - 1)) - p->import_base; }
    }
    if (std::getenv("OOT_POOL_TRACE")) RT_LOG("memory pool %#llx size %#llx flags %#x", (unsigned long long)p->address, (unsigned long long)p->size, p->flags);
    HostPool(object) = p.get();
    // Serialized shader archives contain zero-initialized pool objects and query
    // this field before initializing them (the SDK getter is just LDR +0x18).
    std::memcpy(reinterpret_cast<void*>(object + 0x18), &p->address, sizeof(p->address));
    std::shared_ptr<Pool> retired;
    {
        std::unique_lock lock{pools_mutex};
        retired = std::move(pools_by_address[p->address]);
        pools_by_address[p->address] = std::move(p);
        table_dirty = true; ++pool_generation;
        if (retired) table_changed = true; // (a pool in place of another; a new one is only added)
    }
    if (retired && (retired->buffer || retired->host_import || retired->device_buffer)) Flush(); // (as in nvnMemoryPoolFinalize)
    Ret(c, 1);
}
NATIVE("nvnMemoryPoolFinalize") {
    Pool* p = HostPool(c->x[0]);
    if (std::getenv("OOT_POOL_TRACE")) RT_LOG("memory pool finalize %#llx", p ? (unsigned long long)p->address : 0ull);
    if (!p) return;
    // Commands recorded so far may still use the pool's buffer (descriptors, vertex data, report writes): submit
    // them before the buffer and its memory are released, or the GPU reads and writes freed addresses (amdgpu page
    // faults and GPU resets on the Steam Deck while scenery streams in).
    if (p->buffer || p->host_import || p->device_buffer) Flush();
    DropDeviceAhead(p->address, p->size);
    std::shared_ptr<Pool> retired;
    {
        std::unique_lock lock{pools_mutex};
        auto it = pools_by_address.find(p->address);
        if (it != pools_by_address.end()) {
            retired = std::move(it->second);
            pools_by_address.erase(it);
            table_dirty = true; table_changed = true; ++pool_generation;
        }
    }
    HostPool(c->x[0]) = nullptr;
    *reinterpret_cast<u64*>(c->x[0] + 0x18) = 0;
}
NATIVE("nvnMemoryPoolMap") {
    const Pool* p = HostPool(c->x[0]);
    Ret(c, p->flags & (PoolVirtual | PoolCpuNoAccess) ? 0 : p->address);
}
NATIVE("nvnMemoryPoolMapVirtual") { Ret(c, MapVirtual(*HostPool(c->x[0]), (s32)c->x[1], Ptr<MappingRequest>(c, 2))); }
NATIVE("nvnMemoryPoolGetBufferAddress") { Ret(c, *reinterpret_cast<u64*>(c->x[0] + 0x18)); }
NATIVE("nvnMemoryPoolGetSize") { Ret(c, HostPool(c->x[0])->size); }
NATIVE("nvnMemoryPoolGetFlags") { Ret(c, HostPool(c->x[0])->flags); }
NATIVE("nvnMemoryPoolFlushMappedRange") {}      // host-coherent import
NATIVE("nvnMemoryPoolInvalidateMappedRange") { // (pool, offset, size)
    if (Pool* p = HostPool(c->x[0])) nvn::WaitForWrites(p->address + c->x[1], c->x[2]);
}
NATIVE("nvnMemoryPoolSetDebugLabel") {}
