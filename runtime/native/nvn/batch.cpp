#include <thread>
#include <condition_variable>
#include <mutex>
#include <string>
#include "trace_compat.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <map>
#include <set>
#include <vector>

#include "../../native.h"
#include "batch.h"
#include "../../platform.h"
#include "memory.h"
#include "commands.h"

namespace nvn {
namespace {

struct Batch {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<std::function<void()>> after;
    std::vector<VkDescriptorPool> pools; // all reset after a flush
    size_t pool_in_use = 0;
    std::map<u64, u64> pending_writes; // start -> end of guest ranges written by the open batch
    std::map<u64, u64> pending_writeback; // those whose results are copied back to guest memory
    std::map<u64, u64> pending_reports;
    bool pass_open = false;            // a dynamic-rendering instance is open in cmd
    std::function<void(VkCommandBuffer)> pass_end_hook;
    std::function<void()> pre_submit_hook;
    VkQueryPool timer = VK_NULL_HANDLE; // batch start/end timestamps
    std::atomic<u64> gpu_ns{0}, wait_ns{0}, writeback_ns{0};
    std::vector<u64> pass_key;
    // Since the last render pass instance ended, only "light" work (with its own narrow
    // barriers) was recorded: reopening the same attachments then needs only an attachment barrier.
    std::vector<u64> ended_pass_key;
    bool light_only = false;
} batch;
// Submitted batches the GPU may still be executing, oldest first. Their deferred actions run, in
// order, once they complete (RetireOne); the CPU records the next batch meanwhile.
struct InFlight {
    VkCommandBuffer cmd = VK_NULL_HANDLE, uploads = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool timer = VK_NULL_HANDLE;
    u64 serial = 0; // (the timeline value signalled at its completion)
    std::vector<std::function<void()>> after;
    std::vector<VkDescriptorPool> pools;
    std::map<u64, u64> writes, reports;
    std::vector<u64> watched; // (OOT_READ_WATCH)
};
std::deque<InFlight> in_flight;
struct Writeback { u64 serial; VkCommandBuffer cmd; std::vector<std::pair<u64, u64>> pages; };
std::deque<Writeback> writebacks; // (submitted write-back copies, oldest first)
// (batch lock held) Completed write-back copies: their pages become accessible again.
void ReapWritebacks() {
    if (writebacks.empty()) return;
    auto& vk = Vk();
    u64 done = 0;
    VK_CHECK(vkGetSemaphoreCounterValue(vk.device, WritebackTimeline(), &done));
    while (!writebacks.empty() && writebacks.front().serial <= done) {
        Writeback& w = writebacks.front();
        WritebackDone(w.pages, w.serial);
        { std::lock_guard queue{vk.queue_mutex}; vkFreeCommandBuffers(vk.device, vk.command_pool, 1, &w.cmd); }
        writebacks.pop_front();
    }
}
std::vector<VkFence> spare_fences;
std::vector<VkQueryPool> spare_timers;
std::vector<VkDescriptorPool> spare_pools;
u64 submitted_serial = 0, retired_serial = 0;
bool reopen_light = false;
u64 record_serial = 0;
u64 other_serial = 0; // changes when anything but a pass draw is recorded (draw state caching) // changes whenever anything is recorded or submitted
u64 stat_unordered = 0;
u64 stat_passes = 0, stat_full_barriers = 0, stat_items = 0, stat_reopens = 0; // the pass being begun follows only light work (see PassDraw) // batches submitted / completed
constexpr size_t MaxInFlight = 16; // about two frames of batches; guest fences throttle the game
struct GpuProfile {
    static constexpr u32 Capacity=8192;
    VkQueryPool pool=VK_NULL_HANDLE;
    std::vector<u64> keys;
    std::map<u64,std::pair<double,u64>> totals;
} gpu_profile;
bool GpuProfiling() { static const bool enabled=std::getenv("OOT_GPU_PROFILE")!=nullptr;return enabled; }

VkDescriptorPool MakePool() {
    auto& vk = Vk();
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024}};
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.maxSets = 2048;
    info.poolSizeCount = (u32)std::size(sizes);
    info.pPoolSizes = sizes;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(vk.device, &info, nullptr, &pool));
    return pool;
}

} // namespace

BatchLock& BatchMutex() {
    static BatchLock mutex;
    return mutex;
}
// Signalled with each batch's serial when it completes: an outermost waiter releases the batch lock
// while it waits on this (a fence would be recycled by whichever thread retires the batch), so other
// threads record and submit the next work while the GPU finishes the last.
VkSemaphore MakeTimeline() {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore s = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSemaphore(Vk().device, &info, nullptr, &s));
        return s;
}
// Signalled with a batch's serial when its write-back copy (memory.cpp, "GPU writes on their way back")
// has completed.
VkSemaphore WritebackTimeline() { static VkSemaphore s = MakeTimeline(); return s; }
void WaitWriteback(u64 serial) {
    const VkSemaphore s = WritebackTimeline();
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1; wi.pSemaphores = &s; wi.pValues = &serial;
    VK_CHECK(vkWaitSemaphores(Vk().device, &wi, UINT64_MAX));
}
VkSemaphore Timeline() {
    static VkSemaphore timeline = [] {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VkSemaphore s = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSemaphore(Vk().device, &info, nullptr, &s));
        return s;
    }();
    return timeline;
}

// Accesses recorded since the last full barrier (see PassAccess).
namespace {
struct Unsynced {
    std::vector<const void*> read_textures, written_textures;
    std::vector<std::pair<u64, u64>> reads, writes;
    bool global_read = false, global_write = false, unknown = false;
    void Clear() { read_textures.clear(); written_textures.clear(); reads.clear(); writes.clear(); global_read = global_write = unknown = false; }
    static bool Overlap(const std::vector<std::pair<u64, u64>>& a, const std::vector<std::pair<u64, u64>>& b) {
        for (const auto& x : a) for (const auto& y : b) if (x.first < y.second && y.first < x.second) return true;
        return false;
    }
    static bool Common(const std::vector<const void*>& a, const std::vector<const void*>& b) {
        for (auto* x : a) for (auto* y : b) if (x == y) return true;
        return false;
    }
    bool Conflicts(const PassAccess& a) const {
        if (unknown) return true;
        if (Common(a.read_textures, written_textures) || Common(a.written_textures, written_textures) || Common(a.written_textures, read_textures)) return true;
        const bool any_writes = !writes.empty() || global_write, any_reads = !reads.empty() || global_read;
        if (a.global_read && any_writes) return true;
        if (a.global_write && (any_writes || any_reads)) return true;
        if ((global_write && (!a.reads.empty() || !a.writes.empty())) || (global_read && !a.writes.empty())) return true;
        return Overlap(a.reads, writes) || Overlap(a.writes, writes) || Overlap(a.writes, reads);
    }
    void Add(const PassAccess& a) {
        constexpr size_t Cap = 4096; // (past this many, everything conflicts: the next pass takes a barrier)
        if (reads.size() + writes.size() + read_textures.size() + written_textures.size() > Cap) { unknown = true; return; }
        read_textures.insert(read_textures.end(), a.read_textures.begin(), a.read_textures.end());
        written_textures.insert(written_textures.end(), a.written_textures.begin(), a.written_textures.end());
        reads.insert(reads.end(), a.reads.begin(), a.reads.end());
        writes.insert(writes.end(), a.writes.begin(), a.writes.end());
        global_read |= a.global_read; global_write |= a.global_write;
    }
} unsynced;
u64 stat_passes_unbarriered = 0;
} // namespace
void EndPassLocked() {
    if (!batch.pass_open) return;
    static const bool trace = std::getenv("OOT_PASS_TRACE") != nullptr; // (what ends render passes: callers by offset)
    if (trace) {
        static std::map<std::string, u64> callers; static u64 n = 0;
        void* frames[6]; const int k = backtrace(frames, 6);
        Dl_info info{}; dladdr(reinterpret_cast<void*>(&BatchMutex), &info);
        std::string key;
        for (int i = 1; i < k && i < 6; ++i) { char t[24]; std::snprintf(t, sizeof t, " %#llx", (unsigned long long)(reinterpret_cast<u64>(frames[i]) - reinterpret_cast<u64>(info.dli_fbase) - 1)); key += t; }
        ++callers[key];
        if (++n % 30000 == 0) {
            std::vector<std::pair<u64, std::string>> top;
            for (auto& [kk, c] : callers) top.push_back({c, kk});
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 12; ++i) RT_LOG("pass ended by%s: %llu", top[i].second.c_str(), (unsigned long long)top[i].first);
            callers.clear();
        }
    }
    if (batch.pass_end_hook) { auto hook = std::move(batch.pass_end_hook); batch.pass_end_hook = nullptr; hook(batch.cmd); }
    vkCmdEndRendering(batch.cmd);
    batch.pass_open = false;
    batch.ended_pass_key = batch.pass_key;
    batch.light_only = true;
}

bool RecordInPass(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard lock{BatchMutex()};
    if (!batch.cmd || !batch.pass_open) return false;
    record(batch.cmd);
    return true;
}

void SetPassEndHook(std::function<void(VkCommandBuffer)> hook) {
    std::lock_guard lock{BatchMutex()};
    batch.pass_end_hook = std::move(hook);
}

void SetPreSubmitHook(std::function<void()> hook) {
    std::lock_guard lock{BatchMutex()};
    batch.pre_submit_hook = std::move(hook);
}

void EndPass() {
    std::lock_guard lock{BatchMutex()};
    EndPassLocked();
}

// Opens the batch command buffer, or orders the next work after everything recorded so far.
void BeginWork() {
    auto& vk = Vk();
    ++record_serial;
    ++other_serial;
    EndPassLocked();
    if (!batch.cmd) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = vk.command_pool;
        ai.commandBufferCount = 1;
        std::lock_guard queue{vk.queue_mutex}; // the command pool is shared with presentation
        VK_CHECK(vkAllocateCommandBuffers(vk.device, &ai, &batch.cmd));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(batch.cmd, &bi));
        if (!spare_timers.empty()) { batch.timer = spare_timers.back(); spare_timers.pop_back(); }
        else {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = 3;
            VK_CHECK(vkCreateQueryPool(vk.device, &qi, nullptr, &batch.timer));
        }
        vkCmdResetQueryPool(batch.cmd, batch.timer, 0, 3);
        // Batches are submitted without waiting: order this one after all earlier submissions
        // (a barrier's scope covers prior commands in queue submission order).
        VkMemoryBarrier first{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        first.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        first.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(batch.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &first, 0, nullptr, 0, nullptr);
        vkCmdWriteTimestamp(batch.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, batch.timer, 0);
        batch.light_only = false;
        batch.ended_pass_key.clear();
        unsynced.Clear();
    } else {
        // Each piece of work sees everything recorded before it.
        work_kind=0;
        unsynced.Clear();
        ++stat_full_barriers;
        batch.light_only = false;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(batch.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
    }
}

bool PassOpenWithKey(const std::vector<u64>& key) {
    std::lock_guard lock{BatchMutex()};
    return batch.cmd && batch.pass_open && batch.pass_key == key && !GpuProfiling();
}

void NoteUnorderedWork() { std::lock_guard lock{BatchMutex()}; unsynced.unknown = true; }

void PassDraw(const std::vector<u64>& key, const std::function<void(VkCommandBuffer)>& begin,
              const std::function<void(VkCommandBuffer)>& draw, u64 profile_key, const PassAccess* access) {
    std::lock_guard lock{BatchMutex()};
    if (GpuProfiling()) { // per-item timing needs separate items
        Immediate([&](VkCommandBuffer cmd) {
            begin(cmd); draw(cmd);
            if (batch.pass_end_hook) { auto hook = std::move(batch.pass_end_hook); batch.pass_end_hook = nullptr; hook(cmd); }
            vkCmdEndRendering(cmd); }, profile_key);
        return;
    }
    if (!batch.cmd || !batch.pass_open || batch.pass_key != key) {
        if (batch.cmd && !batch.pass_open && batch.light_only && batch.ended_pass_key == key) {
            // Same attachments as the pass that just ended, with only light work in between:
            // order attachment accesses across the two instances, without draining everything.
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            vkCmdPipelineBarrier(batch.cmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            batch.light_only = false;
            reopen_light = true;
            ++stat_reopens;
        } else {
            static const bool no_barrier = std::getenv("OOT_PASS_NO_BARRIER") != nullptr; // (measurement only: unsafe)
            static const bool always_barrier = std::getenv("OOT_PASS_HAZARDS") == nullptr; // (dependency tracking: no gain measured yet)
            if (batch.cmd && (no_barrier || (!always_barrier && access && !unsynced.Conflicts(*access)))) {
                // (independent of everything since the last full barrier: no drain)
                EndPassLocked(); ++record_serial; ++other_serial; ++stat_passes_unbarriered;
            } else { work_kind = 2; BeginWork(); }
        }
        begin(batch.cmd);
        ++stat_passes;
        reopen_light = false;
        batch.pass_open = true;
        batch.pass_key = key;
    }
    draw(batch.cmd);
    ++record_serial;
    static const bool tracking = std::getenv("OOT_PASS_HAZARDS") != nullptr;
    if (tracking) { if (access) unsynced.Add(*access); else unsynced.unknown = true; }
}

u64 RecordSerial() { std::lock_guard lock{BatchMutex()}; return record_serial; }
u64 OtherWorkSerial() { return other_serial; } // (batch lock held by the caller's recording)

bool ImmediateUnordered(const std::function<void(VkCommandBuffer)>& record, u64 serial) {
    std::lock_guard lock{BatchMutex()};
    if (GpuProfiling() || !batch.cmd || batch.pass_open || serial != record_serial) return false;
    ++stat_items; ++stat_unordered;
    batch.light_only = false;
    unsynced.unknown = true;
    record(batch.cmd);
    ++record_serial;
    ++other_serial;
    return true;
}

void ImmediateLight(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard lock{BatchMutex()};
    ++stat_items;
    if (GpuProfiling() || !batch.cmd) { Immediate(record); return; }
    const bool light = batch.pass_open || batch.light_only;
    EndPassLocked();
    unsynced.unknown = true;
    ++record_serial;
    ++other_serial;
    batch.light_only = light;
    record(batch.cmd);
}

void Immediate(const std::function<void(VkCommandBuffer)>& record, u64 profile_key) {
    std::lock_guard lock{BatchMutex()};
    static const bool trace = std::getenv("OOT_IMMEDIATE_TRACE") != nullptr;
    if (trace) {
        static std::map<void*, u64> callers; static u64 total = 0;
        ++callers[__builtin_return_address(0)];
        if (++total % 200000 == 0) {
            std::vector<std::pair<u64, void*>> top;
            for (auto& [a, n] : callers) top.push_back({n, a});
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 10; ++i) {
#ifdef _WIN32
                RT_LOG("immediate caller %p: %llu", top[i].second, (unsigned long long)top[i].first);
#else
                Dl_info di{};dladdr(top[i].second,&di);RT_LOG("immediate caller +%#lx: %llu", (unsigned long)((char*)top[i].second-(char*)di.dli_fbase), (unsigned long long)top[i].first);
#endif
            }
            callers.clear();
        }
    }
    auto& vk = Vk();
    ++stat_items;
    BeginWork();
    {
    const bool profile=GpuProfiling()&&gpu_profile.keys.size()*2<GpuProfile::Capacity;
    const u32 query=gpu_profile.keys.size()*2;
    if(profile){
        if(!gpu_profile.pool){VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qi.queryType=VK_QUERY_TYPE_TIMESTAMP;qi.queryCount=GpuProfile::Capacity;
            VK_CHECK(vkCreateQueryPool(vk.device,&qi,nullptr,&gpu_profile.pool));}
        vkCmdResetQueryPool(batch.cmd,gpu_profile.pool,query,2);
        vkCmdWriteTimestamp(batch.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,gpu_profile.pool,query);
        gpu_profile.keys.push_back(profile_key);
    }
    record(batch.cmd);
    if(profile)vkCmdWriteTimestamp(batch.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,gpu_profile.pool,query+1);
    }
}

void ReportGpuProfile() {
    if(!GpuProfiling())return;
    std::vector<std::pair<double,u64>> top;
    for(const auto& [key,value]:gpu_profile.totals)top.emplace_back(value.first,key);
    std::sort(top.rbegin(),top.rend());
    {double sum=0;u64 calls=0;for(const auto& [k,v]:gpu_profile.totals){sum+=v.first;calls+=v.second;}
     RT_LOG("GPU profile total: %.3f ms/frame in %llu items over %zu shaders",sum/300e6,(unsigned long long)calls/300,gpu_profile.totals.size());}
    for(size_t i=0;i<std::min<size_t>(12,top.size());++i){const auto [ns,key]=top[i];
        RT_LOG("GPU profile shader %#llx: %.3f ms/frame, %llu calls",(unsigned long long)key,ns/300e6,(unsigned long long)gpu_profile.totals[key].second);}
    gpu_profile.totals.clear();
}

std::atomic<u64> flush_count{0};
u64 FlushCount() { return flush_count.load(); }
void ReportDeviceAhead();
void ReportBatchStats() {
    RT_LOG("batch structure per frame: %.1f render passes (%.1f reopened cheaply), %.1f full barriers (%.1f compute dispatches recorded without one), %.1f other items",
           stat_passes / 300.0, stat_reopens / 300.0, stat_full_barriers / 300.0, stat_unordered / 300.0, stat_items / 300.0);
    stat_unordered = 0;
    ReportPoolStats();
    ReportDeviceAhead();
    RT_LOG("guest blocked on the GPU per frame: %.2f ms in sync waits, %.2f ms in readbacks, %.2f ms in queue finishes",
           cpu_wait_ns[0].exchange(0) / 300e6, cpu_wait_ns[1].exchange(0) / 300e6, cpu_wait_ns[2].exchange(0) / 300e6);
    stat_passes = stat_full_barriers = stat_items = stat_reopens = 0;
}
bool ReopeningAfterLightWork() { return reopen_light; }
u64 GpuBusyNanoseconds() { return batch.gpu_ns.load(); }
u64 FenceWaitNanoseconds() { return batch.wait_ns.load(); }
u64 WritebackNanoseconds() { return batch.writeback_ns.load(); }

namespace {
// Completes the oldest in-flight batch: waits for it if `wait`, else only if already done.
bool RetireOne(bool wait) {
    auto& vk = Vk();
    if (in_flight.empty()) return false;
    // (the game's sync waits held the batch lock while the GPU worked: the executor, blocked behind it 23% of
    // its time in Kokiri, could not record the next frame meanwhile. OOT_LOCKED_GPU_WAITS=1: as before.)
    static const bool unlocked_waits = std::getenv("OOT_LOCKED_GPU_WAITS") == nullptr;
    if (wait && unlocked_waits && BatchMutex().depth == 1 && vkGetFenceStatus(vk.device, in_flight.front().fence) != VK_SUCCESS) {
        const u64 serial = in_flight.front().serial;
        const VkSemaphore timeline = Timeline();
        BatchMutex().unlock();
        {
            rt::CoreRelease release; // other threads of this core run while the GPU works
            const auto t0 = std::chrono::steady_clock::now();
            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.semaphoreCount = 1; wi.pSemaphores = &timeline; wi.pValues = &serial;
            VK_CHECK(vkWaitSemaphores(vk.device, &wi, UINT64_MAX));
            batch.wait_ns += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        BatchMutex().lock();
        if (in_flight.empty() || in_flight.front().serial != serial) return true; // (another thread retired it)
    }
    InFlight& f = in_flight.front();
    if (!wait && vkGetFenceStatus(vk.device, f.fence) != VK_SUCCESS) return false;
    if (wait) {
        rt::CoreRelease release; // other threads of this core run while the GPU works
        const auto t0 = std::chrono::steady_clock::now();
        VK_CHECK(vkWaitForFences(vk.device, 1, &f.fence, VK_TRUE, UINT64_MAX));
        batch.wait_ns += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    VK_CHECK(vkResetFences(vk.device, 1, &f.fence));
    spare_fences.push_back(f.fence);
    u64 stamps[3]{};
    if (TimelineOn() && vkGetQueryPoolResults(vk.device, f.timer, 0, 3, sizeof stamps, stamps, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && vk.calibrated_timestamps) {
        // (GPU start and end of the batch on the timeline's clock)
        static auto calibrate = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(vkGetDeviceProcAddr(vk.device, vk.calibrated_timestamps));
        static double offset_ms = 0; static auto calibrated = std::chrono::steady_clock::time_point{};
        if (std::chrono::steady_clock::now() - calibrated > std::chrono::seconds(2)) {
            VkCalibratedTimestampInfoKHR info[2]{{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}, {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}};
            info[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR; info[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
            u64 t[2]{}, deviation = 0;
            if (calibrate(vk.device, 2, info, t, &deviation) == VK_SUCCESS) {
                const double mono_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
                const double base_ms = mono_ms - TimelineMs(); // (the timeline's zero on the monotonic clock)
                offset_ms = t[1] / 1e6 - t[0] * (double)vk.properties.limits.timestampPeriod / 1e6 - base_ms;
                calibrated = std::chrono::steady_clock::now();
            }
        }
        const double period = vk.properties.limits.timestampPeriod / 1e6;
        RT_LOG("T %.2f GPU batch %llu ran %.2f-%.2f", TimelineMs(), (unsigned long long)f.serial, stamps[0] * period + offset_ms, stamps[1] * period + offset_ms);
    }
    if (vkGetQueryPoolResults(vk.device, f.timer, 0, 3, sizeof stamps, stamps, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && stamps[1] > stamps[0]) {
        // (batches overlap on the GPU timeline when one starts before the previous ends: only new time counts)
        static u64 last_end = 0;
        const u64 from = std::max(stamps[0], last_end);
        if (stamps[1] > from) batch.gpu_ns += (u64)((stamps[1] - from) * (double)vk.properties.limits.timestampPeriod);
        last_end = std::max(last_end, stamps[1]);
        if (stamps[1] >= stamps[2] && stamps[2] >= stamps[0]) batch.writeback_ns += (u64)((stamps[1] - stamps[2]) * (double)vk.properties.limits.timestampPeriod);
    }
    spare_timers.push_back(f.timer);
    if(!gpu_profile.keys.empty()){
        std::vector<u64> times(gpu_profile.keys.size()*2);
        VK_CHECK(vkGetQueryPoolResults(vk.device,gpu_profile.pool,0,times.size(),times.size()*sizeof(u64),times.data(),sizeof(u64),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
        for(size_t i=0;i<gpu_profile.keys.size();++i){auto& total=gpu_profile.totals[gpu_profile.keys[i]];
            total.first+=(times[i*2+1]-times[i*2])*vk.properties.limits.timestampPeriod;++total.second;}
        gpu_profile.keys.clear();
    }
    {
        std::lock_guard queue{vk.queue_mutex};
        vkFreeCommandBuffers(vk.device, vk.command_pool, 1, &f.cmd);
        if (f.uploads) vkFreeCommandBuffers(vk.device, vk.command_pool, 1, &f.uploads);
    }
    UnwatchPages(f.watched);
    for (VkDescriptorPool pool : f.pools) { vkResetDescriptorPool(vk.device, pool, 0); spare_pools.push_back(pool); }
    auto actions = std::move(f.after);
    in_flight.pop_front();
    ++retired_serial;
    for (auto& action : actions) action();
    ReapWritebacks();
    return true;
}
} // namespace

namespace {
// OOT_WAIT_TRACE: which callers block on the GPU, by time waited.
void TraceWait(void*, u64 waited_ns) {
    // (by thread and the four callers up, as offsets in the executable: addr2line -f -C -e tootr <offsets>)
    static const bool trace = std::getenv("OOT_WAIT_TRACE") != nullptr;
    if (!trace || waited_ns < 20'000) return;
    static std::mutex m; static std::map<std::string, std::pair<u64, u64>> callers; static auto since = std::chrono::steady_clock::now();
    void* frames[7]; const int n = backtrace(frames, 7);
    Dl_info info{}; dladdr(reinterpret_cast<void*>(&BatchMutex), &info);
    char name[32] = "?"; pthread_getname_np(pthread_self(), name, sizeof name);
    std::string key = name;
    for (int i = 2; i < n && i < 7; ++i) { char t[24]; std::snprintf(t, sizeof t, " %#llx", (unsigned long long)(reinterpret_cast<u64>(frames[i]) - reinterpret_cast<u64>(info.dli_fbase) - 1)); key += t; }
    std::lock_guard lock{m};
    auto& e = callers[key]; e.first += waited_ns; ++e.second;
    if (std::chrono::steady_clock::now() - since > std::chrono::seconds(10)) {
        since = std::chrono::steady_clock::now();
        std::vector<std::pair<u64, std::string>> top;
        for (auto& [k, v] : callers) top.push_back({v.first, k});
        std::sort(top.rbegin(), top.rend());
        for (size_t i = 0; i < top.size() && i < 10; ++i) RT_LOG("waits %.0f ms (%llu) %s", top[i].first / 1e6, (unsigned long long)callers[top[i].second].second, top[i].second.c_str());
        callers.clear();
    }
}
} // namespace

namespace {
std::atomic<u64> submitted_atomic{0};
std::mutex retire_mutex;
std::condition_variable retire_cv;
void StartRetirer();
} // namespace

void Submit() {
    std::lock_guard lock{BatchMutex()};
    auto& vk = Vk();
    if (batch.cmd && batch.pre_submit_hook) batch.pre_submit_hook();
    if (!batch.cmd) return;
    ++flush_count; ++record_serial; ++other_serial;
    { static const bool t = std::getenv("OOT_TIMELINE") != nullptr; if (t) { static const auto t0 = std::chrono::steady_clock::now(); (void)t0; } }
    static const bool trace = std::getenv("OOT_FLUSH_TRACE") != nullptr;
    if (trace) {
        // (three callers up, as offsets in the executable: addr2line -f -e tootr <offsets>)
        static std::map<std::string, u64> callers;
        void* frames[6]; const int n = backtrace(frames, 6);
        Dl_info info{}; dladdr(reinterpret_cast<void*>(&BatchMutex), &info);
        std::string key;
        for (int i = 1; i < n && i < 5; ++i) { char t[24]; std::snprintf(t, sizeof t, " %#llx", (unsigned long long)(reinterpret_cast<u64>(frames[i]) - reinterpret_cast<u64>(info.dli_fbase) - 1)); key += t; }
        ++callers[key];
        if (flush_count % 3000 == 0) {
            std::vector<std::pair<u64, std::string>> top;
            for (auto& [k, c] : callers) top.push_back({c, k});
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 10; ++i) RT_LOG("flush callers%s: %llu", top[i].second.c_str(), (unsigned long long)top[i].first);
            callers.clear();
        }
    }
    EndPassLocked();
    InFlight f;
    VkCommandBuffer writeback_cmd = VK_NULL_HANDLE;
    std::vector<std::pair<u64, u64>> pending_pages;
    // Guest memory: CPU writes go to the pools' device copies before the batch, its GPU writes
    // come back to guest memory at its end.
    {
        std::lock_guard queue{vk.queue_mutex}; // the command pool is shared with presentation
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = vk.command_pool; ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(vk.device, &ai, &f.uploads));
        VK_CHECK(vkAllocateCommandBuffers(vk.device, &ai, &writeback_cmd));
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(f.uploads, &bi));
    VK_CHECK(vkBeginCommandBuffer(writeback_cmd, &bi));
    bool any_uploads = false;
    vkCmdWriteTimestamp(batch.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, batch.timer, 2); // (write-back start)
    auto pools_lock = RecordPoolTransfers(f.uploads, &any_uploads, batch.cmd, batch.pending_writeback, batch.pending_reports,
                                          writeback_cmd, &pending_pages);
    VK_CHECK(vkEndCommandBuffer(f.uploads));
    VK_CHECK(vkEndCommandBuffer(writeback_cmd));
    vkCmdWriteTimestamp(batch.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, batch.timer, 1);
    VK_CHECK(vkEndCommandBuffer(batch.cmd));
    if (!spare_fences.empty()) { f.fence = spare_fences.back(); spare_fences.pop_back(); }
    else { VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK_CHECK(vkCreateFence(vk.device, &fi, nullptr, &f.fence)); }
    {
        std::lock_guard queue{vk.queue_mutex};
        const VkCommandBuffer buffers[2]{f.uploads, batch.cmd};
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        f.serial = submitted_serial + 1;
        const VkSemaphore timeline = Timeline();
        VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ti.signalSemaphoreValueCount = 1; ti.pSignalSemaphoreValues = &f.serial;
        si.pNext = &ti; si.signalSemaphoreCount = 1; si.pSignalSemaphores = &timeline;
        si.commandBufferCount = any_uploads ? 2 : 1;
        si.pCommandBuffers = any_uploads ? buffers : &batch.cmd;
        {   // (time in vkQueueSubmit: on amdgpu it grows with the imported memory)
            const auto t0 = std::chrono::steady_clock::now();
            VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, f.fence));
            if (TimelineOn()) RT_LOG("T %.2f submit batch %llu", TimelineMs(), (unsigned long long)f.serial);
            static double total = 0; static u64 n = 0;
            total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (++n % 500 == 0) { RT_LOG("vkQueueSubmit: %.2f ms each (last 500)", total / 500); total = 0; }
        }
        if (!any_uploads) { vkFreeCommandBuffers(vk.device, vk.command_pool, 1, &f.uploads); f.uploads = VK_NULL_HANDLE; }
        if (!pending_pages.empty()) { // (the write-back copy: after the batch, which completes without it)
            const VkSemaphore wb = WritebackTimeline();
            // On its own queue (the main queue's next submissions, such as the fence the game waits on,
            // do not queue behind it), after the batch (its timeline value).
            const bool own = vk.copy_queue != vk.queue;
            VkTimelineSemaphoreSubmitInfo wti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            wti.signalSemaphoreValueCount = 1; wti.pSignalSemaphoreValues = &f.serial;
            wti.waitSemaphoreValueCount = own ? 1 : 0; wti.pWaitSemaphoreValues = &f.serial;
            const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo wsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            wsi.pNext = &wti; wsi.signalSemaphoreCount = 1; wsi.pSignalSemaphores = &wb;
            if (own) { wsi.waitSemaphoreCount = 1; wsi.pWaitSemaphores = &timeline; wsi.pWaitDstStageMask = &stage; }
            wsi.commandBufferCount = 1; wsi.pCommandBuffers = &writeback_cmd;
            if (own) { std::lock_guard copy{vk.copy_mutex}; VK_CHECK(vkQueueSubmit(vk.copy_queue, 1, &wsi, VK_NULL_HANDLE)); }
            else VK_CHECK(vkQueueSubmit(vk.queue, 1, &wsi, VK_NULL_HANDLE));
        } else {
            vkFreeCommandBuffers(vk.device, vk.command_pool, 1, &writeback_cmd);
        }
    }
    pools_lock.unlock();
    if (!pending_pages.empty()) {
        MarkWritebackPending(pending_pages, f.serial);
        writebacks.push_back({f.serial, writeback_cmd, std::move(pending_pages)});
    }
    f.cmd = batch.cmd; f.timer = batch.timer;
    f.after = std::move(batch.after); f.pools = std::move(batch.pools);
    WatchGpuWrites(batch.pending_writeback, &f.watched);
    f.writes = std::move(batch.pending_writes); f.reports = std::move(batch.pending_reports);
    batch.cmd = VK_NULL_HANDLE; batch.timer = VK_NULL_HANDLE;
    batch.after.clear(); batch.pools.clear(); batch.pool_in_use = 0;
    batch.pending_writes.clear(); batch.pending_reports.clear(); batch.pending_writeback.clear();
    in_flight.push_back(std::move(f));
    ++submitted_serial;
    submitted_atomic.store(submitted_serial, std::memory_order_release);
    StartRetirer();
    retire_cv.notify_one();
    if (GpuProfiling()) { while (RetireOne(true)) {} return; } // per-item timings share one query pool
    const u64 before = batch.wait_ns;
    while (in_flight.size() > MaxInFlight) RetireOne(true);
    TraceWait(reinterpret_cast<void*>(1), batch.wait_ns - before); // (the in-flight limit)
    while (RetireOne(false)) {}
}

u64 SubmittedBatches() {
    std::lock_guard lock{BatchMutex()};
    return submitted_serial;
}
void WaitForBatches(u64 count) {
    std::lock_guard lock{BatchMutex()};
    const u64 before = batch.wait_ns;
    while (retired_serial < count && RetireOne(true)) {}
    TraceWait(reinterpret_cast<void*>(2), batch.wait_ns - before); // (frame pacing)
}

// Retires batches as the GPU completes them (OOT_NO_RETIRE_THREAD=1: only when a game thread polls or waits).
// The game polls GPU-signalled events and syncs with short sleeps; their values land when the batch writing
// them is retired, which otherwise happened only on such a poll, late.
namespace {
void RetireLoop() {
    rt::SetThreadName("nvn-retire");
    auto& vk = Vk();
    for (u64 done = 0;;) {
        const u64 target = submitted_atomic.load(std::memory_order_acquire);
        if (target <= done) {
            std::unique_lock lock{retire_mutex};
            retire_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return submitted_atomic.load() > done; });
            continue;
        }
        const VkSemaphore timeline = Timeline();
        const u64 next = done + 1;
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1; wi.pSemaphores = &timeline; wi.pValues = &next;
        if (vkWaitSemaphores(vk.device, &wi, 20'000'000) != VK_SUCCESS) continue;
        VK_CHECK(vkGetSemaphoreCounterValue(vk.device, timeline, &done));
        std::lock_guard lock{BatchMutex()};
        while (RetireOne(false)) {}
        ReapWritebacks();
    }
}
void StartRetirer() {
    static const bool off = std::getenv("OOT_NO_RETIRE_THREAD") != nullptr;
    static std::once_flag once;
    if (!off) std::call_once(once, [] { std::thread(RetireLoop).detach(); });
}
} // namespace

void Poll() {
    // (event polls come often: while the executor records, nothing is retired from here)
    std::unique_lock lock{BatchMutex(), std::try_to_lock};
    if (!lock.owns_lock()) return;
    while (RetireOne(false)) {}
    ReapWritebacks();
}

void WaitUntil(const std::function<bool()>& done) {
    std::lock_guard lock{BatchMutex()};
    const u64 before = batch.wait_ns;
    struct Trace { u64 before; void* caller; ~Trace() { TraceWait(caller, batch.wait_ns - before); } } trace{before, __builtin_return_address(0)};
    while (!done()) {
        if (!in_flight.empty()) RetireOne(true);
        else if (batch.cmd) Submit();
        else break;
    }
}


void Flush() {
    std::lock_guard lock{BatchMutex()};
    const u64 before = batch.wait_ns;
    Submit();
    while (RetireOne(true)) {}
    TraceWait(__builtin_return_address(0), batch.wait_ns - before);
    // Actions deferred while no batch was open (AfterBatch) run now.
    auto actions = std::move(batch.after);
    batch.after.clear();
    for (auto& action : actions) action();
}

namespace {
void AddRange(std::map<u64, u64>& ranges, u64 start, u64 end) {
    auto it = ranges.upper_bound(start);
    if (it != ranges.begin()) {
        auto prev = std::prev(it);
        if (prev->second >= start) { start = prev->first; end = std::max(end, prev->second); it = ranges.erase(prev); }
    }
    while (it != ranges.end() && it->first <= end) { end = std::max(end, it->second); it = ranges.erase(it); }
    ranges[start] = end;
}
} // namespace
void NoteWriteBack(u64 address, u64 size) {
    if (!size) return;
    std::lock_guard lock{BatchMutex()};
    AddRange(batch.pending_writeback, address, address + size);
}
// ---- GPU storage writes copied back when read ---------------------------------------------------
// Storage results reach guest memory only when something reads them: the game invalidates the range (its
// readback), or the runtime looks at it (HasPendingWrite guards those reads). Until then they stay in the
// pool's device copy (OoT writes ~78 MiB a frame on the GPU and reads a few words of it). Pages the CPU read
// once are copied back eagerly from then on; the rest stays inaccessible to the CPU (a touch copies that page
// back first). OOT_EAGER_WRITEBACK=1: every batch copies all back (as before).
namespace {
std::map<u64, u64> device_ahead;           // binding start -> end: GPU-written, not yet (all) in guest memory
std::set<u64> cpu_pages;                   // pages of GPU results the CPU reads (copied back every time; by address:
                                           // the game binds the same memory at other offsets frame to frame)
u64 deferred_bytes = 0, recalled_bytes = 0, eager_bytes = 0;
bool DeferWriteBack() { static const bool on = std::getenv("OOT_EAGER_WRITEBACK") == nullptr; return on; }
constexpr u64 Page = 4096;
// (batch lock held) [address, address + size) of GPU results still on the GPU comes back with the open batch; those
// pages are copied back eagerly from then on (OoT reads a few words at the start of buffers of megabytes).
bool RecallDeviceAhead(u64 address, u64 size) {
    if (device_ahead.empty() || !size) return false;
    bool any = false;
    // (whole pages, and every binding on them: a page becomes accessible as a whole)
    const u64 first = address & ~(Page - 1), last = (address + size + Page - 1) & ~(Page - 1);
    auto it = device_ahead.lower_bound(last);
    while (it != device_ahead.begin()) {
        --it;
        const u64 start = it->first, end = it->second;
        if (end <= first) { if (start + (64ull << 20) < first) break; continue; } // (bindings are at most 64 MiB)
        const u64 lo = std::max(start, first), hi = std::min(end, last);
        if (lo >= hi) continue;
        // (pages already known go back with the batches writing them: waiting for those is enough)
        for (u64 page = lo & ~(Page - 1); page < hi; page += Page) {
            if (cpu_pages.count(page)) continue;
            const u64 a = std::max(page, lo), b = std::min(page + Page, hi);
            if (!batch.cmd) BeginWork();
            AddRange(batch.pending_writes, a, b);
            AddRange(batch.pending_writeback, a, b);
            cpu_pages.insert(page);
            recalled_bytes += b - a;
            SettleDeviceAhead(a, b);
            any = true;
        }
        static int shown = 0;
        if (any && shown++ < 24) RT_LOG("GPU results %#llx-%#llx (of %#llx+%#llx) are read on the CPU: those pages copied back from now on",
                                        (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)start, (unsigned long long)(end - start));
    }
    return any;
}
} // namespace
void DropDeviceAhead(u64 address, u64 size) { // (a pool goes: its results are not wanted)
    std::lock_guard lock{BatchMutex()};
    for (auto it = device_ahead.lower_bound(address); it != device_ahead.end() && it->first < address + size;) {
        SettleDeviceAhead(it->first, it->second);
        it = device_ahead.erase(it);
    }
}
void ReportDeviceAhead() {
    std::lock_guard lock{BatchMutex()};
    if (!DeferWriteBack()) return;
    const size_t pages = cpu_pages.size();
    RT_LOG("GPU results kept on the GPU: %.2f MiB per frame; copied back: %.2f MiB of pages the CPU reads, %.2f MiB on first reads (%zu pages known)",
           (deferred_bytes - eager_bytes) / 300.0 / 1048576.0, eager_bytes / 300.0 / 1048576.0, recalled_bytes / 300.0 / 1048576.0, pages);
    deferred_bytes = recalled_bytes = eager_bytes = 0;
}
void NotePendingWrite(u64 address, u64 size, bool write_back) {
    if (!size) return;
    static const u64 watch = std::getenv("OOT_WRITE_WATCH") ? std::strtoull(std::getenv("OOT_WRITE_WATCH"), nullptr, 16) : 0;
    if (watch && address <= watch && watch < address + size) {
        static int shown = 0;
        if (shown++ < 20) RT_LOG("pending write %#llx+%#llx covers watched address (caller %p)", (unsigned long long)address, (unsigned long long)size, __builtin_return_address(0));
    }
    std::lock_guard lock{BatchMutex()};
    AddRange(batch.pending_writes, address, address + size);
    if (write_back && DeferWriteBack()) {
        u64& end = device_ahead[address];
        if (end < address + size) { end = address + size; ProtectDeviceAhead(address, end, &cpu_pages); }
        deferred_bytes += size;
        for (auto page = cpu_pages.lower_bound(address & ~(Page - 1)); page != cpu_pages.end() && *page < address + size; ++page) {
            const u64 lo = std::max(*page, address), hi = std::min(*page + Page, address + size); // (pages the CPU reads: back with this batch)
            if (lo < hi) { AddRange(batch.pending_writeback, lo, hi); eager_bytes += hi - lo; }
        }
    } else if (write_back) AddRange(batch.pending_writeback, address, address + size);
}

namespace {
bool Overlaps(const std::map<u64, u64>& ranges, u64 address, u64 size) {
    auto it = ranges.upper_bound(address + size - 1);
    if (it == ranges.begin()) return false;
    --it;
    return it->second > address;
}
} // namespace
bool PendingWriteRange(u64 address, u64 size, u64* start, u64* end) {
    std::lock_guard lock{BatchMutex()};
    RecallDeviceAhead(address, size);
    const auto find = [&](const std::map<u64, u64>& ranges) {
        auto it = ranges.upper_bound(address + size - 1);
        if (it == ranges.begin()) return false;
        --it; if (it->second <= address) return false;
        *start = it->first; *end = it->second; return true; };
    if (find(batch.pending_writes)) return true;
    for (const auto& f : in_flight) if (find(f.writes)) return true;
    return false;
}
// Before the CPU reads GPU results there (an invalidate): the GPU work that writes the range, and its copy
// back to guest memory, complete first. On the console's coherent memory the results are there once that work
// is; here they arrive with the write-back at the end of the batch. (OoT's auto-exposure sums brightness on
// the GPU and the CPU reads and clears the sum: read early, it stepped up and down, the white flashes.)
// OOT_NO_READBACK_WAIT=1: not.
std::atomic<u64> cpu_wait_ns[3];
void WaitForWrites(u64 address, u64 size) {
    static const bool off = std::getenv("OOT_NO_READBACK_WAIT") != nullptr;
    if (off || !size) return;
    struct Timed { std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timed() { cpu_wait_ns[1] += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); } } timed;
    static const bool trace = std::getenv("OOT_INVALIDATE_TRACE") != nullptr;
    if (trace) { static std::map<std::pair<u64,u64>, u64> seen; static u64 n = 0; ++seen[{address, size}];
        if (++n % 3000 == 0) { for (auto& [k, c] : seen) RT_LOG("invalidate %#llx+%#llx: %llu", (unsigned long long)k.first, (unsigned long long)k.second, (unsigned long long)c); seen.clear(); } }
    std::lock_guard lock{BatchMutex()};
    RecallDeviceAhead(address, size);
    static const bool trace_waits = std::getenv("OOT_READBACK_TRACE") != nullptr;
    const auto show = [&](const char* what, const std::map<u64, u64>& ranges) {
        if (!trace_waits) return;
        static std::atomic<u32> shown{0}; if (shown++ >= 20000) return;
        auto it = ranges.upper_bound(address + size - 1); if (it == ranges.begin()) return; --it;
        RT_LOG("readback %#llx+%#llx waits for %s: pending %#llx-%#llx", (unsigned long long)address, (unsigned long long)size, what,
               (unsigned long long)it->first, (unsigned long long)it->second);
    };
    if (Overlaps(batch.pending_writes, address, size)) show("the open batch's writes", batch.pending_writes);
    else if (Overlaps(batch.pending_reports, address, size)) show("the open batch's reports", batch.pending_reports);
    if (Overlaps(batch.pending_writes, address, size) || Overlaps(batch.pending_reports, address, size)) Submit();
    for (;;) {
        size_t last = in_flight.size();
        for (size_t i = in_flight.size(); i-- > 0;)
            if (Overlaps(in_flight[i].writes, address, size) || Overlaps(in_flight[i].reports, address, size)) { last = i; break; }
        if (last == in_flight.size()) break;
        if (trace_waits) show("a submitted batch", Overlaps(in_flight[last].writes, address, size) ? in_flight[last].writes : in_flight[last].reports);
        RetireOne(true); // (oldest first, up to the last batch writing the range)
    }
}
bool HasPendingWrite(u64 address, u64 size) {
    std::lock_guard lock{BatchMutex()};
    RecallDeviceAhead(address, size); // (the runtime reads it: its results come back with the open batch)
    if (Overlaps(batch.pending_writes, address, size)) return true;
    for (const auto& f : in_flight) if (Overlaps(f.writes, address, size)) return true;
    return false;
}

void NotePendingReport(u64 address) {
    std::lock_guard lock{BatchMutex()};
    batch.pending_reports[address] = address + 16;
}

bool HasPendingReport(u64 address, u64 size) {
    std::lock_guard lock{BatchMutex()};
    if (Overlaps(batch.pending_reports, address, size)) return true;
    for (const auto& f : in_flight) if (Overlaps(f.reports, address, size)) return true;
    return false;
}

void AfterFlush(std::function<void()> action) {
    std::lock_guard lock{BatchMutex()};
    if (batch.cmd) batch.after.push_back(std::move(action));
    else if (!in_flight.empty()) in_flight.back().after.push_back(std::move(action)); // after submitted work
    else action(); // nothing pending
}
void AfterBatch(std::function<void()> action) {
    std::lock_guard lock{BatchMutex()};
    batch.after.push_back(std::move(action));
}

VkDescriptorSet AllocateBatchSet(VkDescriptorSetLayout layout) {
    std::lock_guard lock{BatchMutex()};
    auto& vk = Vk();
    for (;;) {
        if (batch.pool_in_use == batch.pools.size()) {
            if (!spare_pools.empty()) { batch.pools.push_back(spare_pools.back()); spare_pools.pop_back(); }
            else batch.pools.push_back(MakePool());
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = batch.pools[batch.pool_in_use];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set;
        const VkResult result = vkAllocateDescriptorSets(vk.device, &ai, &set);
        if (result == VK_SUCCESS) return set;
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) VK_CHECK(result);
        ++batch.pool_in_use; // this pool is full; continue with the next
    }
}

} // namespace nvn
