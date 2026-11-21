#ifndef _WIN32
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#endif
#include <cstring>
#include <filesystem>
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
extern std::atomic<u64> passes_begun; // (present.cpp)
namespace {

struct Batch {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> earlier; // (SplitForHoist) ended parts and hoist buffers, in execution order before cmd
    VkCommandBuffer hoist = VK_NULL_HANDLE; // the last of `earlier` while it still takes commands
    bool hoist_live = false, hoist_pass_begun = false; // (HoistBuffer: nothing but one pass's draws since the split)
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
    std::vector<VkCommandBuffer> earlier;
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
// The batches' command buffers have a command pool of their own. A Vulkan command pool takes one thread at a time, for
// recording into any of its buffers as much as for allocating and freeing them: the recorder thread records a pass's
// draws while other threads retire finished batches and the presenting thread records its frame. So this pool is
// touched only where the recorder is idle (no batch open, or its queue drained under the batch lock), and buffers of
// finished batches wait in `dead_commands` until such a point (FreeDeadCommands) instead of being freed by whichever
// thread retires them. (Sharing one pool went unnoticed on NVIDIA; Vulkan's validation layer reports it, and a driver
// that keeps per-pool lists unlocked, as Mesa does, can lose command data to it.)
VkCommandPool BatchPool() {
    static VkCommandPool pool = [] {
        auto& vk = Vk();
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; ci.queueFamilyIndex = vk.queue_family;
        VkCommandPool made = VK_NULL_HANDLE; VK_CHECK(vkCreateCommandPool(vk.device, &ci, nullptr, &made));
        return made;
    }();
    return pool;
}
std::mutex dead_commands_mutex;
std::vector<VkCommandBuffer> dead_commands;
void DeadCommand(VkCommandBuffer cmd) { if (cmd) { std::lock_guard lock{dead_commands_mutex}; dead_commands.push_back(cmd); } }
void FreeDeadCommands() { // (batch lock held, the recorder idle)
    std::vector<VkCommandBuffer> dead;
    { std::lock_guard lock{dead_commands_mutex}; dead.swap(dead_commands); }
    if (!dead.empty()) vkFreeCommandBuffers(Vk().device, BatchPool(), (u32)dead.size(), dead.data());
}

void ReapWritebacks() {
    if (writebacks.empty()) return;
    auto& vk = Vk();
    u64 done = 0;
    VK_CHECK(vkGetSemaphoreCounterValue(vk.device, WritebackTimeline(), &done));
    while (!writebacks.empty() && writebacks.front().serial <= done) {
        Writeback& w = writebacks.front();
        WritebackDone(w.pages, w.serial);
        DeadCommand(w.cmd);
        writebacks.pop_front();
    }
}
std::vector<VkFence> spare_fences;
std::vector<VkQueryPool> spare_timers;
std::vector<VkDescriptorPool> spare_pools;
u64 submitted_serial = 0, retired_serial = 0;
std::atomic<u64> retired_serial_atomic{0};
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
// OOT_GPU_PROFILE=1: from the start. OOT_GPU_PROFILE_FILE=<path>: while that file exists (checked every 30 frames:
// started in play, as a start in this mode no longer reaches the title).
std::atomic<bool> gpu_profiling{std::getenv("OOT_GPU_PROFILE")!=nullptr};
bool GpuProfiling() { return gpu_profiling.load(std::memory_order_relaxed); }
}
bool GpuProfilingNow() { return GpuProfiling(); }
namespace {

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
void BatchLock::Yield() {
    static const bool off = std::getenv("OOT_NO_BATCH_YIELD") != nullptr;
    if (off || depth != 1 || !waiters.load(std::memory_order_relaxed)) return;
    unlock();
    // (until each waiter has the lock or had it: a futex wake takes a few microseconds; bounded, in case one went away)
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    while (waiters.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < until) std::this_thread::yield();
    lock();
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
u64 stat_passes_unbarriered = 0, stat_passes_relaxed = 0;
// Draw passes only since the last full barrier (nothing else recorded: no copy, clear, dispatch or upload). A new pass then
// starts with a barrier that waits only for fragment work, not everything (RelaxedPassBarriers). draw_begin_pending: the
// full barrier being made is for a draw pass.
bool only_draws_since_full = false, draw_begin_pending = false;
// OOT_RELAXED_PASS_BARRIERS=1: passes that follow only draw passes start with the fragment-stage barrier.
bool RelaxedPassBarriers() { static const bool on = [] { const char* e = std::getenv("OOT_RELAXED_PASS_BARRIERS"); return e && std::atoi(e) != 0; }(); return on; }

// ---- clears in the pass that follows (OOT_PASS_CLEARS=1) ----
// A clear of a draw target was its own piece of work: the open pass ended, a full barrier, the clear, and a new pass with
// another full barrier for the draws into it. Held instead, it becomes the first command of the next pass when that pass
// draws into the cleared view at its size (as the other renderer of this game does); in every other case - other work,
// a submission, the pass already open, a different target - it is recorded as before, ahead of that.
struct HeldClear { VkImageView view; VkImageAspectFlags aspect; VkClearValue value; VkExtent2D extent; std::function<void(VkCommandBuffer)> record; };
std::vector<HeldClear> held_clears; std::vector<u64> held_key; bool flushing_held = false; u64 stat_clears_in_pass = 0, stat_clears_held = 0;
bool PassClears() { static const bool on = [] { const char* e = std::getenv("OOT_PASS_CLEARS"); return e && std::atoi(e) != 0 && !std::getenv("OOT_PASS_HAZARDS"); }(); return on; }
} // namespace
// ---- the recorder thread (PassDraw's async draws) ----
namespace {
struct AsyncItem { AsyncDraw run; void* packet; VkCommandBuffer cmd; u64 serial; };
struct AsyncQueueState {
    AsyncItem items[AsyncDrawQueue];
    alignas(64) std::atomic<u64> head{0};   // queued (written by the thread holding the batch lock)
    alignas(64) std::atomic<u64> tail{0};   // recorded (written by the recorder thread)
    alignas(64) std::atomic<bool> sleeping{false};
    std::atomic<bool> kick{false};          // (a sleeping recorder is asked to record what waits, however little)
    std::mutex mutex; std::condition_variable wake;
    std::once_flag started;
} async_queue;
void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}
// How the recorder waits for draws. With cores to spare it spins through the gaps between them (microseconds) and is
// woken for the first draw after a sleep. On a CPU of 8 threads or fewer (a Steam Deck: 4 cores, and a power budget
// the GPU shares) the spinning was half a hardware thread of nothing, 13 of the thread's 16 percent of the process's
// samples: there it sleeps at once and is woken when a batch of draws waits or somebody needs them recorded.
// OOT_RECORD_SPIN=<iterations>, OOT_RECORD_BATCH=<draws>.
int RecorderSpin() {
    static const int n = [] { const char* e = std::getenv("OOT_RECORD_SPIN"); return e ? std::atoi(e) : std::thread::hardware_concurrency() <= 8 ? 64 : 4000; }();
    return n;
}
u64 RecorderBatch() {
    static const u64 n = [] { const char* e = std::getenv("OOT_RECORD_BATCH"); return (u64)std::max(1, e ? std::atoi(e) : std::thread::hardware_concurrency() <= 8 ? 12 : 1); }();
    return n;
}
void KickRecorder() {
    auto& q = async_queue;
    std::lock_guard lock{q.mutex}; q.kick.store(true, std::memory_order_seq_cst); q.wake.notify_one();
}
void RecorderLoop() {
    rt::SetThreadName("nvn-record");
    auto& q = async_queue;
    for (;;) {
        const u64 t = q.tail.load(std::memory_order_relaxed);
        if (t == q.head.load(std::memory_order_acquire)) {
            // (draws come in runs, microseconds apart: spin through the gaps, sleep when the run is over)
            bool found = false;
            for (int spin = 0; spin < RecorderSpin() && !found; ++spin) { CpuRelax(); found = t != q.head.load(std::memory_order_acquire); }
            if (!found) {
                std::unique_lock lock{q.mutex};
                q.sleeping.store(true, std::memory_order_seq_cst);
                q.wake.wait(lock, [&] { return q.kick.load(std::memory_order_seq_cst) || q.head.load(std::memory_order_seq_cst) - t >= RecorderBatch(); });
                q.kick.store(false, std::memory_order_seq_cst);
                q.sleeping.store(false, std::memory_order_seq_cst);
            }
            continue;
        }
        const AsyncItem& item = q.items[t % AsyncDrawQueue];
        item.run(item.cmd, item.packet, item.serial);
        // A driver may record straight into write-combined GPU memory (Mesa's RADV does: its command stream). Such
        // stores sit in this core's write-combining buffers until something evicts them, and nothing in this loop
        // did: the thread that submits the buffer, on another core, handed the GPU a command stream whose last
        // lines were not there yet. Steam Deck: command-processor and index-fetch page faults and ring timeouts
        // within seconds of the title in every run with this thread, none with its path run on the executor's
        // thread (2026-11-17). The fence evicts them before the item counts as recorded.
#if defined(__x86_64__) || defined(__i386__)
        static const bool fence = std::getenv("OOT_RECORD_NO_FENCE") == nullptr;
        if (fence) _mm_sfence();
#endif
        q.tail.store(t + 1, std::memory_order_release);
    }
}
// (batch lock held) Every queued draw is in the command buffer when this returns.
inline void DrainAsync() {
    auto& q = async_queue;
    const u64 h = q.head.load(std::memory_order_relaxed);
    while (q.tail.load(std::memory_order_acquire) != h) {
        if (q.sleeping.load(std::memory_order_seq_cst)) KickRecorder(); // (it may sleep on less than a batch)
        CpuRelax();
    }
}
void PushAsync(const AsyncItem& item) {
    auto& q = async_queue;
    // (OOT_ASYNC_INLINE=1: the recorder's path, on this thread; =0: the recorder thread. On a CPU of 8 threads or fewer -
    // a Steam Deck - that is the default: there the executor stood a seventh of its time waiting for the recorder at
    // the points that need its draws recorded (every mesh draw of the compute fallback is one), which cost what the
    // thread saved: Castle Town 35.6 ms a frame of executor time inline against 37.2 with the thread, 2026-11-17.)
    static const bool inline_mode = [] { const char* e = std::getenv("OOT_ASYNC_INLINE"); return e ? std::atoi(e) != 0 : std::thread::hardware_concurrency() <= 8; }();
    if (inline_mode) { item.run(item.cmd, item.packet, item.serial); return; }
    std::call_once(q.started, [] { std::thread(RecorderLoop).detach(); });
    const u64 h = q.head.load(std::memory_order_relaxed);
    while (h - q.tail.load(std::memory_order_acquire) >= AsyncDrawQueue) { if (q.sleeping.load(std::memory_order_seq_cst)) KickRecorder(); CpuRelax(); }
    q.items[h % AsyncDrawQueue] = item;
    q.head.store(h + 1, std::memory_order_seq_cst);
    if (q.sleeping.load(std::memory_order_seq_cst) && h + 1 - q.tail.load(std::memory_order_acquire) >= RecorderBatch()) { std::lock_guard lock{q.mutex}; q.wake.notify_one(); }
    // (OOT_ASYNC_SYNC=1, diagnosis: the recorder thread records, and this thread waits for it - the other thread
    // without the concurrency)
    static const bool sync = std::getenv("OOT_ASYNC_SYNC") != nullptr;
    if (sync) while (q.tail.load(std::memory_order_acquire) != h + 1) { if (q.sleeping.load(std::memory_order_seq_cst)) KickRecorder(); CpuRelax(); }
}
} // namespace
void WaitAsyncDraws() { std::lock_guard lock{BatchMutex()}; DrainAsync(); }
bool AsyncDrawsEnabled() {
    static const bool on = [] { const char* e = std::getenv("OOT_ASYNC_DRAWS"); if (e) return std::atoi(e) != 0;
        return std::thread::hardware_concurrency() >= 4; }();
    return on;
}
// OOT_BATCH_TRACE=<file> (diagnosis of a GPU that stops answering): a line for every batch when it is submitted, naming
// what was recorded into it in order (i<key>: an item and its profile key, a compute program's or a shader's address, or
// r<offset>: the caller's return address in the executable when it has none; p<key>: a render pass's first draw; l: light
// work; h: a split for hoisted work), and a line when it has finished. The last batches submitted and not finished are
// the ones the GPU was working on. Written with plain write(): the lines survive the game being killed.
#ifndef _WIN32
int batch_trace_fd = [] { const char* f = std::getenv("OOT_BATCH_TRACE"); return f ? open(f, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644) : -1; }();
#else
int batch_trace_fd = -1;
#endif
std::string batch_trace_items;
void TraceItem(char kind, u64 key) {
    if (batch_trace_fd < 0 || batch_trace_items.size() > 60000) return;
    char t[32]; std::snprintf(t, sizeof t, " %c%llx", kind, (unsigned long long)key); batch_trace_items += t;
}
#ifndef _WIN32
void TraceLine(const std::string& line) { if (batch_trace_fd >= 0) { [[maybe_unused]] auto n = write(batch_trace_fd, line.data(), line.size()); } }
u64 TraceCaller(void* address) { Dl_info info{}; dladdr(reinterpret_cast<void*>(&BatchMutex), &info); return reinterpret_cast<u64>(address) - reinterpret_cast<u64>(info.dli_fbase); }
#else
void TraceLine(const std::string&) {}
u64 TraceCaller(void*) { return 0; }
#endif

bool BatchTraceOn() { return batch_trace_fd >= 0; }
void BatchTraceNote(const char* text) { if (batch_trace_fd >= 0 && batch_trace_items.size() < 60000) { std::lock_guard lock{BatchMutex()}; batch_trace_items += ' '; batch_trace_items += text; } }

void EndPassLocked() {
    if (!batch.pass_open) return;
    DrainAsync();
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
    DrainAsync();
    record(batch.cmd);
    return true;
}

bool RecordInPassAsync(AsyncDraw run, void* packet) {
    std::lock_guard lock{BatchMutex()};
    if (!batch.cmd || !batch.pass_open) return false;
    if (AsyncDrawsEnabled()) PushAsync({run, packet, batch.cmd, other_serial});
    else run(batch.cmd, packet, other_serial);
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

void FlushHeldClears(); // (batch lock held)
// Opens the batch command buffer, or orders the next work after everything recorded so far.
void BeginWork() {
    auto& vk = Vk();
    if (!held_clears.empty() && !draw_begin_pending) FlushHeldClears(); // (a draw pass's own: its begin takes them)
    ++record_serial;
    ++other_serial;
    EndPassLocked();
    only_draws_since_full = draw_begin_pending; draw_begin_pending = false; // (a full barrier follows: what comes after it decides)
    if (!batch.cmd) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        FreeDeadCommands(); // (no batch is open: nothing records)
        ai.commandPool = BatchPool();
        ai.commandBufferCount = 1;
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

void FlushHeldClears() {
    if (flushing_held) return;
    flushing_held = true;
    auto held = std::move(held_clears); held_clears.clear();
    for (auto& c : held) Immediate(c.record);
    flushing_held = false;
}
bool HoldClear(VkImageView view, VkImageAspectFlags aspect, const VkClearValue& value, VkExtent2D extent, std::function<void(VkCommandBuffer)> record) {
    if (!PassClears() || !view || GpuProfiling()) return false;
    std::lock_guard lock{BatchMutex()};
    if (!batch.cmd) return false;
    for (const auto& c : held_clears) if (c.view == view) { FlushHeldClears(); break; } // (a second clear of it: in order)
    EndPassLocked(); // (no draw of the open pass may come after the clear unseen)
    held_clears.push_back({view, aspect, value, extent, std::move(record)});
    ++stat_clears_held;
    return true;
}
void PrepareHeldClears(const std::vector<u64>& key, const VkImageView* colors, u32 count, VkImageView depth, bool has_depth, bool has_stencil, VkExtent2D extent) {
    std::lock_guard lock{BatchMutex()};
    if (held_clears.empty()) return;
    held_key = key;
    // (the pass still open: these clears come after its draws - recorded as before, which ends it)
    if (batch.cmd && batch.pass_open && batch.pass_key == key) { FlushHeldClears(); return; }
    const auto fits = [&](const HeldClear& c) {
        if (c.extent.width != extent.width || c.extent.height != extent.height) return false;
        if (c.aspect == VK_IMAGE_ASPECT_COLOR_BIT) { for (u32 i = 0; i < count; ++i) if (colors[i] == c.view) return true; return false; }
        return c.view == depth && (!(c.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || has_depth) && (!(c.aspect & VK_IMAGE_ASPECT_STENCIL_BIT) || has_stencil);
    };
    bool all = true; for (const auto& c : held_clears) all = all && fits(c);
    if (!all) FlushHeldClears(); // (keep it simple: any that does not fit, all in order as before)
}
void ApplyHeldClears(VkCommandBuffer cmd, const VkImageView* colors, u32 count, VkExtent2D extent) {
    // (batch lock held: inside PassDraw's begin)
    if (held_clears.empty() || flushing_held) return;
    for (const auto& c : held_clears) {
        VkClearAttachment a{}; a.aspectMask = c.aspect; a.clearValue = c.value;
        if (c.aspect == VK_IMAGE_ASPECT_COLOR_BIT) { for (u32 i = 0; i < count; ++i) if (colors[i] == c.view) a.colorAttachment = i; }
        const VkClearRect rect{{{0, 0}, extent}, 0, 1};
        vkCmdClearAttachments(cmd, 1, &a, 1, &rect);
        ++stat_clears_in_pass;
    }
    held_clears.clear();
}

bool PassOpenWithKey(const std::vector<u64>& key) {
    if (resolve_worker) return true; // (its draws go into one pass per key when they are committed)
    std::lock_guard lock{BatchMutex()};
    return batch.cmd && batch.pass_open && batch.pass_key == key && !GpuProfiling();
}

void NoteUnorderedWork() { std::lock_guard lock{BatchMutex()}; unsynced.unknown = true; }

void PassDraw(const std::vector<u64>& key, const std::function<void(VkCommandBuffer)>& begin,
              const std::function<void(VkCommandBuffer)>& draw, u64 profile_key, const PassAccess* access,
              AsyncDraw async, void* packet) {
    std::lock_guard lock{BatchMutex()};
    if (GpuProfiling()) { // per-item timing needs separate items
        Immediate([&](VkCommandBuffer cmd) {
            begin(cmd); draw(cmd);
            if (batch.pass_end_hook) { auto hook = std::move(batch.pass_end_hook); batch.pass_end_hook = nullptr; hook(cmd); }
            vkCmdEndRendering(cmd); }, profile_key);
        return;
    }
    if (!held_clears.empty() && (held_key != key || (batch.pass_open && batch.pass_key == key))) FlushHeldClears(); // (not prepared for this pass)
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
            } else if (batch.cmd && only_draws_since_full && RelaxedPassBarriers()) {
                // (only draw passes since the last full barrier: their attachment and storage writes ordered before this
                // pass's fragment work - the barrier the other renderer of this game uses there)
                EndPassLocked(); ++record_serial; ++other_serial; ++stat_passes_relaxed;
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                vkCmdPipelineBarrier(batch.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                batch.light_only = false;
            } else { work_kind = 2; draw_begin_pending = true; BeginWork(); }
        }
        if (batch.hoist_live) { if (batch.hoist_pass_begun) batch.hoist_live = false; else batch.hoist_pass_begun = true; } // (the split's own pass, once)
        if (batch_trace_fd >= 0) TraceItem('p', profile_key);
        begin(batch.cmd);
        ++stat_passes; passes_begun.fetch_add(1, std::memory_order_relaxed); // (present.cpp: a loading screen draws next to nothing)
        reopen_light = false;
        batch.pass_open = true;
        batch.pass_key = key;
    }
    if (async) PushAsync({async, packet, batch.cmd, other_serial});
    else { DrainAsync(); draw(batch.cmd); }
    ++record_serial;
    static const bool tracking = std::getenv("OOT_PASS_HAZARDS") != nullptr;
    if (tracking) { if (access) unsynced.Add(*access); else unsynced.unknown = true; }
}

u64 RecordSerial() { std::lock_guard lock{BatchMutex()}; return record_serial; }
u64 OtherWorkSerial() { return other_serial; } // (batch lock held by the caller's recording)

bool ImmediateUnordered(const std::function<void(VkCommandBuffer)>& record, u64 serial) {
    std::lock_guard lock{BatchMutex()};
    if (!held_clears.empty()) return false; // (ordered work, after the held clears)
    if (GpuProfiling() || !batch.cmd || batch.pass_open || serial != record_serial) return false;
    ++stat_items; ++stat_unordered;
    only_draws_since_full = false;
    batch.hoist_live = false;
    batch.light_only = false;
    unsynced.unknown = true;
    record(batch.cmd);
    ++record_serial;
    ++other_serial;
    return true;
}

void ImmediateLight(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard lock{BatchMutex()};
    if (!held_clears.empty()) FlushHeldClears();
    ++stat_items;
    if (GpuProfiling() || !batch.cmd) { Immediate(record); return; }
    const bool light = batch.pass_open || batch.light_only;
    only_draws_since_full = false;
    batch.hoist_live = false;
    if (batch_trace_fd >= 0) TraceItem('l', TraceCaller(__builtin_return_address(0)));
    EndPassLocked();
    unsynced.unknown = true;
    ++record_serial;
    ++other_serial;
    batch.light_only = light;
    record(batch.cmd);
}

VkCommandBuffer SplitForHoist() {
    std::lock_guard lock{BatchMutex()};
    auto& vk = Vk();
    if (!batch.cmd || batch.pass_open || GpuProfiling() || !held_clears.empty()) return VK_NULL_HANDLE;
    only_draws_since_full = false; // (a new command buffer: its first pass takes the full barrier)
    VkCommandBuffer made[2];
    {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = BatchPool(); ai.commandBufferCount = 2;
        VK_CHECK(vkAllocateCommandBuffers(vk.device, &ai, made));
    }
    if (batch.hoist) VK_CHECK(vkEndCommandBuffer(batch.hoist));
    VK_CHECK(vkEndCommandBuffer(batch.cmd));
    batch.earlier.push_back(batch.cmd);
    batch.earlier.push_back(made[0]);
    batch.hoist = made[0]; batch.cmd = made[1];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(batch.hoist, &bi));
    VK_CHECK(vkBeginCommandBuffer(batch.cmd, &bi));
    ++record_serial; ++other_serial; // (a new command buffer: nothing a draw set on the old one is in effect)
    if (batch_trace_fd >= 0) TraceItem('h', 0);
    batch.hoist_live = true; batch.hoist_pass_begun = false;
    return batch.hoist;
}

VkCommandBuffer HoistBuffer() {
    std::lock_guard lock{BatchMutex()};
    if (!batch.hoist_live || !batch.hoist || GpuProfiling()) return VK_NULL_HANDLE;
    DrainAsync(); // (the recorder thread records the pass's draws into a buffer of the same command pool: one at a time)
    return batch.hoist;
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
    if (batch_trace_fd >= 0) { if (profile_key) TraceItem('i', profile_key); else TraceItem('r', TraceCaller(__builtin_return_address(0))); }
    batch.hoist_live = false;
    BeginWork();
    {
    const bool profile=GpuProfiling()&&gpu_profile.keys.size()*2<GpuProfile::Capacity;
    const u32 query=gpu_profile.keys.size()*2;
    if(profile){
        if(!gpu_profile.pool){VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qi.queryType=VK_QUERY_TYPE_TIMESTAMP;qi.queryCount=GpuProfile::Capacity;
            VK_CHECK(vkCreateQueryPool(vk.device,&qi,nullptr,&gpu_profile.pool));}
        vkCmdResetQueryPool(batch.cmd,gpu_profile.pool,query,2);
        vkCmdWriteTimestamp(batch.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,gpu_profile.pool,query);
        // (work without a shader: by its caller, as an offset in the executable below 2^32: addr2line -f -e tootr <offset>)
        if(!profile_key){Dl_info di{};dladdr(reinterpret_cast<void*>(&BatchMutex),&di);profile_key=(u64)((char*)__builtin_return_address(0)-(char*)di.dli_fbase);}
        gpu_profile.keys.push_back(profile_key);
    }
    record(batch.cmd);
    if(profile)vkCmdWriteTimestamp(batch.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,gpu_profile.pool,query+1);
    }
}

void PollGpuProfileSwitch() {
    static const char* file=std::getenv("OOT_GPU_PROFILE_FILE");
    if(!file)return;
    const bool on=std::filesystem::exists(file);
    if(on!=gpu_profiling.load()){Submit();gpu_profiling.store(on);RT_LOG("GPU profile %s",on?"on":"off");}
}
void ReportGpuProfile() {
    if(!GpuProfiling())return;
    std::vector<std::pair<double,u64>> top;
    for(const auto& [key,value]:gpu_profile.totals)top.emplace_back(value.first,key);
    std::sort(top.rbegin(),top.rend());
    {double sum=0;u64 calls=0;for(const auto& [k,v]:gpu_profile.totals){sum+=v.first;calls+=v.second;}
     RT_LOG("GPU profile total: %.3f ms/frame in %llu items over %zu shaders",sum/300e6,(unsigned long long)calls/300,gpu_profile.totals.size());}
    for(size_t i=0;i<std::min<size_t>(45,top.size());++i){const auto [ns,key]=top[i];
        RT_LOG("GPU profile shader %#llx: %.3f ms/frame, %llu calls",(unsigned long long)key,ns/300e6,(unsigned long long)gpu_profile.totals[key].second);}
    gpu_profile.totals.clear();
}

std::atomic<u64> flush_count{0};
u64 FlushCount() { return flush_count.load(); }
void ReportDeviceAhead();
void ReportBatchStats() {
    RT_LOG("batch structure per frame: %.1f render passes (%.1f reopened cheaply, %.1f with a fragment-stage barrier), %.1f full barriers (%.1f compute dispatches recorded without one), %.1f other items",
           stat_passes / 300.0, stat_reopens / 300.0, stat_passes_relaxed / 300.0, stat_full_barriers / 300.0, stat_unordered / 300.0, stat_items / 300.0);
    if (stat_clears_held) RT_LOG("clears held for the next pass: %.1f per frame, %.1f of them made in it", stat_clears_held / 300.0, stat_clears_in_pass / 300.0);
    stat_unordered = 0; stat_passes_relaxed = 0; stat_clears_held = stat_clears_in_pass = 0;
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
    if (wait && unlocked_waits && !GpuProfiling() && BatchMutex().depth == 1 && vkGetFenceStatus(vk.device, in_flight.front().fence) != VK_SUCCESS) {
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
        DeadCommand(f.cmd);
        for (VkCommandBuffer part : f.earlier) DeadCommand(part);
        DeadCommand(f.uploads);
    }
    UnwatchPages(f.watched);
    for (VkDescriptorPool pool : f.pools) { vkResetDescriptorPool(vk.device, pool, 0); spare_pools.push_back(pool); }
    auto actions = std::move(f.after);
    if (batch_trace_fd >= 0) TraceLine("done " + std::to_string(in_flight.front().serial) + "\n");
    in_flight.pop_front();
    ++retired_serial; retired_serial_atomic.store(retired_serial, std::memory_order_release);
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
    if (!held_clears.empty()) FlushHeldClears();
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
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = BatchPool(); ai.commandBufferCount = 1; // (the pass has ended: the recorder's queue is drained)
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
    if (batch_trace_fd >= 0) { TraceLine("submit " + std::to_string(submitted_serial + 1) + ":" + batch_trace_items + "\n"); batch_trace_items.clear(); } // (before the submit: a driver that waits in it may never return)
    if (batch.hoist) { VK_CHECK(vkEndCommandBuffer(batch.hoist)); batch.hoist = VK_NULL_HANDLE; }
    batch.hoist_live = false;
    if (!spare_fences.empty()) { f.fence = spare_fences.back(); spare_fences.pop_back(); }
    else { VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK_CHECK(vkCreateFence(vk.device, &fi, nullptr, &f.fence)); }
    {
        std::lock_guard queue{vk.queue_mutex};
        std::vector<VkCommandBuffer> buffers; // (uploads, the parts SplitForHoist ended and their hoist buffers, the last part)
        if (any_uploads) buffers.push_back(f.uploads);
        buffers.insert(buffers.end(), batch.earlier.begin(), batch.earlier.end());
        buffers.push_back(batch.cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        f.serial = submitted_serial + 1;
        const VkSemaphore timeline = Timeline();
        VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ti.signalSemaphoreValueCount = 1; ti.pSignalSemaphoreValues = &f.serial;
        si.pNext = &ti; si.signalSemaphoreCount = 1; si.pSignalSemaphores = &timeline;
        si.commandBufferCount = (u32)buffers.size();
        si.pCommandBuffers = buffers.data();
        {   // (time in vkQueueSubmit: on amdgpu it grows with the imported memory)
            const auto t0 = std::chrono::steady_clock::now();
            VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, f.fence));
            if (TimelineOn()) RT_LOG("T %.2f submit batch %llu", TimelineMs(), (unsigned long long)f.serial);
            static double total = 0; static u64 n = 0;
            total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (++n % 500 == 0) { RT_LOG("vkQueueSubmit: %.2f ms each (last 500)", total / 500); total = 0; }
        }
        if (!any_uploads) { DeadCommand(f.uploads); f.uploads = VK_NULL_HANDLE; }
        if (!pending_pages.empty()) { // (the write-back copy: after the batch, which completes without it)
            const VkSemaphore wb = WritebackTimeline();
            // On its own queue (the main queue's next submissions, such as the fence the game waits on,
            // do not queue behind it), after the batch (its timeline value).
            // The driver may expose two queues from this family, but a
            // timeline wait between them is not guaranteed to establish
            // execution order.  The copy queue can begin its writeback while
            // the batch still has access to the same guest allocation.  Keep
            // the copy on the main queue so the batch signal orders it even
            // when both queues share the same family.
            const bool own = false;
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
            DeadCommand(writeback_cmd);
        }
    }
    pools_lock.unlock();
    if (!pending_pages.empty()) {
        MarkWritebackPending(pending_pages, f.serial);
        writebacks.push_back({f.serial, writeback_cmd, std::move(pending_pages)});
    }
    f.cmd = batch.cmd; f.timer = batch.timer;
    f.earlier = std::move(batch.earlier); batch.earlier.clear();
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
        // (without queueing for the lock: the executor, holding it while it records, retires completed batches
        // itself at each submission, about every millisecond; handing it over cost it 9% of its time)
        static const bool queue_for_lock = std::getenv("OOT_RETIRE_LOCKS") != nullptr;
        std::unique_lock lock{BatchMutex(), std::defer_lock};
        if (queue_for_lock) lock.lock();
        else while (!lock.try_lock()) {
            if (retired_serial_atomic.load(std::memory_order_acquire) >= done) break;
            rt::SleepNs(200000);
        }
        if (!lock.owns_lock()) continue;
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
    // RetireOne can release the batch lock while waiting. Another thread may
    // have recorded a new batch in that interval; its staging buffers and views
    // must survive until that batch is submitted and completes.
    if (!batch.cmd) {
        auto actions = std::move(batch.after);
        batch.after.clear();
        for (auto& action : actions) action();
    }
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
// Binding start -> end: GPU-written, not yet (all) in guest memory. Bindings overlap, so a search for an address walks back
// as far as the longest binding could reach. Kept in two maps: the many small bindings (reach at most 1 MiB) and the few
// large ones, each searched with its own reach. One map with one reach had every search walk back over thousands of
// small bindings as far as the longest large one reached - 6-7% of the renderer thread on a player's 4 GB laptop
// (123 MiB of results kept a frame). OOT_AHEAD_SPLIT_KIB=<n>: the size from which a binding counts as large.
struct AheadSet { std::map<u64, u64> map; u64 reach = 0; };
AheadSet ahead_small, ahead_large;
u64 AheadSplit() { static const u64 split = std::getenv("OOT_AHEAD_SPLIT_KIB") ? std::strtoull(std::getenv("OOT_AHEAD_SPLIT_KIB"), nullptr, 10) << 10 : 1ull << 20; return split; }
bool AheadEmpty() { return ahead_small.map.empty() && ahead_large.map.empty(); }
// Calls f(start, end) for each binding overlapping [first, last) (newest key first within each map); stops when f returns true.
template <class F> bool ForAhead(u64 first, u64 last, F&& f) {
    for (AheadSet* set : {&ahead_small, &ahead_large}) {
        auto it = set->map.lower_bound(last);
        while (it != set->map.begin()) {
            --it;
            if (it->second <= first) { if (it->first + set->reach < first) break; continue; } // (none reaches further back)
            if (f(it->first, it->second)) return true;
        }
    }
    return false;
}
std::set<u64> cpu_pages;                   // pages of GPU results the CPU reads (copied back every time; by address:
                                           // the game binds the same memory at other offsets frame to frame)
u64 deferred_bytes = 0, recalled_bytes = 0, eager_bytes = 0;
bool DeferWriteBack() { static const bool on = std::getenv("OOT_EAGER_WRITEBACK") == nullptr; return on; }
constexpr u64 Page = 4096;
// (batch lock held) [address, address + size) of GPU results still on the GPU comes back with the open batch; those
// pages are copied back eagerly from then on (OoT reads a few words at the start of buffers of megabytes).
bool RecallDeviceAhead(u64 address, u64 size) {
    if (AheadEmpty() || !size) return false;
    bool any = false;
    // (whole pages, and every binding on them: a page becomes accessible as a whole)
    const u64 first = address & ~(Page - 1), last = (address + size + Page - 1) & ~(Page - 1);
    ForAhead(first, last, [&](u64 start, u64 end) {
        const u64 lo = std::max(start, first), hi = std::min(end, last);
        if (lo >= hi) return false;
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
        return false;
    });
    return any;
}
} // namespace
void DropDeviceAhead(u64 address, u64 size) { // (a pool goes: its results are not wanted)
    std::lock_guard lock{BatchMutex()};
    for (AheadSet* set : {&ahead_small, &ahead_large})
        for (auto it = set->map.lower_bound(address); it != set->map.end() && it->first < address + size;) {
            SettleDeviceAhead(it->first, it->second);
            it = set->map.erase(it);
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
std::atomic<u64> pending_write_serial{0};
u64 PendingWriteSerial() { return pending_write_serial.load(std::memory_order_relaxed); }
void NotePendingWrite(u64 address, u64 size, bool write_back) {
    pending_write_serial.fetch_add(1, std::memory_order_relaxed);
    if (pending_write_watch) pending_write_watch->push_back({address, size});
    if (!size) return;
    static const u64 watch = std::getenv("OOT_WRITE_WATCH") ? std::strtoull(std::getenv("OOT_WRITE_WATCH"), nullptr, 16) : 0;
    if (watch && address <= watch && watch < address + size) {
        static int shown = 0;
        if (shown++ < 20) RT_LOG("pending write %#llx+%#llx covers watched address (caller %p)", (unsigned long long)address, (unsigned long long)size, __builtin_return_address(0));
    }
    std::lock_guard lock{BatchMutex()};
    AddRange(batch.pending_writes, address, address + size);
    if (write_back && DeferWriteBack()) {
        u64 end = 0;
        const auto small = ahead_small.map.find(address), large = ahead_large.map.find(address);
        if (small != ahead_small.map.end()) end = small->second; else if (large != ahead_large.map.end()) end = large->second;
        if (end < address + size) { end = address + size; ProtectDeviceAhead(address, end, &cpu_pages); }
        if (end - address > AheadSplit()) {
            if (small != ahead_small.map.end()) ahead_small.map.erase(small);
            ahead_large.map[address] = end; ahead_large.reach = std::max(ahead_large.reach, end - address);
        } else { ahead_small.map[address] = end; ahead_small.reach = std::max(ahead_small.reach, end - address); }
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
    const double tl_from = TimelineOn() ? TimelineMs() : 0;
    struct TlEnd { double from; u64 a, s; double locked = 0; int open = 0, flight = 0; ~TlEnd() { if (TimelineOn() && TimelineMs() - from > 0.05) RT_LOG("T %.2f readback %#llx+%#llx waited %.2f (lock %.2f, open batch %d, in-flight waits %d)", TimelineMs(), (unsigned long long)a, (unsigned long long)s, TimelineMs() - from, locked - from, open, flight); } } tl_end{tl_from, address, size};
    std::lock_guard lock{BatchMutex()};
    if (TimelineOn()) tl_end.locked = TimelineMs();
    RecallDeviceAhead(address, size);
    {   // OOT_READBACK_LATE=<rule>[,<rule>...] (after Longshot's readback_late.h): reads that do not wait for the GPU work
        // writing them; the game then reads what memory holds, the results of the work before. Rules: `newest`, the
        // range's writer is the open batch or the newest submission (the game invalidates right after submitting the
        // work that fills the range: on the console it could not count on that work being done either); <hex size>,
        // every range of exactly that many bytes. In Kokiri Forest the game's main thread stood 5.6 of its 18.3 ms a
        // frame in these waits, most of it for two ranges of 0x3980 bytes. Not every range: the exposure meter read
        // late is white flashes. Off unless set (the wait went from 4.7 to 1.2 ms with `3980`, but the frame rate did not
        // move on an RTX 4070: the executor is the limit there); meant for the Steam Deck, where Longshot measured +25%.
        struct Rules { bool newest = false; std::vector<u64> sizes; };
        static const Rules rules = [] { Rules r; const char* e = std::getenv("OOT_READBACK_LATE");
            // (on by default where the GPU shares the system's memory - the Steam Deck: Longshot measured +20% there; =0: off)
            if (!e) e = Vk().properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "newest" : ""; if (!std::strcmp(e, "0")) e = "";
            for (const char* p = e; p && *p;) { if (!std::strncmp(p, "newest", 6)) r.newest = true; else if (const u64 n = std::strtoull(p, nullptr, 16)) r.sizes.push_back(n);
                p = std::strchr(p, ','); if (p) ++p; }
            return r; }();
        bool late = std::find(rules.sizes.begin(), rules.sizes.end(), size) != rules.sizes.end();
        if (!late && rules.newest) {
            bool older = false;
            for (size_t i = 0; i + 1 < in_flight.size(); ++i) older |= Overlaps(in_flight[i].writes, address, size) || Overlaps(in_flight[i].reports, address, size);
            const bool newest = Overlaps(batch.pending_writes, address, size) || Overlaps(batch.pending_reports, address, size) ||
                                (!in_flight.empty() && (Overlaps(in_flight.back().writes, address, size) || Overlaps(in_flight.back().reports, address, size)));
            late = newest && !older;
        }
        if (late) { NoteLateReadRange(address, size); static std::atomic<u64> n{0}; if (n++ < 6) RT_LOG("readback %#llx+%#llx is read late (OOT_READBACK_LATE)", (unsigned long long)address, (unsigned long long)size); return; }
    }
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
    if (Overlaps(batch.pending_writes, address, size) || Overlaps(batch.pending_reports, address, size)) { tl_end.open = 1; Submit(); }
    for (;;) {
        size_t last = in_flight.size();
        for (size_t i = in_flight.size(); i-- > 0;)
            if (Overlaps(in_flight[i].writes, address, size) || Overlaps(in_flight[i].reports, address, size)) { last = i; break; }
        if (last == in_flight.size()) break;
        if (trace_waits) show("a submitted batch", Overlaps(in_flight[last].writes, address, size) ? in_flight[last].writes : in_flight[last].reports);
        ++tl_end.flight;
        RetireOne(true); // (oldest first, up to the last batch writing the range)
    }
}
bool HasPendingWrite(u64 address, u64 size) {
    if (resolve_worker) {
        if (resolve_checked) resolve_checked->push_back({address, size});
        // (GPU results the CPU has not read yet would be recalled here, which records: the executor's to do)
        if (!AheadEmpty() && size) {
            const u64 first = address & ~(Page - 1), last = (address + size + Page - 1) & ~(Page - 1);
            const bool unread = ForAhead(first, last, [&](u64 start, u64 end) {
                const u64 lo = std::max(start, first), hi = std::min(end, last);
                for (u64 page = lo & ~(Page - 1); page < hi; page += Page) if (!cpu_pages.count(page)) return true;
                return false;
            });
            if (unread) { RESOLVE_BAIL(); return true; }
        }
        if (Overlaps(batch.pending_writes, address, size)) return true;
        for (const auto& f : in_flight) if (Overlaps(f.writes, address, size)) return true;
        return false;
    }
    std::lock_guard lock{BatchMutex()};
    RecallDeviceAhead(address, size); // (the runtime reads it: its results come back with the open batch)
    if (Overlaps(batch.pending_writes, address, size)) return true;
    for (const auto& f : in_flight) if (Overlaps(f.writes, address, size)) return true;
    return false;
}

// Several small ranges at once (a texture handle table: a word per element). Each is recorded and answered as
// HasPendingWrite would, but when nothing GPU-written lies anywhere between the first and the last (the usual
// case) that is one look instead of one per range.
bool HasPendingWriteAny(const std::vector<std::pair<u64, u64>>& ranges) {
    if (ranges.empty()) return false;
    if (ranges.size() == 1) return HasPendingWrite(ranges[0].first, ranges[0].second);
    u64 lo = ~0ull, hi = 0;
    for (const auto& [address, size] : ranges) { lo = std::min(lo, address); hi = std::max(hi, address + size); }
    const auto clear = [&] {
        if (!AheadEmpty()) {
            const u64 first = lo & ~(Page - 1), last = (hi + Page - 1) & ~(Page - 1);
            for (AheadSet* set : {&ahead_small, &ahead_large}) {
                auto it = set->map.upper_bound(last);
                if (it != set->map.begin() && std::prev(it)->second > first) return false;
            }
        }
        if (Overlaps(batch.pending_writes, lo, hi - lo)) return false;
        for (const auto& f : in_flight) if (Overlaps(f.writes, lo, hi - lo)) return false;
        return true;
    };
    if (resolve_worker) {
        if (clear()) { if (resolve_checked) resolve_checked->insert(resolve_checked->end(), ranges.begin(), ranges.end()); return false; }
    } else {
        std::lock_guard lock{BatchMutex()};
        if (clear()) return false;
    }
    for (const auto& [address, size] : ranges) if (HasPendingWrite(address, size)) return true;
    return false;
}

// Waits for the submitted batches whose reports cover the range (not for the open batch: nothing is submitted).
// Returns whether the open batch still has reports there.
bool WaitForSubmittedReports(u64 address, u64 size) {
    BatchLock lock;
    for (;;) {
        size_t last = in_flight.size();
        for (size_t i = in_flight.size(); i-- > 0;)
            if (Overlaps(in_flight[i].reports, address, size) || Overlaps(in_flight[i].writes, address, size)) { last = i; break; }
        if (last == in_flight.size()) break;
        RetireOne(true); // (oldest first, up to the last batch reporting into the range)
    }
    return Overlaps(batch.pending_reports, address, size) || Overlaps(batch.pending_writes, address, size);
}

void NotePendingReport(u64 address) {
    std::lock_guard lock{BatchMutex()};
    // (reports follow each other in memory: the run they form is one range, not a map entry each)
    auto& reports = batch.pending_reports;
    if (!reports.empty()) { auto last = std::prev(reports.end()); if (last->second == address) { last->second = address + 16; return; }
        if (last->first <= address && address + 16 <= last->second) return; }
    AddRange(reports, address, address + 16);
}

bool HasPendingReport(u64 address, u64 size) {
    if (resolve_worker) { // (nothing is recorded or retired meanwhile: read without the lock its executor holds)
        if (Overlaps(batch.pending_reports, address, size)) return true;
        for (const auto& f : in_flight) if (Overlaps(f.reports, address, size)) return true;
        return false;
    }
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
