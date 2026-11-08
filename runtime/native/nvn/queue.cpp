// NVN queues, syncs, events and windows.
//
// Submissions execute on the host GPU in order; a fence sync placed on a queue is signalled once
// the work submitted before it has completed, which the queue guarantees by finishing submissions
// before returning from nvnQueueFlush/Finish and FenceSync.

#include <atomic>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <map>
#include <chrono>
#include <thread>

#include "../../native.h"
#include "commands.h"
#include "memory.h"
#include "present.h"
#include "batch.h"

using namespace rt;
using namespace nvn;

namespace {

struct Queue {
    u64 object = 0;
};

Queue*& HostQueue(u64 object) { return *reinterpret_cast<Queue**>(object); }

// NVNsync: +0 state (1 = signalled), +8 owning queue.
void SignalSync(u64 sync) { __atomic_store_n(reinterpret_cast<u32*>(sync), 1u, __ATOMIC_RELEASE); }

} // namespace

// ---- queues ---------------------------------------------------------------------------------------
NATIVE("nvnQueueInitialize") {
    HostQueue(c->x[0]) = new Queue{c->x[0]};
    Ret(c, 1);
}
NATIVE("nvnQueueFinalize") {
    Executor().Finish();
    delete HostQueue(c->x[0]);
}
NATIVE("nvnQueueSetDebugLabel") {}
NATIVE("nvnQueueSubmitCommands") {
    const s32 count = (s32)c->x[1];
    const u64* handles = Ptr<const u64>(c, 2);
    for (s32 i = 0; i < count; ++i) Executor().Submit(c->x[0], handles[i]);
    // Submitted without waiting: results become CPU-visible (fences, events, written buffers) as
    // the batch retires, while the game prepares the next frame.
    nvn::Submit();
}
NATIVE("nvnQueueFlush") { nvn::Submit(); }
NATIVE("nvnQueueFinish") {
    const auto t0 = std::chrono::steady_clock::now();
    Executor().Finish();
    nvn::cpu_wait_ns[2] += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
NATIVE("nvnQueueFenceSync") {
    if (nvn::TimelineOn()) RT_LOG("T %.2f fence %#llx on queue %#llx", nvn::TimelineMs(), (unsigned long long)c->x[1], (unsigned long long)c->x[0]);
    // FenceSync(queue, sync, condition, flags): signalled when the queue's prior work completes.
    __atomic_store_n(Ptr<u32>(c, 1), 0u, __ATOMIC_RELEASE); // (unsignalled from now, also while the queue is parked)
    Executor().QueueOperation(c->x[0], nvn::Op::FenceSync, c->x[1]);
    nvn::Submit();
}
NATIVE("nvnQueueWaitSync") {
    // WaitSync(queue, sync): the queue's later work runs after the sync's fence.
    Executor().QueueOperation(c->x[0], nvn::Op::WaitSync, c->x[1]);
    Ret(c, 1);
}
NATIVE("nvnQueueGetError") {
    if (c->x[1]) std::memset(Ptr(c, 1), 0, 0x10);
    Ret(c, 0); // NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR
}
NATIVE("nvnQueueGetTotalCommandMemoryUsed") { Ret(c, 0); }
NATIVE("nvnQueueGetTotalComputeMemoryUsed") { Ret(c, 0); }
NATIVE("nvnQueueGetTotalControlMemoryUsed") { Ret(c, 0); }
NATIVE("nvnQueueGetTotalRayTracingMemoryUsed") { Ret(c, 0); }
NATIVE("nvnQueueResetMemoryUsageCounts") {}
NATIVE("nvnQueuePresentTexture") {
    // PresentTexture(queue, window, texture index): queue order puts the blit after the frame.
    PresentPacing();
    Executor().QueueOperation(c->x[0], nvn::Op::QueuePresent, c->x[1], (u64)(s32)c->x[2]);
}

// ---- syncs ----------------------------------------------------------------------------------------
NATIVE("nvnSyncInitialize") {
    std::memset(Ptr(c, 0), 0, 16);
    Ret(c, 1);
}
NATIVE("nvnSyncFinalize") {}
NATIVE("nvnSyncSetDebugLabel") {}
NATIVE("nvnSyncWait") {
    // Wait(sync, timeout ns): NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED (0) / TIMEOUT_EXPIRED (2)
    const u32* state = Ptr<const u32>(c, 0);
    static const bool trace = std::getenv("OOT_SYNC_TRACE") != nullptr;
    if (trace) { static std::map<std::pair<u64, bool>, u64> calls; static u64 n = 0; ++calls[{c->x[1] == 0 ? 0 : c->x[1] < 1000000 ? 1 : 2, __atomic_load_n(state, __ATOMIC_ACQUIRE) != 0}];
        if (++n % 2000 == 0) { for (auto& [k, v] : calls) RT_LOG("sync wait timeout class %llu signalled %d: %llu (lr %#llx)", (unsigned long long)k.first, (int)k.second, (unsigned long long)v, (unsigned long long)c->x[30]); calls.clear(); } }
    if (__atomic_load_n(state, __ATOMIC_ACQUIRE)) return Ret(c, 0);
    struct Timed { std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timed() { nvn::cpu_wait_ns[0] += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); } } timed;
    if (nvn::TimelineOn()) RT_LOG("T %.2f sync wait %#llx start", nvn::TimelineMs(), (unsigned long long)c->x[0]);
    struct End { u64 s; ~End() { if (nvn::TimelineOn()) RT_LOG("T %.2f sync wait %#llx end", nvn::TimelineMs(), (unsigned long long)s); } } end{c->x[0]};
    // Retire submitted batches (oldest first) until the sync is signalled. Its fence may still be
    // queued behind a wait (a parked queue): then keep waiting, up to the timeout.
    const u64 timeout = c->x[1];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(std::min<u64>(timeout, 60'000'000'000ull));
    {
        rt::CoreRelease release;
        for (;;) {
            WaitUntil([state] { return __atomic_load_n(state, __ATOMIC_ACQUIRE) != 0; });
            if (__atomic_load_n(state, __ATOMIC_ACQUIRE) || timeout == 0 || std::chrono::steady_clock::now() >= deadline) break;
            Executor().Resume();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    Ret(c, __atomic_load_n(state, __ATOMIC_ACQUIRE) ? 1 : 2);
}

// ---- events: object holds the builder (+0 memory pool, +8 offset); the value lives in the pool ------
NATIVE("nvnEventInitialize") {
    std::memcpy(Ptr(c, 0), Ptr(c, 1), 16);
    Ret(c, 1);
}
NATIVE("nvnEventFinalize") {}
static u32* EventValue(u64 event) {
    const u64* e = reinterpret_cast<const u64*>(event);
    return reinterpret_cast<u32*>(PoolOf(e[0])->address + e[1]);
}
NATIVE("nvnEventGetValue") { Poll(); Ret(c, __atomic_load_n(EventValue(c->x[0]), __ATOMIC_ACQUIRE)); }
#ifdef __linux__
std::atomic<int> profile_signal_tid{0}; // (profiler.cpp: OOT_PROFILE_WALL=signal)
#endif
NATIVE("nvnEventSignal") {
    { static u32 shown = 0; if (std::getenv("OOT_EVENT_TRACE") && shown++ < 30) RT_LOG("CPU signals event %#llx mode %llu value %llu", (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2]); }
    // Signal(event, mode, value): mode 0 writes, 1 adds
    u32* v = EventValue(c->x[0]);
#ifdef __linux__
    { extern std::atomic<int> profile_signal_tid; static thread_local const int tid = (int)syscall(SYS_gettid); profile_signal_tid.store(tid, std::memory_order_relaxed); }
#endif
    nvn::NoteCpuEvent(v);
    if (c->x[1] == 1) __atomic_fetch_add(v, (u32)c->x[2], __ATOMIC_ACQ_REL);
    else __atomic_store_n(v, (u32)c->x[2], __ATOMIC_RELEASE);
    if (nvn::TimelineOn()) RT_LOG("T %.2f CPU signal %p = %llu", nvn::TimelineMs(), (void*)v, (unsigned long long)c->x[2]);
    Executor().Resume(); // queues waiting for it continue
}
NATIVE("nvnEventGetMemoryPool") { Ret(c, Ptr<u64>(c, 0)[0]); }
NATIVE("nvnEventGetMemoryOffset") { Ret(c, Ptr<u64>(c, 0)[1]); }

// ---- windows: object holds the builder; +0x80 current texture index --------------------------------
NATIVE("nvnWindowInitialize") {
    std::memcpy(Ptr(c, 0), Ptr(c, 1), WindowBuilderSize);
    *reinterpret_cast<s32*>(c->x[0] + WindowBuilderSize) = -1;
    const u32 count=*reinterpret_cast<u32*>(c->x[0]+0x10);
    const auto* textures=*reinterpret_cast<const u64**>(c->x[0]+0x38);
    *reinterpret_cast<std::vector<u64>**>(c->x[0]+0x78)=new std::vector<u64>(textures,textures+count);
    CreateSwapchain();
    Ret(c, 1);
}
NATIVE("nvnWindowFinalize") { delete *reinterpret_cast<std::vector<u64>**>(c->x[0]+0x78); }
NATIVE("nvnWindowSetDebugLabel") {}
NATIVE("nvnWindowAcquireTexture") {
    // AcquireTexture(window, sync, int* index): returns NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS
    const u8* w = Ptr<u8>(c, 0);
    s32 active;
    std::memcpy(&active, w + 0x14, 4);
    if (active <= 0) std::memcpy(&active, w + 0x10, 4);
    s32* current = reinterpret_cast<s32*>(c->x[0] + WindowBuilderSize);
    *current = (*current + 1) % active;
    *Ptr<s32>(c, 2) = *current;
    if (c->x[1]) SignalSync(c->x[1]);
    Ret(c, 0);
}
NATIVE("nvnWindowGetNumTextures") { Ret(c, *reinterpret_cast<u32*>(c->x[0] + 0x10)); }
NATIVE("nvnWindowGetNumActiveTextures") { Ret(c, *reinterpret_cast<u32*>(c->x[0] + 0x14)); }
NATIVE("nvnWindowSetNumActiveTextures") { *reinterpret_cast<u32*>(c->x[0] + 0x14) = (u32)c->x[1]; }
NATIVE("nvnWindowGetNativeWindow") { Ret(c, *reinterpret_cast<u64*>(c->x[0] + 0x18)); }
NATIVE("nvnWindowGetPresentInterval") { Ret(c, *reinterpret_cast<u32*>(c->x[0] + 0x20) + 1); }
NATIVE("nvnWindowSetPresentInterval") { *reinterpret_cast<u32*>(c->x[0] + 0x20) = (u32)c->x[1] - 1; static u32 last = ~0u; if ((u32)c->x[1] != last) { last = (u32)c->x[1]; RT_LOG("window present interval %u", last); } }
NATIVE("nvnWindowGetAcquireTimeout") { Ret(c, *reinterpret_cast<u64*>(c->x[0] + 0x28)); }
NATIVE("nvnWindowSetAcquireTimeout") { *reinterpret_cast<u64*>(c->x[0] + 0x28) = c->x[1]; }
NATIVE("nvnWindowSetCrop") {}
NATIVE("nvnWindowGetCrop") {
    s32* out = Ptr<s32>(c, 1);
    out[0] = out[1] = 0;
    out[2] = 1920;
    out[3] = 1080;
}
NATIVE("nvnWindowSetHdrContentMetadata") {}
NATIVE("nvnWindowSetHdrDisplayMetadata") {}
