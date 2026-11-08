// nn::os fibers on host contexts. Recompiled code keeps guest call frames on the host stack, so a
// guest-level stack switch cannot work; instead every guest fiber gets its own host context (and
// host stack) and its own guest CPU state on the guest stack the game supplied. SwitchToFiber swaps
// host contexts; the SDK's assembly context switch is never used.

#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <ucontext.h>
#endif

#include "../native.h"

using namespace rt;

namespace {

constexpr size_t HostStackSize = 8u << 20;

#ifdef _WIN32
using HostContext = void*; // a Win32 fiber
#else
using HostContext = ucontext_t;
#endif

struct HostFiber {
    HostContext context{};
    void* host_stack = nullptr;
    Cpu* cpu = nullptr;
    u64 function = 0, argument = 0;
};

std::mutex fibers_mutex;
std::unordered_map<u64, HostFiber*> fibers; // guest FiberType* -> host fiber

struct ThreadFibers {
    u64 current = 0;          // guest FiberType* running on this thread (0: the thread itself)
    HostContext thread_context{};
    Cpu* thread_cpu = nullptr;
};
thread_local ThreadFibers tf;

HostFiber* Find(u64 fiber) {
    std::lock_guard lock{fibers_mutex};
    auto it = fibers.find(fiber);
    if (it == fibers.end()) Fatal("SwitchToFiber to an uninitialized fiber %#llx", (unsigned long long)fiber);
    return it->second;
}

void SwitchTo(u64 target);

void FiberRun(u64 key) {
    HostFiber* f = Find(key);
    BindCpu(f->cpu);
    // The fiber function returns the fiber to continue with when this one ends.
    const u64 next = CallGuest(f->cpu, f->function, {f->argument});
    SwitchTo(next);
    Fatal("finished fiber %#llx was resumed", (unsigned long long)key);
}

#ifndef _WIN32
void FiberEntry(unsigned hi, unsigned lo) { FiberRun((u64)hi << 32 | lo); }
#endif

void SwitchTo(u64 target) {
    if (!tf.thread_cpu) tf.thread_cpu = CurrentCpu();
    const u64 from = tf.current;
    HostContext* from_context = from ? &Find(from)->context : &tf.thread_context;
    HostContext* to_context = target ? &Find(target)->context : &tf.thread_context;
    if (from == target) return;
    // Fibers run on whichever thread switches to them and share that thread's TLS.
    if (target) {
        Cpu* to = Find(target)->cpu;
        to->tpidrro = tf.thread_cpu->tpidrro;
        to->tpidr = tf.thread_cpu->tpidr;
    }
    tf.current = target;
#ifdef _WIN32
    (void)from_context;
    // The switching thread must itself be a fiber.
    if (!tf.thread_context) {
        tf.thread_context = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiber(nullptr);
        if (!tf.thread_context) Fatal("cannot convert the thread to a fiber");
    }
    SwitchToFiber(*to_context);
#else
    swapcontext(from_context, to_context);
#endif
    // Resumed: this context's own guest CPU state is current again.
    BindCpu(tf.current ? Find(tf.current)->cpu : tf.thread_cpu);
}

} // namespace

// InitializeFiber(FiberType*, FiberType* (*function)(void*), void* argument, void* stack,
//                 size_t stack size, int flags)
NATIVE("_ZN2nn2os15InitializeFiberEPNS0_9FiberTypeEPFS2_PvES3_S3_mi") {
    const u64 key = c->x[0];
    auto* f = new HostFiber{};
    f->function = c->x[1];
    f->argument = c->x[2];
    f->cpu = new Cpu{};
    f->cpu->sp = (c->x[3] + c->x[4]) & ~15ULL;
    f->cpu->tpidr = c->tpidr;
    f->cpu->tpidrro = c->tpidrro;
    f->cpu->fpcr = c->fpcr;
#ifdef _WIN32
    f->context = CreateFiberEx(0, HostStackSize, 0, [](void* k) { FiberRun(reinterpret_cast<u64>(k)); },
                               reinterpret_cast<void*>(key));
    if (!f->context) Fatal("cannot create a fiber");
#else
    f->host_stack = mmap(nullptr, HostStackSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (f->host_stack == MAP_FAILED) Fatal("cannot allocate a fiber host stack");
    getcontext(&f->context);
    f->context.uc_stack.ss_sp = f->host_stack;
    f->context.uc_stack.ss_size = HostStackSize;
    f->context.uc_link = nullptr;
    makecontext(&f->context, reinterpret_cast<void (*)()>(FiberEntry), 2, (unsigned)(key >> 32), (unsigned)key);
#endif
    std::lock_guard lock{fibers_mutex};
    fibers[key] = f;
}
NATIVE("_ZN2nn2os13FinalizeFiberEPNS0_9FiberTypeE") {
    HostFiber* f;
    {
        std::lock_guard lock{fibers_mutex};
        auto it = fibers.find(c->x[0]);
        if (it == fibers.end()) return;
        f = it->second;
        fibers.erase(it);
    }
#ifdef _WIN32
    DeleteFiber(f->context);
#else
    munmap(f->host_stack, HostStackSize);
#endif
    delete f->cpu;
    delete f;
}
NATIVE("_ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE") { SwitchTo(c->x[0]); }
NATIVE("_ZN2nn2os15GetCurrentFiberEv") { Ret(c, tf.current); }
