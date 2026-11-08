#pragma once

#include <thread>

#include "rt.h"

namespace rt {
// Pumps the window's events; called by the window's thread while it waits in the kernel.
extern void (*g_window_pump)();
extern std::thread::id g_window_thread;

struct KObject {
    virtual ~KObject() = default;
    virtual bool Signaled() const { return false; }
    std::string kind;
};

struct KThread : KObject {
    bool Signaled() const override { return exited; }
    u32 handle = 0;
    u64 id = 0;
    u64 entry = 0, arg = 0, stack_top = 0, tlr = 0;
    s32 priority = 44, core = 0; // core: ideal core
    u64 affinity = 0;            // allowed cores (0: the ideal core only)
    Cpu* cpu = nullptr;
    std::thread host;
    bool exited = false;
    bool cancel_wait = false;
    u64 cancels = 0, cancelled_by = 0, cancel_lr = 0; std::chrono::steady_clock::time_point cancel_time{}; // (hang report)
    // (hang report) the thread's recent synchronization SVCs: svc, first argument, result, caller
    struct SyncEvent { std::chrono::steady_clock::time_point t; u32 svc, result; u64 arg, lr; };
    SyncEvent sync_events[24]{}; u32 sync_event_count = 0;
    void NoteSync(u32 svc, u64 arg, u32 result, u64 lr) { sync_events[sync_event_count++ % 24] = {std::chrono::steady_clock::now(), svc, result, arg, lr}; }
};

void InitializeKernel();
void CreateKernelEvent(u32* writable, u32* readable);
void SignalKernelEvent(u32 handle);
void ClearKernelEvent(u32 handle);
KThread* CreateMainThread(u64 stack_top);
KThread* CurrentKThread();
void BindThread(KThread* t);
void ExitCurrentThread();

// Per-core execution (see kernel.cpp): the calling guest thread's core lock.
void EnterCore();
void LeaveCore();
bool HoldsCore();
s32 CurrentCore();
// Gives up the caller's core while it blocks in host code; taken again at the end of the scope.
struct CoreRelease {
    bool held = HoldsCore();
    CoreRelease() { LeaveCore(); }
    ~CoreRelease() { if (held) EnterCore(); }
    CoreRelease(const CoreRelease&) = delete;
    CoreRelease& operator=(const CoreRelease&) = delete;
};

} // namespace rt
