// Native implementation of the Horizon system calls the SDK's own code uses for threads,
// synchronization, time and memory. Guest threads are host threads; waiting uses one kernel lock
// and per-thread condition variables. Service IPC is deliberately absent: SDK service APIs are
// replaced at the nn:: level (see native/), so SendSyncRequest reaching here is a missing override.

#include <cstdarg>
#include <set>
#include <pthread.h>
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "kernel.h"

extern "C" __attribute__((weak)) unsigned long long AudioUpdateCount(); // native/audio.cpp (hang diagnosis; absent in tests)
namespace rt {
void (*g_window_pump)() = nullptr;
std::thread::id g_window_thread{};
namespace {

constexpr u32 ResultSuccess = 0;
constexpr u32 ResultTimedOut = 0xEA01;
constexpr u32 ResultInvalidState = 0xFA01;
constexpr u32 ResultInvalidHandle = 0xE401;
constexpr u32 ResultCancelled = 0xEC01;
constexpr u32 ResultNotFound = 0xF201;
constexpr u32 ResultInvalidSize = 0xCA01;
constexpr u32 ResultInvalidAddress = 0xCC01;
constexpr u32 ResultOutOfResource = 0xCE01;
constexpr u32 ResultInvalidCurrentMemory = 0xD401;
constexpr u32 ResultInvalidMemoryRegion = 0xDC01;
constexpr u32 HandleWaitMask = 0x40000000;
constexpr u32 CurrentThreadHandle = 0xFFFF8000;
constexpr u32 CurrentProcessHandle = 0xFFFF8001;

std::mutex k_lock;

struct KEvent : KObject {
    bool signaled = false;
    bool Signaled() const override { return signaled; }
};

struct Waiter {
    std::condition_variable cv;
    bool woken = false;
    // (hang report) who waits, from where, since when
    KThread* thread = nullptr; u64 lr = 0; std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    s32 count = -1; u32 handles[4]{};
};

std::unordered_map<u32, std::shared_ptr<KObject>> handles;
u32 next_handle = 0x100;
std::atomic<u64> next_thread_id{0x50};

u32 AddHandle(std::shared_ptr<KObject> obj) {
    const u32 h = next_handle;
    next_handle += 1;
    handles[h] = std::move(obj);
    return h;
}

thread_local KThread* current_thread = nullptr;

KThread* CurrentThread() { return current_thread; }

std::shared_ptr<KObject> GetObject(u32 handle) {
    if (handle == CurrentThreadHandle && current_thread) {
        auto it = handles.find(current_thread->handle);
        return it == handles.end() ? nullptr : it->second;
    }
    auto it = handles.find(handle);
    return it == handles.end() ? nullptr : it->second;
}

using Clock = std::chrono::steady_clock;
Clock::time_point Deadline(s64 timeout_ns) {
    if (timeout_ns < 0) return Clock::time_point::max();
    return Clock::now() + std::chrono::nanoseconds(timeout_ns);
}

// Waits on the thread's own condition variable (k_lock held). Returns false on timeout.
// The window's thread (the guest main thread) pumps window events while it waits: during long
// loads it otherwise blocked for seconds and the desktop reported the game as not responding.
bool Block(std::unique_lock<std::mutex>& lock, Waiter& w, Clock::time_point deadline) {
    if (g_window_pump && std::this_thread::get_id() == g_window_thread) {
        while (!w.woken) {
            const auto slice = std::min(deadline, Clock::now() + std::chrono::milliseconds(50));
            if (w.cv.wait_until(lock, slice) == std::cv_status::timeout && !w.woken) {
                if (Clock::now() >= deadline) return false;
                lock.unlock(); g_window_pump(); lock.lock();
            }
        }
        return true;
    }
    while (!w.woken) {
        if (deadline == Clock::time_point::max()) {
            w.cv.wait(lock);
        } else if (w.cv.wait_until(lock, deadline) == std::cv_status::timeout && !w.woken) {
            return false;
        }
    }
    return true;
}

// ---- synchronization objects (WaitSynchronization) --------------------------------------------
std::list<Waiter*> sync_waiters;
void NotifySync() {
    for (Waiter* w : sync_waiters) {
        w->woken = true;
        w->cv.notify_one();
    }
}

// ---- mutexes and condition variables (ArbitrateLock / WaitProcessWideKeyAtomic) --------------
struct LockWaiter {
    u64 addr;
    u32 tag;
    Waiter* w;
};
std::list<LockWaiter> lock_waiters;
struct KeyWaiter {
    u64 key;
    u64 addr;
    u32 tag;
    Waiter* w;
};
std::list<KeyWaiter> key_waiters;

u32* U32At(u64 addr) { return reinterpret_cast<u32*>(addr); }

// Hands the mutex at addr to its next waiter, or clears it. k_lock held.
void ReleaseMutex(u64 addr) {
    auto it = std::find_if(lock_waiters.begin(), lock_waiters.end(), [&](auto& lw) { return lw.addr == addr; });
    if (it == lock_waiters.end()) {
        __atomic_store_n(U32At(addr), 0u, __ATOMIC_SEQ_CST);
        return;
    }
    const LockWaiter next = *it;
    lock_waiters.erase(it);
    const bool more = std::any_of(lock_waiters.begin(), lock_waiters.end(), [&](auto& lw) { return lw.addr == addr; });
    __atomic_store_n(U32At(addr), next.tag | (more ? HandleWaitMask : 0), __ATOMIC_SEQ_CST);
    next.w->woken = true;
    next.w->cv.notify_one();
}

// Gives the mutex at addr to `tag` if free, else queues it as a lock waiter. k_lock held.
// Returns true if acquired immediately.
bool AcquireOrQueue(u64 addr, u32 tag, Waiter* w) {
    // One atomic update, as the console kernel does: guest code releases the mutex in user mode
    // without the kernel lock, and a release between a failed acquire and setting the waiter
    // bit left "no owner, waiters" (0x40000000) that no unlock ever cleared (boot hangs).
    u32 value = __atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST);
    for (;;) {
        const u32 next = value == 0 ? tag : (value | HandleWaitMask);
        if (__atomic_compare_exchange_n(U32At(addr), &value, next, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) break;
    }
    if (value == 0) return true;
    lock_waiters.push_back({addr, tag, w});
    return false;
}

// ---- address arbitration (WaitForAddress / SignalToAddress) -----------------------------------
struct AddressWaiter {
    u64 addr;
    Waiter* w;
    u32 type = 0; u64 value = 0, lr = 0, thread = 0; Clock::time_point since{};
};
std::list<AddressWaiter> address_waiters;
// (OOT_ARB_TRACE) recent arbiter operations, for the addresses a hang is stuck on.
struct ArbiterOp { double t; u64 thread, addr, lr; u32 op, type; s64 value, count; u32 before, after; s32 result; };
std::unordered_map<u64, std::deque<ArbiterOp>> arbiter_history; // last operations per address; k_lock held
double ArbNow() { static const auto t0 = Clock::now(); return std::chrono::duration<double>(Clock::now() - t0).count(); }
void NoteArbiterOp(u32 op, u32 type, u64 addr, s64 value, s64 count, u32 before, s32 result, u64 lr) {
    static const bool on = std::getenv("OOT_ARB_TRACE") != nullptr; if (!on) return;
    auto& h = arbiter_history[addr];
    h.push_back({ArbNow(), current_thread ? current_thread->id : 0, addr, lr, op, type, value, count, before, __atomic_load_n(reinterpret_cast<u32*>(addr), __ATOMIC_SEQ_CST), result});
    if (h.size() > 24) h.pop_front();
}
std::unordered_map<u64, std::deque<std::string>> thread_history; // (OOT_ARB_TRACE) recent sync events per thread; k_lock held
void NoteThreadEvent(u64 thread, const char* format, ...) {
    static const bool on = std::getenv("OOT_ARB_TRACE") != nullptr; if (!on) return;
    char text[200]; va_list args; va_start(args, format); std::vsnprintf(text, sizeof text, format, args); va_end(args);
    char line[240]; std::snprintf(line, sizeof line, "%.4f %s", ArbNow(), text);
    auto& h = thread_history[thread]; h.push_back(line); if (h.size() > 16) h.pop_front();
}
struct SyncWaitInfo { u64 thread, lr; std::string kinds; Clock::time_point since; };
std::list<SyncWaitInfo> sync_infos; // (OOT_ARB_TRACE) threads in WaitSynchronization; k_lock held

// OOT_ARB_TRACE: every 3 s, threads blocked in WaitForAddress for over 2 s (hang diagnosis).
void StartArbiterWatchdog() {
    static const bool on = std::getenv("OOT_ARB_TRACE") != nullptr;
    static std::once_flag once;
    if (!on) return;
    std::call_once(once, [] {
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                if (AudioUpdateCount) RT_LOG("watchdog: %llu audio renderer updates so far", AudioUpdateCount());
                std::lock_guard lock{k_lock};
                const auto now = Clock::now();
                for (const auto& l : lock_waiters) {
                    const u32 word = __atomic_load_n(reinterpret_cast<u32*>(l.addr), __ATOMIC_SEQ_CST);
                    auto owner = std::dynamic_pointer_cast<KThread>(GetObject(word & ~HandleWaitMask));
                    auto waiter = std::dynamic_pointer_cast<KThread>(GetObject(l.tag));
                    RT_LOG("mutex: thread %d waits for %#llx = %#x (owner thread %d%s)", waiter ? (int)waiter->id : -1, (unsigned long long)l.addr, word,
                           owner ? (int)owner->id : -1, owner && owner->exited ? ", exited" : "");
                    static std::set<u64> shown_mutexes;
                    if (shown_mutexes.insert(l.addr).second) for (const auto& o : arbiter_history[l.addr])
                        RT_LOG("  %.4f thread %llu %s owner %#llx tag %#llx: %#x -> %#x result %d (lr %#llx)", o.t, (unsigned long long)o.thread,
                               o.op == 2 ? (o.type ? "lock-return" : "lock") : o.op == 3 ? "unlock" : o.op == 4 ? "condvar-wait(releases)" : o.op == 5 ? (o.type ? "condvar-signal(acquired)" : "condvar-signal(queued)") : "?", (unsigned long long)o.value, (unsigned long long)o.count, o.before, o.after, o.result, (unsigned long long)o.lr);
                }
                for (const auto& k : key_waiters) {
                    auto waiter = std::dynamic_pointer_cast<KThread>(GetObject(k.tag));
                    RT_LOG("condvar: thread %d waits on key %#llx (mutex %#llx = %#x)", waiter ? (int)waiter->id : -1, (unsigned long long)k.key, (unsigned long long)k.addr,
                           __atomic_load_n(reinterpret_cast<u32*>(k.addr), __ATOMIC_SEQ_CST));
                }
                for (const auto& i : sync_infos) {
                    if (now - i.since < std::chrono::seconds(2)) continue;
                    RT_LOG("sync wait: thread %llu waits %.1f s on %s(lr %#llx)", (unsigned long long)i.thread, std::chrono::duration<double>(now - i.since).count(), i.kinds.c_str(), (unsigned long long)i.lr);
                    static std::set<u64> shown_threads;
                    if (now - i.since > std::chrono::seconds(10) && shown_threads.insert(i.thread).second)
                        for (const auto& e : thread_history[i.thread]) RT_LOG("  thread %llu: %s", (unsigned long long)i.thread, e.c_str());
                }
                static std::set<u64> dumped;
                for (const auto& a : address_waiters) {
                    if (now - a.since < std::chrono::seconds(10) || dumped.count(a.addr)) continue;
                    dumped.insert(a.addr);
                    RT_LOG("arbiter history for %#llx:", (unsigned long long)a.addr);
                    for (const auto& o : arbiter_history[a.addr]) {
                        RT_LOG("  %.4f thread %llu %s type %u value %lld count %lld: %#x -> %#x result %d (lr %#llx)", o.t, (unsigned long long)o.thread, (o.op == 0 ? "wait" : o.op == 1 ? "signal" : o.op == 2 ? (o.type ? "lock-return" : "lock") : "unlock"), o.type,
                               (long long)o.value, (long long)o.count, o.before, o.after, o.result, (unsigned long long)o.lr); }
                }
                for (const auto& a : address_waiters) {
                    if (now - a.since < std::chrono::seconds(2)) continue;
                    RT_LOG("arbiter: thread %llu waits %.1f s at %#llx type %u for value %#llx (now %#x), lr %#llx",
                           (unsigned long long)a.thread, std::chrono::duration<double>(now - a.since).count(), (unsigned long long)a.addr, a.type,
                           (unsigned long long)a.value, __atomic_load_n(reinterpret_cast<u32*>(a.addr), __ATOMIC_SEQ_CST), (unsigned long long)a.lr);
                }
            }
        }).detach();
    });
}
// Hang report (always on, unless OOT_NO_HANG_WATCH): when a guest mutex wait exceeds 15 s, the
// whole wait graph is printed once - mutex waits with owners, WaitSynchronization waits with
// their handles and cancel state, condition-variable and address waits.
void StartHangWatch() {
    static std::once_flag once;
    static const bool off = std::getenv("OOT_NO_HANG_WATCH") != nullptr;
    if (off) return;
    std::call_once(once, [] {
        std::thread([] {
            rt::SetThreadName("hang-watch");
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                std::lock_guard lock{k_lock};
                const auto now = Clock::now();
                const auto age = [&](Clock::time_point t) { return std::chrono::duration<double>(now - t).count(); };
                const auto id = [](KThread* t) { return t ? (long long)t->id : -1ll; };
                // On request: OOT_HANG_TRIGGER=<path> (the report is made when that file appears).
                // (Mutex waits alone are no sign of a hang: nn::async executors idle with one thread
                // waiting for work while holding the scheduler mutex the others queue on.)
                static const char* trigger = std::getenv("OOT_HANG_TRIGGER");
                if (!trigger || std::remove(trigger) != 0) continue;
                RT_LOG("hang report (requested)");
                // Guest call stacks: the frame-record chain from each blocked thread's x29.
                const auto stack = [](KThread* t) {
                    std::string out; if (!t || !t->cpu) return out;
                    u64 fp = t->cpu->x[29]; char b[32];
                    std::snprintf(b, sizeof b, " %#llx", (unsigned long long)t->cpu->x[30]); out += b;
                    for (int i = 0; i < 14 && fp && fp % 8 == 0 && fp >= 0x1000 && fp < (1ull << 47); ++i) {
                        u64 next = 0, ret = 0;
                        std::memcpy(&next, reinterpret_cast<void*>(fp), 8); std::memcpy(&ret, reinterpret_cast<void*>(fp + 8), 8);
                        if (!ret) break;
                        std::snprintf(b, sizeof b, " %#llx", (unsigned long long)(ret & 0xffffffffffull)); out += b;
                        if (next <= fp) break;
                        fp = next;
                    }
                    return out;
                };
                std::set<KThread*> blocked;
                for (const auto& l : lock_waiters) blocked.insert(l.w->thread);
                for (const Waiter* w : sync_waiters) blocked.insert(w->thread);
                for (const auto& k : key_waiters) blocked.insert(k.w->thread);
                for (const auto& a : address_waiters) blocked.insert(a.w->thread);
                for (KThread* t : blocked) if (t) RT_LOG("  stack of thread %llu:%s", (unsigned long long)t->id, stack(t).c_str());
                for (const auto& l : lock_waiters) {
                    const u32 word = __atomic_load_n(reinterpret_cast<u32*>(l.addr), __ATOMIC_SEQ_CST);
                    auto owner = std::dynamic_pointer_cast<KThread>(GetObject(word & ~HandleWaitMask));
                    RT_LOG("  mutex wait: thread %lld at %#llx = %#x (owner thread %lld%s) for %.1f s, lr %#llx", id(l.w->thread), (unsigned long long)l.addr, word,
                           owner ? (long long)owner->id : -1ll, owner && owner->exited ? ", exited" : "", age(l.w->since), (unsigned long long)l.w->lr);
                }
                for (const Waiter* w : sync_waiters) {
                    std::string hs;
                    for (s32 i = 0; i < w->count && i < 4; ++i) {
                        auto o = GetObject(w->handles[i]); char b[64];
                        std::snprintf(b, sizeof b, " %#x(%s", w->handles[i], o ? o->kind.c_str() : "closed"); hs += b;
                        if (auto t = std::dynamic_pointer_cast<KThread>(o)) { std::snprintf(b, sizeof b, " #%llu", (unsigned long long)t->id); hs += b; }
                        hs += o && o->Signaled() ? ", signaled)" : ")";
                    }
                    KThread* t = w->thread;
                    RT_LOG("  sync wait: thread %lld, %d handles%s, for %.1f s, lr %#llx; cancel flag %d, %llu cancels (last by thread %llu from lr %#llx, %.1f s ago)",
                           id(t), w->count, hs.c_str(), age(w->since), (unsigned long long)w->lr, t ? (int)t->cancel_wait : -1,
                           t ? (unsigned long long)t->cancels : 0ull, t ? (unsigned long long)t->cancelled_by : 0ull, t ? (unsigned long long)t->cancel_lr : 0ull,
                           t && t->cancels ? age(t->cancel_time) : -1.0);
                }
                for (const auto& k : key_waiters)
                    RT_LOG("  condvar wait: thread %lld on key %#llx (mutex %#llx = %#x) for %.1f s, lr %#llx", id(k.w->thread), (unsigned long long)k.key, (unsigned long long)k.addr,
                           __atomic_load_n(reinterpret_cast<u32*>(k.addr), __ATOMIC_SEQ_CST), age(k.w->since), (unsigned long long)k.w->lr);
                std::set<KThread*> involved;
                for (const auto& l : lock_waiters) { involved.insert(l.w->thread);
                    const u32 word = __atomic_load_n(reinterpret_cast<u32*>(l.addr), __ATOMIC_SEQ_CST);
                    if (auto owner = std::dynamic_pointer_cast<KThread>(GetObject(word & ~HandleWaitMask))) involved.insert(owner.get()); }
                for (const Waiter* w : sync_waiters) if (w->count == 0) involved.insert(w->thread);
                for (KThread* t : involved) {
                    if (!t) continue;
                    RT_LOG("  thread %llu recent sync calls (svc 0x1xx = entry, 0x219 = cancel received):", (unsigned long long)t->id);
                    const u32 n = std::min<u32>(t->sync_event_count, 24);
                    for (u32 i = t->sync_event_count - n; i < t->sync_event_count; ++i) { const auto& e = t->sync_events[i % 24];
                        RT_LOG("    %.4f s ago: svc %#x arg %#llx result %#x lr %#llx", age(e.t), e.svc, (unsigned long long)e.arg, e.result, (unsigned long long)e.lr); }
                }
                for (const auto& a : address_waiters)
                    RT_LOG("  address wait: thread %lld at %#llx type %u value %#llx (now %#x) for %.1f s, lr %#llx", id(a.w->thread), (unsigned long long)a.addr, a.type,
                           (unsigned long long)a.value, __atomic_load_n(reinterpret_cast<u32*>(a.addr), __ATOMIC_SEQ_CST), age(a.w->since), (unsigned long long)a.w->lr);
            }
        }).detach();
    });
}
s32 WakeAddress(u64 addr, s32 count) {
    s32 woken = 0;
    for (auto it = address_waiters.begin(); it != address_waiters.end() && (count <= 0 || woken < count);) {
        if (it->addr == addr) {
            it->w->woken = true;
            it->w->cv.notify_one();
            it = address_waiters.erase(it);
            ++woken;
        } else {
            ++it;
        }
    }
    return woken;
}

s32 CountAddressWaiters(u64 addr) {
    return (s32)std::count_if(address_waiters.begin(), address_waiters.end(), [&](auto& a) { return a.addr == addr; });
}

// ---- memory ------------------------------------------------------------------------------------
// Guest memory is private anonymous memory everywhere (OOT_ANONYMOUS_MEMORY): AMD's Vulkan drivers import only
// anonymous pages (VK_EXT_external_memory_host, the kernel's userptr), so a shared memfd heap could not reach the
// GPU on the Steam Deck. Stacks mapped from the heap (SvcMapMemory) are copies then: the source pages are
// inaccessible while mapped, so a copy is exact (copied back on unmap). Discarded pages read as zero again.
#define OOT_ANONYMOUS_MEMORY 1
#ifdef _WIN32
constexpr int DiscardAdvice = MADV_REMOVE;  // (platform.cpp: decommit)
#else
constexpr int DiscardAdvice = MADV_DONTNEED; // (private anonymous: zero-filled on the next access)
#endif
u64 heap_size = 0;
bool heap_reserved = false;
int heap_fd = -1;
constexpr u64 AliasBase = 0x1800000000ULL, AliasSize = 0x1000000000ULL;
int physical_fd = -1;
std::map<u64, u64> physical_mappings; // base -> size, nonoverlapping merged intervals

struct StackMapping {
    u64 destination, source, size;
};
std::map<u64, StackMapping> stack_mappings; // destination -> mapping; protected by k_lock

bool ContainsRange(u64 base, u64 length, u64 address, u64 size) {
    return size && address >= base && size <= length && address - base <= length - size;
}

bool Overlaps(u64 a, u64 size_a, u64 b, u64 size_b) {
    return a < b + size_b && b < a + size_a;
}

u32 SvcMapMemory(Cpu* c, bool map) {
    const u64 dst = c->x[0], src = c->x[1], size = c->x[2];
    if (dst & 0xfff) return ResultInvalidAddress;
    if ((src & 0xfff) || !size || (size & 0xfff)) return ResultInvalidSize;
    if (!ContainsRange(StackBase, StackSize, dst, size)) return ResultInvalidMemoryRegion;
    std::lock_guard lock{k_lock};
    int source_fd = heap_fd;
    u64 source_offset = src - HeapBase;
    if (!ContainsRange(HeapBase, heap_size, src, size)) {
        auto it = physical_mappings.upper_bound(src);
        if (it == physical_mappings.begin()) return ResultInvalidCurrentMemory;
        --it;
        if (!ContainsRange(it->first, it->second, src, size)) return ResultInvalidCurrentMemory;
        source_fd = physical_fd;
        source_offset = src - AliasBase;
    }
    if (map) {
        for (const auto& [address, m] : stack_mappings)
            if (Overlaps(dst, size, m.destination, m.size) || Overlaps(src, size, m.source, m.size))
                return ResultInvalidCurrentMemory;
        // Both mappings use the same pages. Protect the original heap address until unmap,
        // matching the SDK's ownership transfer of heap storage to an alias stack.
#if OOT_ANONYMOUS_MEMORY
        // No shared mappings: the source pages are inaccessible while mapped, so a copy is an
        // exact stand-in (copied back on unmap).
        (void)source_fd; (void)source_offset;
        if (mprotect(reinterpret_cast<void*>(dst), size, PROT_READ | PROT_WRITE) != 0) return ResultOutOfResource;
        std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size);
#else
        void* p = mmap(reinterpret_cast<void*>(dst), size, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FIXED, source_fd, source_offset);
        if (p == MAP_FAILED) return ResultOutOfResource;
#endif
        if (mprotect(reinterpret_cast<void*>(src), size, PROT_NONE) != 0)
            Fatal("cannot protect stack source pages");
        stack_mappings.emplace(dst, StackMapping{dst, src, size});
    } else {
        const auto it = stack_mappings.find(dst);
        if (it == stack_mappings.end() || it->second.source != src || it->second.size != size)
            return ResultInvalidCurrentMemory;
        // Replace only the mapping we own, retaining the reserved address range.
#if OOT_ANONYMOUS_MEMORY
        if (mprotect(reinterpret_cast<void*>(src), size, PROT_READ | PROT_WRITE) != 0)
            Fatal("cannot restore stack source pages");
        std::memcpy(reinterpret_cast<void*>(src), reinterpret_cast<void*>(dst), size);
        if (madvise(reinterpret_cast<void*>(dst), size, DiscardAdvice) != 0) return ResultOutOfResource;
#else
        void* p = mmap(reinterpret_cast<void*>(dst), size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) return ResultOutOfResource;
        if (mprotect(reinterpret_cast<void*>(src), size, PROT_READ | PROT_WRITE) != 0)
            Fatal("cannot restore stack source pages");
#endif
        stack_mappings.erase(it);
    }
    return ResultSuccess;
}

// Memory the process uses besides the heap: its modules (program size as the SDK computes it).
u64 ProgramSize() {
    u64 size = 0;
    for (const auto& m : Modules()) size += (m.size + 0xfff) & ~0xfffULL;
    return size;
}
constexpr u64 SystemResourceBudget = 0x4000000; // enable SDK VAMM metadata allocation (64 MiB)

u64 PhysicalMemorySize() {
    u64 size = 0;
    for (const auto& [base, length] : physical_mappings) size += length;
    return size;
}

// MapPhysicalMemory / UnmapPhysicalMemory: back (or release) pages of the reserved alias region.
u32 SvcMapPhysicalMemory(Cpu* c, bool map) {
    const u64 addr = c->x[0], size = c->x[1];
    if (addr & 0xfff) return ResultInvalidAddress;
    if (!size || (size & 0xfff)) return ResultInvalidSize;
    if (!ContainsRange(AliasBase, AliasSize, addr, size)) return ResultInvalidMemoryRegion;
    std::lock_guard lock{k_lock};
    for (const auto& [address, m] : stack_mappings)
        if (Overlaps(addr, size, m.source, m.size)) return ResultInvalidCurrentMemory;
    if (mprotect(reinterpret_cast<void*>(addr), size, map ? PROT_READ | PROT_WRITE : PROT_NONE) != 0)
        return ResultOutOfResource;
    if (!map && madvise(reinterpret_cast<void*>(addr), size, DiscardAdvice) != 0)
        Fatal("cannot discard unmapped physical pages");
    { static u64 mapped = 0; mapped = map ? mapped + size : mapped - std::min(mapped, size);
      static u64 shown = 0; if (shown++ < 20) RT_LOG("memory: physical %s %#llx+%#llx (%llu MiB mapped)", map ? "map" : "unmap", (unsigned long long)addr, (unsigned long long)size, (unsigned long long)(mapped >> 20)); }
    u64 begin = addr, end = addr + size;
    for (auto it = physical_mappings.begin(); it != physical_mappings.end();) {
        const u64 old_begin = it->first, old_end = old_begin + it->second;
        if (old_end < begin || old_begin > end) { ++it; continue; }
        it = physical_mappings.erase(it);
        if (map) {
            begin = std::min(begin, old_begin);
            end = std::max(end, old_end);
        } else {
            if (old_begin < begin) physical_mappings.emplace(old_begin, begin - old_begin);
            if (old_end > end) { physical_mappings.emplace(end, old_end - end); break; }
        }
    }
    if (map) physical_mappings.emplace(begin, end - begin);
    return 0;
}

struct Region {
    u64 base, size;
    u32 state, perm;
    u32 attr = 0;
};

} // namespace

// ---- threads -----------------------------------------------------------------------------------
struct ThreadExit {};

KThread* CurrentKThread() { return current_thread; }

// Per-core execution. On the console, threads sharing a core never run at the same time, and
// games rely on it (per-core job queues and scratch indexed by GetCurrentProcessorNumber, threads
// that pin themselves to a core to work on its data). A guest thread holds its core's lock while
// it runs guest code and gives it up for every SVC (each one a possible reschedule point).
// Enabled with OOT_CORE_LOCKS=1 (experimental).
namespace {
std::mutex core_locks[8];
thread_local int held_core = -1;
thread_local int svc_core = -1; // the core held when the current SVC was entered
bool CoreLocksEnabled() {
    // Off by default: without priority scheduling a busy thread can starve the others on its
    // core (the audio DSP threads run nearly continuously), which deadlocks guest mutex hand-offs.
    static const bool enabled = [] { const char* v = std::getenv("OOT_CORE_LOCKS"); return v && std::strcmp(v, "0") != 0; }();
    return enabled;
}
} // namespace
void EnterCore() {
    if (!CoreLocksEnabled() || !current_thread || held_core >= 0) return;
    const int ideal = std::clamp(current_thread->core, 0, 7);
    // A thread allowed on several cores takes any free one, its ideal core first.
    const u64 mask = current_thread->affinity & 0xff;
    if (mask & ~(1ULL << ideal)) {
        for (int i = 0; i < 8; ++i) {
            const int core = (ideal + i) % 8;
            if ((mask >> core & 1) && core_locks[core].try_lock()) { held_core = core; return; }
        }
    }
    core_locks[ideal].lock();
    held_core = ideal;
}
// The core the current thread runs on (its ideal core when not holding one).
s32 CurrentCore() {
    if (held_core >= 0) return held_core;
    if (svc_core >= 0) return svc_core;
    return current_thread ? std::max(current_thread->core, 0) : 0;
}
bool HoldsCore() { return held_core >= 0; }
void LeaveCore() {
    if (held_core < 0) return;
    core_locks[held_core].unlock();
    held_core = -1;
}

void BindThread(KThread* t) {
    current_thread = t;
    BindCpu(t->cpu);
    EnterCore();
}

namespace {

void RunThread(std::shared_ptr<KThread> t) {
    BindThread(t.get());
    { // host name: guest thread id and entry (profiles and debuggers show it)
        char name[16]; std::snprintf(name, sizeof name, "g%u-%llx", t->id, (unsigned long long)(t->entry & 0xffffff));
        rt::SetThreadName(name);
    }
    try {
        CallGuest(t->cpu, t->entry, {t->arg, t->handle});
    } catch (const ThreadExit&) {
    }
    LeaveCore();
    std::lock_guard lock{k_lock};
    t->exited = true;
    NotifySync();
}

u32 SvcCreateThread(Cpu* c) {
    auto t = std::make_shared<KThread>();
    t->kind = "thread";
    t->entry = c->x[1];
    t->arg = c->x[2];
    t->stack_top = c->x[3];
    t->priority = (s32)c->x[4];
    t->core = (s32)c->x[5] == -2 ? 0 : (s32)c->x[5];
    t->id = next_thread_id++;
    t->tlr = ArenaAlloc(0x200, 0x200);
    t->cpu = CreateCpu(t->stack_top, t->tlr);
    std::lock_guard lock{k_lock};
    t->handle = AddHandle(t);
    *reinterpret_cast<u32*>(t->tlr + 0x110) = t->handle; // current thread handle, read by the SDK
    static const bool trace = std::getenv("OOT_THREAD_TRACE") != nullptr;
    if (trace) RT_LOG("thread %u created: entry %#llx arg %#llx priority %d core %d", t->id, (unsigned long long)t->entry, (unsigned long long)t->arg, t->priority, (s32)c->x[5]);
    c->x[1] = t->handle;
    return ResultSuccess;
}

u32 SvcStartThread(Cpu* c) {
    std::shared_ptr<KThread> t;
    {
        std::lock_guard lock{k_lock};
        t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[0]));
    }
    if (!t) return ResultInvalidHandle;
    t->host = std::thread(RunThread, t);
    t->host.detach();
    return ResultSuccess;
}

u32 SvcWaitSynchronization(Cpu* c) {
    const u32* list = reinterpret_cast<const u32*>(c->x[1]);
    const s32 count = (s32)c->x[2];
    const s64 timeout = (s64)c->x[3];
    std::vector<std::shared_ptr<KObject>> objects;
    std::unique_lock lock{k_lock};
    for (s32 i = 0; i < count; ++i) {
        auto obj = GetObject(list[i]);
        if (!obj) return ResultInvalidHandle;
        objects.push_back(obj);
    }
    const auto deadline = Deadline(timeout);
    Waiter w; w.lr = c->x[30]; w.thread = current_thread; w.count = count;
    for (s32 i = 0; i < count && i < 4; ++i) w.handles[i] = list[i];
    struct SyncTrace { SyncTrace(std::list<SyncWaitInfo>::iterator i, bool on) : it(i), active(on) {} ~SyncTrace() { if (active) sync_infos.erase(it); } std::list<SyncWaitInfo>::iterator it; bool active; };
    static const bool trace_waits = std::getenv("OOT_ARB_TRACE") != nullptr;
    std::list<SyncWaitInfo>::iterator info_it{};
    if (trace_waits && timeout != 0) {
        std::string kinds; for (auto& o : objects) { kinds += o->kind; if (auto t = std::dynamic_pointer_cast<KThread>(o)) kinds += "#" + std::to_string(t->id); kinds += ' '; }
        info_it = sync_infos.insert(sync_infos.end(), {current_thread ? current_thread->id : 0, c->x[30], kinds, Clock::now()});
        StartArbiterWatchdog();
    }
    SyncTrace sync_trace{info_it, trace_waits && timeout != 0};
    const u64 me = current_thread ? current_thread->id : 0;
    NoteThreadEvent(me, "wait %d handles timeout %lld (lr %#llx) cancel flag %d", count, (long long)timeout, (unsigned long long)c->x[30], current_thread ? (int)current_thread->cancel_wait : -1);
    struct ExitNote { u64 me; Cpu* c; ~ExitNote() { NoteThreadEvent(me, "wait returned (x1 %#llx)", (unsigned long long)c->x[1]); } } exit_note{me, c};
    for (;;) {
        for (s32 i = 0; i < count; ++i) {
            if (objects[i]->Signaled()) {
                c->x[1] = (u64)i;
                return ResultSuccess;
            }
        }
        if (timeout == 0) return ResultTimedOut;
        if (current_thread && current_thread->cancel_wait) {
            current_thread->cancel_wait = false;
            return ResultCancelled;
        }
        w.woken = false;
        sync_waiters.push_back(&w);
        const bool ok = Block(lock, w, deadline);
        sync_waiters.remove(&w);
        if (!ok) return ResultTimedOut;
    }
}

u32 SvcArbitrateLockImpl(Cpu* c) {
    const u32 owner = (u32)c->x[0];
    const u64 addr = c->x[1];
    const u32 tag = (u32)c->x[2];
    std::unique_lock lock{k_lock};
    if (__atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST) != (owner | HandleWaitMask)) return ResultSuccess;
    Waiter w; w.lr = c->x[30]; w.thread = current_thread;
    lock_waiters.push_back({addr, tag, &w});
    StartHangWatch();
    Block(lock, w, Clock::time_point::max());
    return ResultSuccess;
}

u32 SvcArbitrateUnlockImpl(Cpu* c) {
    std::lock_guard lock{k_lock};
    ReleaseMutex(c->x[0]);
    return ResultSuccess;
}

u32 SvcWaitProcessWideKeyAtomic(Cpu* c) {
    const u64 addr = c->x[0], key = c->x[1];
    const u32 tag = (u32)c->x[2];
    const s64 timeout = (s64)c->x[3];
    std::unique_lock lock{k_lock};
    NoteArbiterOp(4, 0, addr, (s64)key, (s64)tag, __atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST), -1, c->x[30]);
    // The "has waiters" key is set BEFORE the mutex is released, as the console kernel does: the
    // SDK's Signal reads the key in user mode and skips the SVC when it is 0, so a signaller that
    // took the released mutex before the key was set lost the wakeup for good (boot hangs in the
    // movie player's NuCachedSource2/ALooper hand-off).
    Waiter w; w.lr = c->x[30]; w.thread = current_thread;
    key_waiters.push_back({key, addr, tag, &w});
    __atomic_store_n(U32At(key), 1u, __ATOMIC_SEQ_CST);
    ReleaseMutex(addr);
    const bool ok = Block(lock, w, Deadline(timeout));
    if (!ok) {
        // Timed out while still waiting on the key (a signal would have woken or queued us).
        auto it = std::find_if(key_waiters.begin(), key_waiters.end(), [&](auto& kw) { return kw.w == &w; });
        if (it != key_waiters.end()) {
            key_waiters.erase(it);
            return ResultTimedOut;
        }
        // Signalled and queued on the mutex: wait for the hand-off.
        auto lw = std::find_if(lock_waiters.begin(), lock_waiters.end(), [&](auto& l) { return l.w == &w; });
        if (lw != lock_waiters.end()) Block(lock, w, Clock::time_point::max());
    }
    return ResultSuccess;
}

u32 SvcSignalProcessWideKey(Cpu* c) {
    const u64 key = c->x[0];
    const s32 count = (s32)c->x[1];
    std::lock_guard lock{k_lock};
    s32 n = 0;
    for (auto it = key_waiters.begin(); it != key_waiters.end() && (count <= 0 || n < count);) {
        if (it->key != key) { ++it; continue; }
        const KeyWaiter kw = *it;
        it = key_waiters.erase(it);
        ++n;
        const u32 before = __atomic_load_n(U32At(kw.addr), __ATOMIC_SEQ_CST);
        if (AcquireOrQueue(kw.addr, kw.tag, kw.w)) {
            kw.w->woken = true;
            kw.w->cv.notify_one();
            NoteArbiterOp(5, 1, kw.addr, (s64)key, (s64)kw.tag, before, 0, c->x[30]);
        } else {
            kw.w->woken = false; // now waits for the mutex hand-off
            NoteArbiterOp(5, 0, kw.addr, (s64)key, (s64)kw.tag, before, 0, c->x[30]);
        }
    }
    if (std::none_of(key_waiters.begin(), key_waiters.end(), [&](auto& kw) { return kw.key == key; }))
        __atomic_store_n(U32At(key), 0u, __ATOMIC_SEQ_CST);
    return ResultSuccess;
}

u32 SvcWaitForAddressImpl(Cpu* c) {
    const u64 addr = c->x[0];
    const u32 type = (u32)c->x[1];
    const s64 timeout = (s64)c->x[3];
    std::unique_lock lock{k_lock};
    switch (type) {
    case 0: // WaitIfLessThan
        if (!((s32)__atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST) < (s32)c->x[2])) return ResultInvalidState;
        break;
    case 1: { // DecrementAndWaitIfLessThan
        // Atomic: guest threads change the word with exclusive accesses without the kernel lock,
        // and a separate load and store lost their updates (job-system hangs).
        u32 v = __atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST);
        do {
            if (!((s32)v < (s32)c->x[2])) return ResultInvalidState;
        } while (!__atomic_compare_exchange_n(U32At(addr), &v, v - 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
        break;
    }
    case 2: // WaitIfEqual
        if (__atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST) != (u32)c->x[2]) return ResultInvalidState;
        break;
    case 3: // WaitIfEqual64
        if (__atomic_load_n(reinterpret_cast<u64*>(addr), __ATOMIC_SEQ_CST) != c->x[2]) return ResultInvalidState;
        break;
    default:
        Fatal("WaitForAddress: arbitration type %u", type);
    }
    if (timeout == 0) return ResultTimedOut;
    Waiter w; w.lr = c->x[30]; w.thread = current_thread;
    address_waiters.push_back({addr, &w, type, c->x[2], c->x[30], current_thread ? current_thread->id : 0, Clock::now()});
    StartArbiterWatchdog();
    if (!Block(lock, w, Deadline(timeout))) {
        address_waiters.remove_if([&](auto& a) { return a.w == &w; });
        return ResultTimedOut;
    }
    return ResultSuccess;
}

u32 SvcSignalToAddressImpl(Cpu* c) {
    const u64 addr = c->x[0];
    const u32 type = (u32)c->x[1];
    const s32 value = (s32)c->x[2];
    const s32 count = (s32)c->x[3];
    std::lock_guard lock{k_lock};
    switch (type) {
    case 0: // Signal
        break;
    case 1: { // SignalAndIncrementIfEqual
        u32 expected = (u32)value;
        if (!__atomic_compare_exchange_n(U32At(addr), &expected, (u32)(value + 1), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return ResultInvalidState;
        break;
    }
    case 2: { // SignalAndModifyByWaitingCountIfEqual
        const s32 waiters = CountAddressWaiters(addr);
        s32 new_value;
        if (count <= 0) new_value = waiters > 0 ? value - 2 : value + 1;
        else if (waiters > 0) new_value = (waiters - 1) < count ? value - 1 : value;
        else new_value = value + 1;
        u32 expected = (u32)value;
        if (new_value != value) {
            if (!__atomic_compare_exchange_n(U32At(addr), &expected, (u32)new_value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
                return ResultInvalidState;
        } else if (__atomic_load_n(U32At(addr), __ATOMIC_SEQ_CST) != (u32)value) {
            return ResultInvalidState;
        }
        break;
    }
    default:
        Fatal("SignalToAddress: signal type %u", type);
    }
    WakeAddress(addr, count);
    return ResultSuccess;
}

u32 SvcArbitrateLock(Cpu* c) {
    const u64 addr = c->x[1], lr = c->x[30];
    const u32 before = __atomic_load_n(reinterpret_cast<u32*>(addr), __ATOMIC_SEQ_CST);
    { std::lock_guard lock{k_lock}; NoteArbiterOp(2, 0, addr, (s64)c->x[0], (s64)c->x[2], before, -1, lr); }
    const u32 r = SvcArbitrateLockImpl(c);
    { std::lock_guard lock{k_lock}; NoteArbiterOp(2, 1, addr, (s64)c->x[0], (s64)c->x[2], before, (s32)r, lr); }
    return r;
}
u32 SvcArbitrateUnlock(Cpu* c) {
    const u64 addr = c->x[0], lr = c->x[30];
    const u32 before = __atomic_load_n(reinterpret_cast<u32*>(addr), __ATOMIC_SEQ_CST);
    const u32 r = SvcArbitrateUnlockImpl(c);
    { std::lock_guard lock{k_lock}; NoteArbiterOp(3, 0, addr, 0, 0, before, (s32)r, lr); }
    return r;
}
u32 SvcWaitForAddress(Cpu* c) {
    const u64 addr = c->x[0], lr = c->x[30]; const u32 type = (u32)c->x[1]; const s64 value = (s64)c->x[2];
    const u32 before = __atomic_load_n(reinterpret_cast<u32*>(addr), __ATOMIC_SEQ_CST);
    { std::lock_guard lock{k_lock}; NoteArbiterOp(0, type, addr, value, -1, before, -1, lr); }
    const u32 r = SvcWaitForAddressImpl(c);
    { std::lock_guard lock{k_lock}; NoteArbiterOp(0, type, addr, value, -2, before, (s32)r, lr); }
    return r;
}
u32 SvcSignalToAddress(Cpu* c) {
    const u64 addr = c->x[0], lr = c->x[30]; const u32 type = (u32)c->x[1];
    const u32 before = __atomic_load_n(reinterpret_cast<u32*>(addr), __ATOMIC_SEQ_CST);
    const u32 r = SvcSignalToAddressImpl(c);
    { std::lock_guard lock{k_lock}; NoteArbiterOp(1, type, addr, (s64)(s32)c->x[2], (s64)(s32)c->x[3], before, (s32)r, lr); }
    return r;
}

u32 SvcCreateEvent(Cpu* c) {
    auto e = std::make_shared<KEvent>();
    e->kind = "event";
    std::lock_guard lock{k_lock};
    c->x[1] = AddHandle(e);
    c->x[2] = AddHandle(e);
    return ResultSuccess;
}

u32 SetEvent(u32 handle, bool value) {
    std::lock_guard lock{k_lock};
    auto e = std::dynamic_pointer_cast<KEvent>(GetObject(handle));
    if (!e) return ResultInvalidHandle;
    e->signaled = value;
    if (value) NotifySync();
    return ResultSuccess;
}

u32 SvcGetInfo(Cpu* c) {
    const u32 type = (u32)c->x[1];
    const u64 sub = c->x[3];
    u64 v = 0;
    std::lock_guard lock{k_lock};
    switch (type) {
    case 0: v = 0xff; break;                            // core mask
    case 1: v = 0xfffffffffffffff0ULL; break;            // priority mask
    case 2: v = AliasBase; break;                        // alias region
    case 3: v = AliasSize; break;
    case 4: v = HeapBase; break;                         // heap region
    case 5: v = HeapReserve; break;
    case 6: v = 0x300000000ULL; break;                   // total memory (12 GiB)
    case 7: v = heap_size + ProgramSize() + PhysicalMemorySize(); break;
    case 8: v = 0; break;                                // debugger attached
    case 11: v = 0x0123456789abcdefULL ^ sub; break;     // random entropy
    case 12: v = 0x8000000ULL; break;                    // ASLR region
    case 13: v = 0x7ff8000000ULL; break;
    case 14: v = StackBase; break;                       // stack region
    case 15: v = StackSize; break;
    case 16: v = SystemResourceBudget; break;            // nonzero enables the SDK VAMM manager
    case 17: v = 0; break;                              // host manages metadata outside guest memory
    case 18: v = 0x0400486024DEC000ULL; break;           // program id
    case 20: { // The SDK stores dying-message metadata in the process exception context.
        static const u64 exception_context = ArenaAlloc(0x1000, 0x1000);
        v = exception_context; break;
    }
    case 21: v = 0x300000000ULL; break;
    case 22: v = heap_size + ProgramSize() + PhysicalMemorySize(); break;
    case 23: v = 1; break;                               // is application
    case 24: v = 0x100; break;                           // free thread count
    case 25: v = rt_cntpct(); break;                     // thread tick count
    case 10: v = 0; break;                               // idle tick count
    case 26: v = 1; break;                               // is SVC permitted
    case 27: case 28: case 34: case 37: case 38: v = 0; break;
    case 29: v = AliasBase + AliasSize / 2; break;       // (Switch 2) VAMM region: upper half of alias
    case 30: v = AliasSize / 2; break;
    case 31: case 32: v = 0; break;                      // (Switch 2) remote memory usage
    case 33: v = 0x1000; break;                          // (Switch 2) page granule
    default:
        RT_LOG("GetInfo: unknown type %u sub %#llx", type, (unsigned long long)sub);
        return 0xF001;
    }
    c->x[1] = v;
    return ResultSuccess;
}

u32 SvcSetHeapSize(Cpu* c) {
    const u64 size = c->x[1];
    std::lock_guard lock{k_lock};
    if (size > HeapReserve || (size & 0xfff)) return ResultInvalidSize;
    for (const auto& [address, m] : stack_mappings)
        if (m.source < HeapBase + HeapReserve && m.source + m.size > HeapBase + size)
            return ResultInvalidCurrentMemory;
    if (!heap_reserved) {
#if OOT_ANONYMOUS_MEMORY
        void* p = mmap(reinterpret_cast<void*>(HeapBase), HeapReserve, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
#else
        heap_fd = memfd_create("tootr-heap", MFD_CLOEXEC);
        if (heap_fd < 0 || ftruncate(heap_fd, HeapReserve) != 0) Fatal("cannot create shared heap backing");
        void* p = mmap(reinterpret_cast<void*>(HeapBase), HeapReserve, PROT_NONE,
                       MAP_SHARED | MAP_FIXED_NOREPLACE | MAP_NORESERVE, heap_fd, 0);
#endif
        if (p != reinterpret_cast<void*>(HeapBase)) Fatal("cannot reserve heap");
#ifdef __linux__
        // (no transparent huge pages: the renderer imports guest pages into the GPU (userptr on amdgpu), and a huge
        // page split or merged invalidates the imports, pinned again at the next submission; from Hookshot, issue 312)
        if (!std::getenv("OOT_THP")) madvise(reinterpret_cast<void*>(HeapBase), HeapReserve, MADV_NOHUGEPAGE);
#endif
        heap_reserved = true;
    }
    if (size > heap_size) {
        if (mprotect(reinterpret_cast<void*>(HeapBase + heap_size), size - heap_size, PROT_READ | PROT_WRITE) != 0)
            return ResultOutOfResource;
    } else if (size < heap_size) {
        if (madvise(reinterpret_cast<void*>(HeapBase + size), heap_size - size, DiscardAdvice) != 0 ||
            mprotect(reinterpret_cast<void*>(HeapBase + size), heap_size - size, PROT_NONE) != 0)
            return ResultOutOfResource;
    }
    heap_size = size;
    RT_LOG("memory: heap %llu MiB", (unsigned long long)(size >> 20));
    c->x[1] = HeapBase;
    return ResultSuccess;
}

// MemoryInfo { u64 base, size; u32 state, attr, perm, ipc_refcount, device_refcount, pad; }
u32 SvcQueryMemory(Cpu* c) {
    const u64 addr = c->x[2];
    std::lock_guard lock{k_lock};
    std::vector<Region> regions;
    for (const auto& m : Modules()) regions.push_back({m.base, (m.size + 0xfff) & ~0xfffULL, 3, 5});
    // Split the heap at locked stack source ranges so SDK memory queries see the
    // correct permissions, and neighbouring free heap pages remain allocatable.
    std::vector<StackMapping> sources;
    for (const auto& [address, m] : stack_mappings) {
        regions.push_back({m.destination, m.size, 0xb, 3});
        sources.push_back(m);
    }
    std::sort(sources.begin(), sources.end(), [](auto& a, auto& b) { return a.source < b.source; });
    const auto normal_range = [&](u64 base, u64 size) {
        u64 cursor = base;
        for (const auto& m : sources) {
            if (!ContainsRange(base, size, m.source, m.size)) continue;
            if (m.source > cursor) regions.push_back({cursor, m.source - cursor, 5, 3});
            regions.push_back({m.source, m.size, 5, 0, 1}); // MemoryAttribute::Locked
            cursor = m.source + m.size;
        }
        if (cursor < base + size) regions.push_back({cursor, base + size - cursor, 5, 3});
    };
    normal_range(HeapBase, heap_size);
    regions.push_back({ArenaBase, ArenaSize, 0xb, 3});
    for (const auto& [base, size] : physical_mappings) normal_range(base, size);
    std::sort(regions.begin(), regions.end(), [](auto& a, auto& b) { return a.base < b.base; });
    Region found{0, 0, 0, 0};
    u64 prev_end = 0;
    for (const auto& r : regions) {
        if (addr < r.base) { found = {prev_end, r.base - prev_end, 0, 0}; break; }
        if (addr < r.base + r.size) { found = r; break; }
        prev_end = r.base + r.size;
    }
    if (!found.size) found = {prev_end, (1ULL << 39) - prev_end, 0, 0};
    u8* out = reinterpret_cast<u8*>(c->x[0]);
    std::memset(out, 0, 0x28);
    std::memcpy(out, &found.base, 8);
    std::memcpy(out + 8, &found.size, 8);
    std::memcpy(out + 16, &found.state, 4);
    std::memcpy(out + 20, &found.attr, 4);
    std::memcpy(out + 24, &found.perm, 4);
    c->x[1] = 0;
    return ResultSuccess;
}

} // namespace

void CreateKernelEvent(u32* writable, u32* readable) {
    auto e = std::make_shared<KEvent>();
    e->kind = "event";
    std::lock_guard lock{k_lock};
    *writable = AddHandle(e);
    *readable = AddHandle(e);
}

void SignalKernelEvent(u32 handle) { SetEvent(handle, true); }
void ClearKernelEvent(u32 handle) { SetEvent(handle, false); }

void InitializeKernel() {
#if OOT_ANONYMOUS_MEMORY
    void* p = mmap(reinterpret_cast<void*>(AliasBase), AliasSize, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
#else
    physical_fd = memfd_create("tootr-physical", MFD_CLOEXEC);
    if (physical_fd < 0 || ftruncate(physical_fd, AliasSize) != 0) Fatal("cannot create physical memory backing");
    void* p = mmap(reinterpret_cast<void*>(AliasBase), AliasSize, PROT_NONE,
                   MAP_SHARED | MAP_FIXED_NOREPLACE | MAP_NORESERVE, physical_fd, 0);
#endif
    if (p != reinterpret_cast<void*>(AliasBase)) Fatal("cannot reserve the alias region");
#ifdef __linux__
    if (!std::getenv("OOT_THP")) madvise(reinterpret_cast<void*>(AliasBase), AliasSize, MADV_NOHUGEPAGE); // (as the heap's)
#endif
    p = mmap(reinterpret_cast<void*>(StackBase), StackSize, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
    if (p != reinterpret_cast<void*>(StackBase)) Fatal("cannot reserve the stack region");
}

KThread* CreateMainThread(u64 stack_top) {
    auto t = std::make_shared<KThread>();
    t->kind = "thread";
    t->id = next_thread_id++;
    t->tlr = ArenaAlloc(0x200, 0x200);
    t->cpu = CreateCpu(stack_top, t->tlr);
    t->priority = 44;
    std::lock_guard lock{k_lock};
    t->handle = AddHandle(t);
    *reinterpret_cast<u32*>(t->tlr + 0x110) = t->handle;
    return t.get();
}

void ExitCurrentThread() { throw ThreadExit{}; }

} // namespace rt

using namespace rt;

void rt_svc(Cpu* c, u32 imm) {
    // Leave the core for the call; re-enter (possibly a new core after SetThreadCoreMask) after it.
    struct CoreYield {
        bool held = held_core >= 0;
        CoreYield() { svc_core = held_core; LeaveCore(); }
        ~CoreYield() { svc_core = -1; if (held) EnterCore(); }
    } core_yield;
    static const bool trace = std::getenv("OOT_TRACE_SVC") != nullptr;
    if (trace)
        std::fprintf(stderr, "[svc] %#x(%#llx, %#llx, %#llx, %#llx) from %s\n", imm, (unsigned long long)c->x[0],
                     (unsigned long long)c->x[1], (unsigned long long)c->x[2], (unsigned long long)c->x[3],
                     DescribeAddress(c->x[30]).c_str());
    u32 result = ResultSuccess;
    const u64 sync_arg = imm == 0x18 ? c->x[2] : c->x[imm == 0x1A ? 1 : 0]; // (hang report: handle count, mutex or key address, handle)
    if (imm >= 0x18 && imm <= 0x1D && current_thread) current_thread->NoteSync(imm | 0x100, sync_arg, 0, c->x[30]); // entry
    switch (imm) {
    case 0x01: result = SvcSetHeapSize(c); break;
    case 0x04: result = SvcMapMemory(c, true); break;
    case 0x05: result = SvcMapMemory(c, false); break;
    case 0x06: result = SvcQueryMemory(c); break;
    case 0x08: result = SvcCreateThread(c); break;
    case 0x09: result = SvcStartThread(c); break;
    case 0x0A: ExitCurrentThread(); break;
    case 0x0B: { // SleepThread
        const s64 ns = (s64)c->x[0];
        static const bool trace = std::getenv("OOT_SLEEP_TRACE") != nullptr;
        if (trace) { static std::mutex m; static std::map<std::pair<u64, s64>, u64> seen; static u64 n = 0; std::lock_guard l{m};
            ++seen[{c->x[30], ns < 1000000 ? ns : ns / 100000 * 100000}];
            if (++n % 3000 == 0) { for (auto& [k, v] : seen) if (v > 20) RT_LOG("sleep %.4f ms from %s: %llu", k.second / 1e6, DescribeAddress(k.first).c_str(), (unsigned long long)v); seen.clear(); } }
        if (ns <= 0) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
        return;
    }
    case 0x0C: { // GetThreadPriority
        std::lock_guard lock{k_lock};
        auto t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[1]));
        c->x[1] = t ? (u64)t->priority : 44;
        break;
    }
    case 0x0D: { // SetThreadPriority
        std::lock_guard lock{k_lock};
        if (auto t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[0]))) t->priority = (s32)c->x[1];
        break;
    }
    // Cores: games index per-core scratch buffers by the current processor, so every thread
    // reports its own ideal core (-2 = process default, core 0).
    case 0x0E: { // GetThreadCoreMask(handle) -> ideal core, affinity mask
        std::lock_guard lock{k_lock};
        auto t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[2]));
        c->x[1] = t ? (u64)(u32)std::max(t->core, 0) : 0;
        c->x[2] = t ? (1ULL << std::max(t->core, 0)) : 1;
        break;
    }
    case 0x0F: { // SetThreadCoreMask(handle, ideal core, affinity mask)
        std::lock_guard lock{k_lock};
        if (auto t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[0]))) {
            const s32 ideal = (s32)c->x[1];
            const u64 mask = c->x[2];
            if (ideal >= 0) t->core = ideal;
            else if (ideal == -2) t->core = 0;
            if (mask && !(mask >> std::max(t->core, 0) & 1)) t->core = __builtin_ctzll(mask);
            if (mask) t->affinity = mask;
            static const bool trace = std::getenv("OOT_THREAD_TRACE") != nullptr;
            if (trace) RT_LOG("thread %u core mask: ideal %d mask %#llx -> core %d", t->id, ideal, (unsigned long long)mask, t->core);
        }
        break;
    }
    case 0x10: c->x[0] = (u64)CurrentCore(); return; // GetCurrentProcessorNumber
    case 0x11: result = SetEvent((u32)c->x[0], true); break;
    case 0x12: case 0x17: result = SetEvent((u32)c->x[0], false); break;
    case 0x16: { // CloseHandle
        std::lock_guard lock{k_lock};
        handles.erase((u32)c->x[0]);
        break;
    }
    case 0x18: result = SvcWaitSynchronization(c); break;
    case 0x19: { // CancelSynchronization
        std::lock_guard lock{k_lock};
        if (auto t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[0]))) {
            t->cancel_wait = true;
            t->cancels++; t->cancelled_by = current_thread ? current_thread->id : 0; t->cancel_lr = c->x[30]; t->cancel_time = Clock::now();
            t->NoteSync(0x219, t->cancelled_by, t->cancel_wait, c->x[30]); // (received: by thread, flag)
            NoteThreadEvent(t->id, "cancelled by thread %llu (lr %#llx)", (unsigned long long)(current_thread ? current_thread->id : 0), (unsigned long long)c->x[30]);
        }
        NotifySync();
        break;
    }
    case 0x1A: result = SvcArbitrateLock(c); break;
    case 0x1B: result = SvcArbitrateUnlock(c); break;
    case 0x1C: result = SvcWaitProcessWideKeyAtomic(c); break;
    case 0x1D: result = SvcSignalProcessWideKey(c); break;
    case 0x1E: c->x[0] = rt_cntpct(); return;
    case 0x24: c->x[1] = 0x51; break; // GetProcessId
    case 0x25: { // GetThreadId
        std::lock_guard lock{k_lock};
        auto t = std::dynamic_pointer_cast<KThread>(GetObject((u32)c->x[1]));
        if (!t) { result = ResultInvalidHandle; break; }
        c->x[1] = t->id;
        break;
    }
    case 0x26: // Break
        Fatal("guest Break(reason %#llx, %#llx, %#llx) from %s", (unsigned long long)c->x[0],
              (unsigned long long)c->x[1], (unsigned long long)c->x[2], DescribeAddress(c->x[30]).c_str());
    case 0x27: { // OutputDebugString
        std::fprintf(stderr, "[guest] %.*s\n", (int)c->x[1], reinterpret_cast<const char*>(c->x[0]));
        break;
    }
    case 0x29: result = SvcGetInfo(c); break;
    case 0x2C: result = SvcMapPhysicalMemory(c, true); break;
    case 0x2D: result = SvcMapPhysicalMemory(c, false); break;
    case 0x34: result = SvcWaitForAddress(c); break;
    case 0x35: result = SvcSignalToAddress(c); break;
    case 0x36: break; // SynchronizePreemptionState
    case 0x45: result = SvcCreateEvent(c); break;
    default:
        Fatal("SVC %#x reached in %s (lr %s): this SDK path needs a native implementation", imm,
              DescribeAddress(c->pc).c_str(), DescribeAddress(c->x[30]).c_str());
    }
    c->x[0] = result;
    if (trace) std::fprintf(stderr, "[svc]   -> %#x, x1=%#llx x2=%#llx\n", result, (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
    if (imm >= 0x18 && imm <= 0x1D && current_thread) current_thread->NoteSync(imm, sync_arg, result, c->x[30]);
}
