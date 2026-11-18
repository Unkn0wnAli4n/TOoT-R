#include "platform.h"

#include <chrono>
#include <thread>

#include <cstdlib>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <string>

namespace {

DWORD WinProtect(int prot) {
    if (prot & PROT_EXEC) return (prot & PROT_WRITE) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    if (prot & PROT_WRITE) return PAGE_READWRITE;
    if (prot & PROT_READ) return PAGE_READONLY;
    return PAGE_NOACCESS;
}

// Calls fn(base, size, info) for each run of [addr, addr + len) that lies in one allocation and one
// state (VirtualProtect and VirtualFree must not cross allocations).
template <class F> bool ForEachRegion(void* addr, size_t len, F&& fn) {
    auto p = static_cast<std::uint8_t*>(addr);
    auto* const end = p + len;
    while (p < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery(p, &info, sizeof info)) return false;
        auto* const region_end = static_cast<std::uint8_t*>(info.BaseAddress) + info.RegionSize;
        auto* const run_end = region_end < end ? region_end : end;
        if (!fn(p, (size_t)(run_end - p), info)) return false;
        p = run_end;
    }
    return true;
}

} // namespace

void* mmap(void* addr, size_t len, int prot, int flags, int fd, long long offset) {
    (void)offset;
    if (fd != -1 || !(flags & MAP_ANONYMOUS) || (flags & MAP_SHARED)) { errno = ENOTSUP; return MAP_FAILED; }
    const DWORD type = MEM_RESERVE | (prot == PROT_NONE ? 0 : MEM_COMMIT);
    void* p = VirtualAlloc((flags & (MAP_FIXED | MAP_FIXED_NOREPLACE)) ? addr : nullptr, len, type, WinProtect(prot));
    if (!p) { errno = ENOMEM; return MAP_FAILED; }
    return p;
}

int munmap(void* addr, size_t len) {
    (void)len;
    return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : (errno = EINVAL, -1);
}

int mprotect(void* addr, size_t len, int prot) {
    DWORD old;
    if (VirtualProtect(addr, len, WinProtect(prot), &old)) return 0; // (the common case: committed pages)
    const bool ok = ForEachRegion(addr, len, [&](void* p, size_t n, const MEMORY_BASIC_INFORMATION& info) {
        if (info.State == MEM_FREE) return false;
        if (prot == PROT_NONE) return info.State != MEM_COMMIT || VirtualProtect(p, n, PAGE_NOACCESS, &old) != 0;
        return VirtualAlloc(p, n, MEM_COMMIT, WinProtect(prot)) != nullptr &&
               VirtualProtect(p, n, WinProtect(prot), &old) != 0;
    });
    return ok ? 0 : (errno = ENOMEM, -1);
}

namespace rt { void LazyForget(void* addr, size_t len); }
int madvise(void* addr, size_t len, int advice) {
    if (advice != MADV_REMOVE) return 0;
    rt::LazyForget(addr, len); // (decommitted below: no longer known to be usable)
    const bool ok = ForEachRegion(addr, len, [](void* p, size_t n, const MEMORY_BASIC_INFORMATION& info) {
        return info.State != MEM_COMMIT || VirtualFree(p, n, MEM_DECOMMIT) != 0;
    });
    return ok ? 0 : (errno = EINVAL, -1);
}

namespace rt {
namespace {
std::atomic<std::uint64_t> lazy_base{0}, lazy_end{0};
constexpr std::uint64_t LazyChunk = 64u << 10; // (a megabyte at a time held 16 GiB committed for 6 GiB in use: the heap is touched sparsely)
// Commits the reserved pages of [from, to) (pages already committed keep their protection: only reserved runs are touched).
bool CommitReserved(std::uint64_t from, std::uint64_t to) {
    auto* p = reinterpret_cast<std::uint8_t*>(from); auto* const end = reinterpret_cast<std::uint8_t*>(to);
    while (p < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery(p, &info, sizeof info)) return false;
        auto* const region_end = static_cast<std::uint8_t*>(info.BaseAddress) + info.RegionSize;
        auto* const run_end = region_end < end ? region_end : end;
        if (info.State == MEM_RESERVE && !VirtualAlloc(p, (size_t)(run_end - p), MEM_COMMIT, PAGE_READWRITE)) return false;
        if (info.State == MEM_FREE) return false;
        p = run_end;
    }
    return true;
}
// Which 64 KiB steps of the range are known to be committed whole: one bit each, set when a step is committed and
// never by guessing. EnsureCommitted is called before every write the renderer makes into the game's memory -
// thousands a frame - and asked the system each time (VirtualQuery): 60% of the renderer thread's time on a
// reporter's PC, 16 frames a second where the graphics card had room for 40. Now it reads a bit.
constexpr std::uint64_t LazyMaxChunks = (32ull << 30) / LazyChunk;
std::atomic<std::uint64_t> lazy_known[LazyMaxChunks / 64];
bool LazyKnown(std::uint64_t chunk) { return chunk < LazyMaxChunks && (lazy_known[chunk / 64].load(std::memory_order_relaxed) >> (chunk % 64) & 1); }
void LazyMark(std::uint64_t chunk) { if (chunk < LazyMaxChunks) lazy_known[chunk / 64].fetch_or(1ull << (chunk % 64), std::memory_order_relaxed); }
LONG CALLBACK LazyHandler(EXCEPTION_POINTERS* e) {
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    return CommitOnFault(reinterpret_cast<void*>(e->ExceptionRecord->ExceptionInformation[1]), e->ExceptionRecord->ExceptionInformation[0] == 1) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
} // namespace
void LazyCommitRange(void* base, size_t size) {
    static bool installed = false;
    if (!installed) { installed = true; AddVectoredExceptionHandler(1, LazyHandler); }
    lazy_end.store(0); lazy_base.store(reinterpret_cast<std::uint64_t>(base)); lazy_end.store(reinterpret_cast<std::uint64_t>(base) + size);
}
bool CommitOnFault(void* address, bool write) {
    const std::uint64_t a = reinterpret_cast<std::uint64_t>(address), base = lazy_base.load(), end = lazy_end.load();
    if (a < base || a >= end) return false;
    MEMORY_BASIC_INFORMATION info;
    if (!VirtualQuery(address, &info, sizeof info)) return false;
    // (committed by now: another thread touched the same megabyte between this thread's fault and here. The access
    // goes again where the page allows it; a write to a read-only page is a fault of another kind, the page tracker's.)
    if (info.State == MEM_COMMIT) return info.Protect == PAGE_READWRITE || (!write && info.Protect == PAGE_READONLY);
    if (info.State != MEM_RESERVE) return false;
    const std::uint64_t from = std::max(base, a & ~(LazyChunk - 1)), to = std::min(end, (a & ~(LazyChunk - 1)) + LazyChunk);
    if (!CommitReserved(from, to)) return false;
    LazyMark(((a & ~(LazyChunk - 1)) - (base & ~(LazyChunk - 1))) / LazyChunk);
    return true;
}
void LazyForget(void* addr, size_t len) {
    const std::uint64_t a = reinterpret_cast<std::uint64_t>(addr), base = lazy_base.load(), end = lazy_end.load();
    if (!len || a + len <= base || a >= end) return;
    const std::uint64_t origin = base & ~(LazyChunk - 1);
    for (std::uint64_t c = (std::max(a, base) - origin) / LazyChunk, last = (std::min(a + len, end) - 1 - origin) / LazyChunk; c <= last && c < LazyMaxChunks; ++c)
        lazy_known[c / 64].fetch_and(~(1ull << (c % 64)), std::memory_order_relaxed);
}
void EnsureCommitted(const void* address, size_t size) {
    const std::uint64_t a = reinterpret_cast<std::uint64_t>(address), base = lazy_base.load(), end = lazy_end.load();
    if (!size || a + size <= base || a >= end) return;
    const std::uint64_t origin = base & ~(LazyChunk - 1);
    for (std::uint64_t c = (std::max(a, base) - origin) / LazyChunk, last = (std::min(a + size, end) - 1 - origin) / LazyChunk; c <= last; ++c) {
        if (LazyKnown(c)) continue;
        // (the whole step, so that its bit can say so from now on)
        if (CommitReserved(std::max(base, origin + c * LazyChunk), std::min(end, origin + (c + 1) * LazyChunk))) LazyMark(c);
    }
}
} // namespace rt

static ssize_t PositionalIo(int fd, void* buffer, size_t size, long long offset, bool write) {
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    OVERLAPPED o{};
    o.Offset = (DWORD)(unsigned long long)offset;
    o.OffsetHigh = (DWORD)((unsigned long long)offset >> 32);
    DWORD done = 0;
    const DWORD n = size > 0x7fffffff ? 0x7fffffff : (DWORD)size;
    rt::EnsureCommitted(buffer, n); // (the system fails the call on a page not yet committed: see LazyCommitRange)
    const BOOL ok = write ? WriteFile(h, buffer, n, &done, &o) : ReadFile(h, buffer, n, &done, &o);
    if (!ok && GetLastError() != ERROR_HANDLE_EOF) { errno = EIO; return -1; }
    return (ssize_t)done;
}
ssize_t pread(int fd, void* buffer, size_t size, long long offset) { return PositionalIo(fd, buffer, size, offset, false); }
ssize_t pwrite(int fd, const void* buffer, size_t size, long long offset) {
    return PositionalIo(fd, const_cast<void*>(buffer), size, offset, true);
}

namespace rt {
const std::string& DataDir() {
    static const std::string dir = [] {
        if (const char* env = std::getenv("OOT_RECOMP_DATA"); env && *env) return std::string(env);
        // (started by hand from an installation's own folder, <installation>\app\tootr.exe: that installation's data)
        char exe[MAX_PATH]; const DWORD n = GetModuleFileNameA(nullptr, exe, MAX_PATH);
        if (n && n < MAX_PATH) {
            std::string up(exe, n);
            for (int i = 0; i < 2; ++i) if (const size_t cut = up.find_last_of("\\/"); cut != std::string::npos) up.resize(cut);
            if (GetFileAttributesA((up + "\\main.img").c_str()) != INVALID_FILE_ATTRIBUTES) return up;
        }
        const char* base = std::getenv("LOCALAPPDATA");
        if (!base) base = std::getenv("USERPROFILE");
        std::string d = std::string(base ? base : ".") + "\\tootr";
        CreateDirectoryA(d.c_str(), nullptr);
        return d;
    }();
    return dir;
}
namespace {
struct NamedThread { char name[32]; HANDLE handle; };
NamedThread named_threads[256]; std::atomic<unsigned> named_thread_count{0};
}
void SetThreadName(const char* name) {
    wchar_t wide[64];
    const int n = MultiByteToWideChar(CP_UTF8, 0, name, -1, wide, 64);
    if (n > 0) SetThreadDescription(GetCurrentThread(), wide);
    // (for the profiler, which stops a thread by its name)
    const unsigned slot = named_thread_count.fetch_add(1);
    if (slot < 256) {
        std::snprintf(named_threads[slot].name, sizeof named_threads[slot].name, "%s", name);
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &named_threads[slot].handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
    }
}
void* ThreadByName(const char* name) {
    const unsigned count = std::min(named_thread_count.load(), 256u);
    for (unsigned i = count; i-- > 0;) if (!std::strcmp(named_threads[i].name, name) && named_threads[i].handle) return named_threads[i].handle;
    return nullptr;
}
namespace {
long long NowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool PlainSleeps() { static const bool plain = std::getenv("OOT_PLAIN_SLEEPS") != nullptr; return plain; }
// (the 1 ms tick: for every timed wait in the process - condition variables, the driver's - not only the sleeps here)
const bool fine_tick = [] {
    if (std::getenv("OOT_PLAIN_SLEEPS")) return false;
    if (const HMODULE winmm = LoadLibraryA("winmm.dll"))
        if (const auto begin = reinterpret_cast<unsigned(__stdcall*)(unsigned)>(reinterpret_cast<void*>(GetProcAddress(winmm, "timeBeginPeriod")))) { begin(1); return true; }
    return false;
}();
}
void SleepNs(long long ns) {
    if (ns <= 0) { SwitchToThread(); return; }
    if (PlainSleeps()) { std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); return; }
    const long long until = NowNs() + ns;
    if (ns >= 100000) {
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
        // (one timer a thread, made at its first sleep; Windows before 10 1803 has no such timer: Sleep, to the 1 ms tick)
        static thread_local const HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        // (it wakes up to a fraction of a millisecond late: sleep to 0.3 ms short of the time, the rest below)
        const long long coarse = ns - 300000;
        if (coarse >= 100000) {
            LARGE_INTEGER due; due.QuadPart = -(coarse / 100);
            if (timer && SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) WaitForSingleObject(timer, INFINITE);
            else if (coarse >= 2000000) Sleep((DWORD)(coarse / 1000000 - 1));
        }
    }
    // (the rest, or a sleep too short for any timer: the core is offered to other threads until the time has passed)
    while (NowNs() < until) { if (!SwitchToThread()) YieldProcessor(); }
}
void SleepUntilNs(long long at) { const long long now = NowNs(); if (at > now) SleepNs(at - now); }
} // namespace rt

#else // POSIX
#include <pthread.h>
#include <unistd.h>

namespace rt {
const std::string& DataDir() {
    static const std::string dir = [] {
        if (const char* env = std::getenv("OOT_RECOMP_DATA"); env && *env) return std::string(env);
        // (started by hand from an installation's own folder, <installation>/app/tootr: that installation's data. The
        // installer's "tootr" command and menu entry name the folder themselves)
        char exe[4096]; const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            std::string up(exe, (size_t)n);
            for (int i = 0; i < 2; ++i) if (const size_t cut = up.rfind('/'); cut != std::string::npos) up.resize(cut);
            if (!up.empty() && access((up + "/main.img").c_str(), R_OK) == 0) return up;
        }
        if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) return std::string(xdg) + "/tootr";
        const char* home = std::getenv("HOME");
        return std::string(home ? home : ".") + "/.local/share/tootr";
    }();
    return dir;
}
void SetThreadName(const char* name) { pthread_setname_np(pthread_self(), name); }
void SleepNs(long long ns) { if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); else std::this_thread::yield(); }
void SleepUntilNs(long long at) { std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::nanoseconds(at)))); }
} // namespace rt
#endif
