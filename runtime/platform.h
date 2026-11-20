// Host platform layer. On Linux this is POSIX (sys/mman.h and friends). On Windows it provides the
// small subset of mmap/mprotect/madvise the runtime uses, on VirtualAlloc/VirtualProtect/VirtualFree:
//   mmap(addr, len, prot, MAP_ANONYMOUS [| MAP_FIXED_NOREPLACE] ...)  reserve (+ commit unless PROT_NONE)
//   mprotect(addr, len, prot)    PROT_NONE: no access (contents kept); otherwise commit + protect
//   madvise(addr, len, MADV_REMOVE)  discard (decommit: zero when committed again)
//   munmap(addr, len)            release a whole mmap'd range
// Shared (file-backed) mappings are not provided; kernel.cpp handles their one use separately.
#pragma once
#include <string>

#ifdef _WIN32
#include <cstddef>
#include <sys/types.h>

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_NORESERVE 0x4000
#define MAP_FIXED_NOREPLACE 0x100000
#define MAP_FAILED ((void*)-1)
#define MADV_REMOVE 9

void* mmap(void* addr, size_t len, int prot, int flags, int fd, long long offset);
int munmap(void* addr, size_t len);
int mprotect(void* addr, size_t len, int prot);
int madvise(void* addr, size_t len, int advice);

// Files: positional I/O, and binary mode for every open() (O_CLOEXEC is the flag every open passes).
#include <fcntl.h>
#include <io.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC O_BINARY
#endif
ssize_t pread(int fd, void* buffer, size_t size, long long offset);
ssize_t pwrite(int fd, const void* buffer, size_t size, long long offset);
inline int fsync(int fd) { return _commit(fd); }
#else
#include <sys/mman.h>
#endif

namespace rt {
// Windows: memory that is reserved and becomes usable where it is first touched (the game's heap). Windows counts
// writable memory against the system's limit when it is mapped; Linux when it is used. Within such a range a first
// access commits the megabyte around it; code that hands a pointer into the range to the system (a file read, a
// socket, the GPU driver) calls EnsureCommitted first, because the system does not ask us when it finds the page
// missing - the call just fails. No-ops elsewhere.
#ifdef _WIN32
void LazyCommitRange(void* base, size_t size);          // [base, base + size) is such a range from now on (size 0: none)
bool CommitOnFault(void* address, bool write);          // an access violation there: true when the access can be tried again
void EnsureCommitted(const void* address, size_t size); // usable before the system or the GPU driver touches it
#else
inline void LazyCommitRange(void*, size_t) {}
inline bool CommitOnFault(void*, bool) { return false; }
inline void EnsureCommitted(const void*, size_t) {}
#endif
// Directory for the game's module images, saves, caches and settings: $OOT_RECOMP_DATA, else
// ~/.local/share/tootr ($XDG_DATA_HOME) on Linux and %LOCALAPPDATA%\tootr on Windows.
const std::string& DataDir();
// Names the calling thread (debuggers, profilers, crash reports).
void SetThreadName(const char* name);
#ifdef _WIN32
// The calling thread's id, and a thread's priority from a Unix nice value (0-1: normal; 2-6: below normal; 7 or more:
// lowest). True when it was set.
int HostThreadId();
bool SetThreadNice(int thread_id, int nice);
#endif
#ifdef _WIN32
// The name of the function at an offset in tootr.exe, from tootr.sym beside it (tools/build_game.py writes it): "" when
// there is no such file. offset_out: how far into the function.
const char* SymbolAt(unsigned long long rva, unsigned long long* offset_out);
void* ThreadByName(const char* name); // (the profiler: a handle of the last thread that took that name; null when none)
#endif
// Sleeps of the length asked for. Windows wakes a sleeping thread only on its timer tick, 15.6 ms apart unless a
// program asks for finer, and its Sleep counts whole milliseconds: the game's job system waits for its workers in
// steps of 20 microseconds, and each became a tick - a frame took four of them, 62 ms, whatever the graphics card.
// On Windows: the 1 ms tick for the process (once), a high-resolution timer for sleeps from 0.1 ms, and for shorter
// ones the time passed yielding the core. Elsewhere the system's own sleep, which is that exact already.
// OOT_PLAIN_SLEEPS=1: the system's own on Windows too, as before.
void SleepNs(long long nanoseconds);
// The PC's memory, as text for the log ("PC memory 10.3 GiB free of 31.2").
std::string SystemMemoryNote();
void SleepUntilNs(long long steady_clock_nanoseconds); // (a std::chrono::steady_clock time since its epoch)
} // namespace rt
