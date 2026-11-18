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
} // namespace rt
