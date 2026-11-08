#include "platform.h"

#include <cstdlib>

#ifdef _WIN32
#include <windows.h>

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

int madvise(void* addr, size_t len, int advice) {
    if (advice != MADV_REMOVE) return 0;
    const bool ok = ForEachRegion(addr, len, [](void* p, size_t n, const MEMORY_BASIC_INFORMATION& info) {
        return info.State != MEM_COMMIT || VirtualFree(p, n, MEM_DECOMMIT) != 0;
    });
    return ok ? 0 : (errno = EINVAL, -1);
}

static ssize_t PositionalIo(int fd, void* buffer, size_t size, long long offset, bool write) {
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    OVERLAPPED o{};
    o.Offset = (DWORD)(unsigned long long)offset;
    o.OffsetHigh = (DWORD)((unsigned long long)offset >> 32);
    DWORD done = 0;
    const DWORD n = size > 0x7fffffff ? 0x7fffffff : (DWORD)size;
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
        const char* base = std::getenv("LOCALAPPDATA");
        if (!base) base = std::getenv("USERPROFILE");
        std::string d = std::string(base ? base : ".") + "\\tootr";
        CreateDirectoryA(d.c_str(), nullptr);
        return d;
    }();
    return dir;
}
void SetThreadName(const char* name) {
    wchar_t wide[64];
    const int n = MultiByteToWideChar(CP_UTF8, 0, name, -1, wide, 64);
    if (n > 0) SetThreadDescription(GetCurrentThread(), wide);
}
} // namespace rt

#else // POSIX
#include <pthread.h>

namespace rt {
const std::string& DataDir() {
    static const std::string dir = [] {
        if (const char* env = std::getenv("OOT_RECOMP_DATA"); env && *env) return std::string(env);
        if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) return std::string(xdg) + "/tootr";
        const char* home = std::getenv("HOME");
        return std::string(home ? home : ".") + "/.local/share/tootr";
    }();
    return dir;
}
void SetThreadName(const char* name) { pthread_setname_np(pthread_self(), name); }
} // namespace rt
#endif
