#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "logfile.h"
#include "rt.h"

namespace rt {

void Fatal(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char message[1024];
    std::vsnprintf(message, sizeof message, fmt, args);
    va_end(args);
    std::fprintf(stderr, "[rt] FATAL: %s\n", message);
    std::fflush(stderr);
    rt::ShowFatal(message); // (a player who started from a shortcut sees why, and where the log is)
    std::abort();
}

void* MapFixed(u64 address, u64 size) {
    size = (size + 0xfff) & ~0xfffULL;
    void* p = mmap(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED || reinterpret_cast<u64>(p) != address)
        Fatal("cannot map %#llx+%#llx: %s", (unsigned long long)address, (unsigned long long)size, std::strerror(errno));
    return p;
}

u64 ArenaAlloc(u64 size, u64 align) {
    static std::mutex mutex;
    static u64 next = 0;
    std::lock_guard lock{mutex};
    if (!next) {
        MapFixed(ArenaBase, ArenaSize);
        next = ArenaBase;
    }
    next = (next + align - 1) & ~(align - 1);
    const u64 result = next;
    next += size;
    if (next > ArenaBase + ArenaSize) Fatal("runtime arena exhausted");
    return result;
}

} // namespace rt
