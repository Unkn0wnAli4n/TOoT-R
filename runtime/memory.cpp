#include <algorithm>
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
#ifdef _WIN32
    // Windows counts memory against the system's limit when it is mapped writable, not when it is first used: the
    // whole 4 GiB of this region counted from the start, though a few hundred MiB of it are ever handed out (the game
    // stood at 16.6 GiB committed on its title screen, and a 16 GiB PC ran out while a save loaded). The region is
    // only reserved here, and made usable piece by piece as it is handed out.
    static u64 usable = 0;
    if (!next) {
        void* p = mmap(reinterpret_cast<void*>(ArenaBase), ArenaSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED || reinterpret_cast<u64>(p) != ArenaBase) Fatal("cannot reserve the runtime arena");
        next = usable = ArenaBase;
    }
#else
    if (!next) {
        MapFixed(ArenaBase, ArenaSize);
        next = ArenaBase;
    }
#endif
    next = (next + align - 1) & ~(align - 1);
    const u64 result = next;
    next += size;
    if (next > ArenaBase + ArenaSize) Fatal("runtime arena exhausted");
#ifdef _WIN32
    if (next > usable) {
        const u64 to = std::min<u64>(ArenaBase + ArenaSize, (next + (16u << 20) - 1) & ~u64((16u << 20) - 1));
        if (mprotect(reinterpret_cast<void*>(usable), to - usable, PROT_READ | PROT_WRITE) != 0) Fatal("out of memory (the runtime arena could not grow)");
        usable = to;
    }
#endif
    return result;
}

} // namespace rt
