// OOT_PROFILE=1 (development, Linux): a sampling CPU profiler. A process CPU-time timer (ITIMER_PROF,
// 2 kHz) interrupts whichever thread is running; the handler records that thread and its call chain
// (frame pointers: the runtime is built with them). Every 10 s the samples are written to
// <data>/profile-<n>.txt: per thread name, the hottest leaf and caller chains as offsets in the
// executable (addr2line -f -C -e tootr <offsets>).
#ifdef __linux__
#include <dirent.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <ucontext.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "platform.h"
#include "rt.h"

extern std::atomic<int> profile_present_tid, profile_signal_tid;
namespace rt {
namespace {
constexpr int Depth = 20;
struct Sample { int tid; uintptr_t pc[Depth]; };
constexpr size_t Capacity = 1 << 18;
Sample* ring = nullptr;
std::atomic<size_t> head{0};
uintptr_t base = 0, text_end = ~uintptr_t(0);

bool unwind = false; // OOT_PROFILE_UNWIND: call chains by the unwinder (through libraries without frame pointers)
void OnProf(int, siginfo_t*, void* context) {
    const size_t i = head.fetch_add(1, std::memory_order_relaxed);
    if (i >= Capacity) return;
    Sample& s = ring[i];
    s.tid = (int)syscall(SYS_gettid);
    const auto& g = static_cast<ucontext_t*>(context)->uc_mcontext.gregs;
    s.pc[0] = (uintptr_t)g[REG_RIP];
    if (unwind) {
        void* frames[Depth + 8];
        const int n = backtrace(frames, Depth + 8);
        int k = 0; // (frames up to the signal frame are the handler's own)
        while (k < n && (uintptr_t)frames[k] != s.pc[0]) ++k;
        for (int d = 1; d < Depth; ++d) s.pc[d] = k + d < n ? (uintptr_t)frames[k + d] : 0;
        return;
    }
    uintptr_t fp = (uintptr_t)g[REG_RBP];
    for (int d = 1; d < Depth; ++d) {
        // (a frame pointer chain inside this thread's stack: stop at anything implausible)
        if (fp < 4096 || (fp & 7) || fp > (uintptr_t)g[REG_RSP] + (8u << 20) || fp < (uintptr_t)g[REG_RSP]) { s.pc[d] = 0; continue; }
        const uintptr_t* frame = reinterpret_cast<const uintptr_t*>(fp);
        s.pc[d] = frame[1];
        fp = frame[0];
    }
}

std::string ThreadName(int tid) {
    char path[64]; std::snprintf(path, sizeof path, "/proc/self/task/%d/comm", tid);
    char name[32] = "?";
    if (FILE* f = std::fopen(path, "r")) { if (std::fgets(name, sizeof name, f)) name[std::strcspn(name, "\n")] = 0; std::fclose(f); }
    std::string n = name;
    if (n.size() > 2 && n[0] == 'g' && std::isdigit((unsigned char)n[1])) n = "guest"; // (guest threads, by kind)
    return n;
}

std::string Off(uintptr_t pc) {
    char t[96];
    if (pc >= base && pc < text_end) std::snprintf(t, sizeof t, "%#lx", (unsigned long)(pc - base - 1));
    else if (Dl_info info{}; dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_fbase)
        std::snprintf(t, sizeof t, "%s+%#lx", std::strrchr(info.dli_fname, '/') ? std::strrchr(info.dli_fname, '/') + 1 : info.dli_fname,
                      (unsigned long)(pc - (uintptr_t)info.dli_fbase));
    else std::snprintf(t, sizeof t, "ext");
    return t;
}

void Writer() {
    for (int n = 0;; ++n) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        const size_t count = std::min(head.exchange(0), Capacity);
        std::map<std::string, std::map<std::string, u64>> leaves, chains, owners, inclusive; // owners: the first frame in the executable
        std::map<uintptr_t, std::string> modules;
        const auto module_of = [&](uintptr_t pc) -> std::string {
            const uintptr_t page = pc >> 16;
            auto it = modules.find(page);
            if (it != modules.end()) return it->second;
            Dl_info info{}; std::string name = "?";
            if (dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_fname) {
                name = info.dli_fname; name = name.substr(name.rfind('/') + 1);
                if (info.dli_sname) return name + ":" + info.dli_sname; // (exported functions by name; not cached per page)
            }
            return modules[page] = name;
        };
        std::map<std::string, u64> per_thread;
        std::map<int, std::string> names;
        for (size_t i = 0; i < count; ++i) {
            const Sample& s = ring[i];
            auto it = names.find(s.tid);
            if (it == names.end()) it = names.emplace(s.tid, ThreadName(s.tid)).first;
            ++per_thread[it->second];
            ++leaves[it->second][s.pc[0] >= base && s.pc[0] < text_end ? Off(s.pc[0]) : "ext:" + module_of(s.pc[0])];
            for (int d = 0; d < Depth && s.pc[d]; ++d) if (s.pc[d] >= base && s.pc[d] < text_end) { ++owners[it->second][Off(s.pc[d])]; break; }
            { std::string raw = it->second; // (full chains, executable offsets only: inclusive time offline)
              for (int d = 0; d < Depth && s.pc[d]; ++d) if (s.pc[d] >= base && s.pc[d] < text_end) raw += " " + Off(s.pc[d]);
              ++inclusive["raw"][raw]; }
            std::string chain;
            for (int d = 0; d < 6 && s.pc[d]; ++d) chain += " " + Off(s.pc[d]);
            ++chains[it->second][chain];
        }
        const std::string path = DataDir() + "/profile-" + std::to_string(n) + ".txt";
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) continue;
        std::fprintf(f, "%zu samples (2 kHz CPU time) in 10 s\n", count);
        std::vector<std::pair<u64, std::string>> threads;
        for (auto& [t, c] : per_thread) threads.push_back({c, t});
        std::sort(threads.rbegin(), threads.rend());
        for (auto& [c, t] : threads) {
            std::fprintf(f, "\n== %s: %llu samples (%.1f%% of a core)\n", t.c_str(), (unsigned long long)c, c / 200.0);
            std::vector<std::pair<u64, std::string>> top;
            for (auto& [k, v] : leaves[t]) top.push_back({v, k});
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 25; ++i) std::fprintf(f, "  leaf %6llu %s\n", (unsigned long long)top[i].first, top[i].second.c_str());
            top.clear();
            for (auto& [k, v] : owners[t]) top.push_back({v, k});
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 30; ++i) std::fprintf(f, "  owner %6llu %s\n", (unsigned long long)top[i].first, top[i].second.c_str());
            top.clear();
            for (auto& [k, v] : chains[t]) top.push_back({v, k});
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 25; ++i) std::fprintf(f, "  chain %6llu%s\n", (unsigned long long)top[i].first, top[i].second.c_str());
        }
        std::fclose(f);
        if (FILE* r = std::fopen((path + ".raw").c_str(), "w")) {
            for (auto& [k, v] : inclusive["raw"]) std::fprintf(r, "%llu %s\n", (unsigned long long)v, k.c_str());
            std::fclose(r);
        }
    }
}
} // namespace

// OOT_PROFILE_WALL=<thread name>: that thread is sampled in wall-clock time instead (blocked time included:
// waits show as their callers), from 20 s after the start (when the thread exists).
void WallTimer(std::string name) {
    const char* delay = std::getenv("OOT_PROFILE_WALL_DELAY"); // (seconds before the thread is picked; default 20)
    std::this_thread::sleep_for(std::chrono::seconds(delay ? std::atoi(delay) : 20));
    int tid = 0;
    // (the thread of that name with the most CPU time over 2 s; guest: any guest thread)
    const auto times = [&] {
        std::map<int, long> t;
        const std::string dir = "/proc/" + std::to_string(getpid()) + "/task/";
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) {
                const int id = std::atoi(e->d_name); if (!id) continue;
                const std::string n = ThreadName(id);
                if (n != name) continue;
                if (FILE* f = std::fopen((dir + e->d_name + "/stat").c_str(), "r")) {
                    char buf[512]; const size_t len = std::fread(buf, 1, sizeof buf - 1, f); buf[len] = 0; std::fclose(f);
                    const char* p = std::strrchr(buf, ')'); long v[16]{};
                    if (p) std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %ld %ld", &v[0], &v[1]);
                    t[id] = v[0] + v[1];
                }
            }
            closedir(d);
        }
        return t;
    };
    const bool named = name == "present" || name == "signal";
    if (named) { // (the thread presenting frames; the guest thread signalling CPU events)
        auto& t = name == "present" ? profile_present_tid : profile_signal_tid;
        for (int i = 0; i < 600 && !t.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        tid = t.load();
    }
    const auto before = named ? decltype(times()){} : times();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    long best = -1;
    if (!named) for (auto& [id, t] : times()) { const long d = t - (before.count(id) ? before.at(id) : 0); if (d > best) best = d, tid = id; }
    if (!tid) { RT_LOG("profiler: no thread named %s", name.c_str()); return; }
    sigevent ev{}; ev.sigev_notify = SIGEV_THREAD_ID; ev.sigev_signo = SIGPROF; ev._sigev_un._tid = tid;
    timer_t timer;
    if (timer_create(CLOCK_MONOTONIC, &ev, &timer) != 0) { RT_LOG("profiler: timer_create failed"); return; }
    itimerspec spec{{0, 500000}, {0, 500000}};
    timer_settime(timer, 0, &spec, nullptr);
    RT_LOG("profiler: sampling thread %s (%d) in wall-clock time at 2 kHz", name.c_str(), tid);
}

void StartProfiler() {
    const char* wall = std::getenv("OOT_PROFILE_WALL");
    if (!std::getenv("OOT_PROFILE") && !wall) return;
    ring = new Sample[Capacity];
    Dl_info info{};
    dladdr(reinterpret_cast<void*>(&StartProfiler), &info);
    base = reinterpret_cast<uintptr_t>(info.dli_fbase);
    text_end = base + (512ull << 20); // (the executable's code: its first 512 MiB)
    struct sigaction action{};
    action.sa_sigaction = OnProf;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    if ((unwind = std::getenv("OOT_PROFILE_UNWIND") != nullptr)) { void* warm[4]; backtrace(warm, 4); } // (loads the unwinder now)
    sigaction(SIGPROF, &action, nullptr);
    std::thread(Writer).detach();
    if (wall) { std::thread(WallTimer, std::string(wall)).detach(); return; }
    itimerval timer{{0, 500}, {0, 500}};
    setitimer(ITIMER_PROF, &timer, nullptr);
    RT_LOG("profiler: sampling CPU time at 2 kHz; reports every 10 s in the data directory");
}
} // namespace rt
#elif defined(_WIN32)
// Windows: OOT_PROFILE_WALL=<thread name> (nvn-exec: the renderer's thread) samples that thread in wall-clock time,
// a thousand times a second: the thread is stopped for a moment, its call chain read from the unwind tables, and
// every 10 s the chains are written to <data>/profile-<n>.txt as "<count> <module>+<offset> <module>+<offset> ..."
// (innermost first), the most frequent first. Offsets are within each module, so a build of the same source with
// debug information names them (llvm-symbolizer --obj=tootr.exe --relative-address). OOT_PROFILE_WALL_DELAY=<s>:
// seconds before sampling starts (20).
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "platform.h"
#include "rt.h"

namespace rt {
namespace {
std::string ModuleOffset(DWORD64 pc) {
    HMODULE module = nullptr; char text[320];
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(pc), &module) || !module) {
        std::snprintf(text, sizeof text, "?+%#llx", (unsigned long long)pc); return text;
    }
    char path[MAX_PATH]; const DWORD n = GetModuleFileNameA(module, path, MAX_PATH);
    const char* name = path; for (DWORD i = 0; i < n; ++i) if (path[i] == '\\' || path[i] == '/') name = path + i + 1;
    std::snprintf(text, sizeof text, "%s+%#llx", n ? name : "?", (unsigned long long)(pc - reinterpret_cast<DWORD64>(module)));
    return text;
}
void Sampler(std::string name) {
    const char* delay = std::getenv("OOT_PROFILE_WALL_DELAY");
    std::this_thread::sleep_for(std::chrono::seconds(delay ? std::atoi(delay) : 20));
    const HANDLE thread = ThreadByName(name.c_str());
    if (!thread) { RT_LOG("profiler: no thread named %s", name.c_str()); return; }
    RT_LOG("profiler: sampling thread %s in wall-clock time, 1000 times a second; reports every 10 s in the data directory", name.c_str());
    std::map<std::vector<DWORD64>, unsigned long long> chains;
    auto written = std::chrono::steady_clock::now(); int file = 0;
    unsigned long long random = 0x9e3779b97f4a7c15ull;
    for (;;) {
        // (0.5 to 1.5 ms apart, at random: an even step falls into step with a thread that itself works by the clock)
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        SleepNs(500000 + (long long)(random % 1000000));
        std::vector<DWORD64> chain;
        if (SuspendThread(thread) != (DWORD)-1) {
            CONTEXT context{}; context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (GetThreadContext(thread, &context)) {
                for (int depth = 0; depth < 28 && context.Rip; ++depth) {
                    chain.push_back(context.Rip);
                    DWORD64 image = 0;
                    // (no allocation and no lock of ours while the thread is stopped: it may hold the heap's)
                    const PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image, nullptr);
                    if (!function) { // (a leaf: its return address is on top of the stack)
                        if (IsBadReadPtr(reinterpret_cast<void*>(context.Rsp), 8)) break;
                        context.Rip = *reinterpret_cast<DWORD64*>(context.Rsp); context.Rsp += 8; continue;
                    }
                    void* handler_data = nullptr; DWORD64 frame = 0;
                    RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, context.Rip, function, &context, &handler_data, &frame, nullptr);
                }
            }
            ResumeThread(thread);
        }
        if (!chain.empty()) ++chains[chain];
        if (std::chrono::steady_clock::now() - written < std::chrono::seconds(10)) continue;
        written = std::chrono::steady_clock::now();
        std::vector<std::pair<unsigned long long, const std::vector<DWORD64>*>> order;
        unsigned long long total = 0;
        for (auto& [c, n] : chains) { order.push_back({n, &c}); total += n; }
        std::sort(order.rbegin(), order.rend());
        const std::string path = DataDir() + "/profile-" + std::to_string(file++) + ".txt";
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            std::fprintf(f, "# thread %s, %llu samples, about 1 ms apart\n", name.c_str(), total);
            // (the innermost frames first: where the time is; then the chains)
            std::map<std::string, unsigned long long> leaves, anywhere;
            for (auto& [n, c] : order) { leaves[ModuleOffset((*c)[0])] += n; std::vector<std::string> seen; for (DWORD64 pc : *c) { std::string m = ModuleOffset(pc); if (std::find(seen.begin(), seen.end(), m) == seen.end()) { anywhere[m] += n; seen.push_back(m); } } }
            std::vector<std::pair<unsigned long long, std::string>> top;
            for (auto& [m, n] : leaves) top.push_back({n, m});
            std::sort(top.rbegin(), top.rend());
            std::fprintf(f, "# innermost\n");
            for (size_t i = 0; i < top.size() && i < 60; ++i) std::fprintf(f, "leaf %llu %s\n", top[i].first, top[i].second.c_str());
            top.clear(); for (auto& [m, n] : anywhere) top.push_back({n, m});
            std::sort(top.rbegin(), top.rend());
            std::fprintf(f, "# anywhere in the chain\n");
            for (size_t i = 0; i < top.size() && i < 120; ++i) std::fprintf(f, "incl %llu %s\n", top[i].first, top[i].second.c_str());
            std::fprintf(f, "# chains\n");
            for (size_t i = 0; i < order.size() && i < 400; ++i) {
                std::fprintf(f, "%llu", order[i].first);
                for (DWORD64 pc : *order[i].second) std::fprintf(f, " %s", ModuleOffset(pc).c_str());
                std::fprintf(f, "\n");
            }
            std::fclose(f);
        }
        chains.clear();
    }
}
} // namespace
void StartProfiler() {
    if (const char* wall = std::getenv("OOT_PROFILE_WALL")) std::thread(Sampler, std::string(wall)).detach();
}
} // namespace rt
#else
namespace rt { void StartProfiler() {} }
#endif
