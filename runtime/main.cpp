// Entry point: loads and links the recompiled modules, then starts the SDK exactly as rtld does,
// by calling nn::init::Start(main thread handle, argument region, init flag, module init, module
// fini) on the main guest thread. nn::init::Start runs libc/os init, nninitStartup, the static
// constructors and finally nnMain.

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "kernel.h"
#include "native.h"

namespace rt {

void ApplyNatives();

namespace {
thread_local Cpu* current_cpu = nullptr;
}

Cpu* CurrentCpu() { return current_cpu; }
void BindCpu(Cpu* c) { current_cpu = c; }

Cpu* CreateCpu(u64 stack_top, u64 tls) {
    Cpu* c = new Cpu{};
    c->sp = stack_top & ~15ULL;
    c->tpidrro = tls;
    return c;
}

} // namespace rt

using namespace rt;

// rtld's callbacks for nn::init::Start.
NATIVE("rt::RtldSetInitialized") {}
NATIVE("rt::RtldInitializeModules") {
    for (const auto& m : Modules())
        if (m.init) CallGuest(c, m.init);
}
NATIVE("rt::RtldFinalizeModules") {}

// Driver shader caches: the game builds thousands of pipelines; the default limits (NVIDIA 1 GB, Mesa
// 1 GB) fill up, after which every run compiles them again (stutter). Set before the driver loads;
// values the user set win.
static void RaiseDriverShaderCaches() {
#ifdef _WIN32
    const auto set = [](const char* k, const char* v) { if (!std::getenv(k)) _putenv_s(k, v); };
#else
    const auto set = [](const char* k, const char* v) { setenv(k, v, 0); };
#endif
    set("__GL_SHADER_DISK_CACHE", "1");
    set("__GL_SHADER_DISK_CACHE_SIZE", "10737418240");
    set("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1");
    set("MESA_SHADER_CACHE_MAX_SIZE", "10G");
}

namespace rt { void StartProfiler(); }
using rt::StartProfiler;
int main(int argc, char** argv) {
    RaiseDriverShaderCaches();
    StartProfiler(); // (OOT_PROFILE=1)
    std::string data_dir;
    if (argc > 1) data_dir = argv[1];
    else data_dir = DataDir();

    // OOT_DEBUG_ATTACH=1 lets a debugger attach (gdb -p) despite Yama ptrace_scope=1.
#ifdef __linux__
    if (const char* attach = std::getenv("OOT_DEBUG_ATTACH"); attach && *attach == '1')
        prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif

    InitializeKernel();
    LoadModules(data_dir);
    ApplyNatives();

    constexpr u64 MainStackSize = 0x100000;
    const u64 stack = ArenaAlloc(MainStackSize, 0x1000);
    KThread* main_thread = CreateMainThread(stack + MainStackSize);
    BindThread(main_thread);

    u64 start = 0;
    for (const auto& m : Modules())
        for (const auto& s : m.exports)
            if (s.name == "_ZN2nn4init5StartEmmPFvvES2_S2_") start = s.address;
    if (!start) Fatal("nn::init::Start not found");

    const u64 argument_region = ArenaAlloc(0x9000, 0x1000);
    RT_LOG("starting the SDK on the main thread");
    CallGuest(main_thread->cpu, start,
              {main_thread->handle, argument_region, RegisterNative("rt::RtldSetInitialized", nullptr),
               RegisterNative("rt::RtldInitializeModules", nullptr),
               RegisterNative("rt::RtldFinalizeModules", nullptr)});
    RT_LOG("nn::init::Start returned");
    return 0;
}
