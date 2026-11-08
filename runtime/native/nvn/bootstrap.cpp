// NVN entry points. The game obtains every NVN function by name (nvnBootstrapLoader, then
// nvnDeviceGetProcAddress); each name maps to its native Vulkan-backed implementation. Names not
// implemented yet resolve to a stub that stops with the function's name when it is first called.

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#include "../../native.h"

using namespace rt;

namespace rt {
extern void (*g_call_tracer)(Cpu* c, u64 target);
}

namespace {

void MissingNvnFunction(Cpu* c) {
    Fatal("NVN function %s is not implemented (called from %s)", NativeName(c->pc).c_str(),
          DescribeAddress(c->x[30]).c_str());
}

// Only data-only builder/state operations may use the recompiled driver. GPU paths
// expect a real console device and can dereference null service objects before any SVC.
bool PureDriverFunction(std::string_view name) {
    if (const auto builder = name.find("Builder"); builder != std::string_view::npos) {
        const auto operation = name.substr(builder + 7);
        if (operation.starts_with("Set") || operation.starts_with("Get")) return true;
    }
    for (const char* prefix : {"nvnBlendState", "nvnChannelMaskState", "nvnColorState",
             "nvnDepthStencilState", "nvnMultisampleState", "nvnPolygonState",
             "nvnVertexAttribState", "nvnVertexStreamState", "nvnTextureView",
             "nvnDlssParameters", "nvnDirtyTileMapParameters"})
        if (name.starts_with(prefix)) return true;
    return name == "nvnDeviceSetWindowOriginMode" || name == "nvnDeviceGetWindowOriginMode" ||
           name == "nvnDeviceSetDepthMode" || name == "nvnDeviceGetDepthMode";
}

const std::unordered_map<std::string, u64>& DriverFunctions() {
    static std::unordered_map<std::string, u64> map;
    static std::once_flag once;
    std::call_once(once, [] {
        u64 sdk_base = 0;
        for (const auto& m : Modules())
            if (m.name == "sdk") sdk_base = m.base;
        const char* override_path = std::getenv("OOT_RECOMP_NVN_MAP");
        const std::string path = override_path ? override_path :
            DataDir() + "/nvn_map.txt";
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) Fatal("missing %s", path.c_str());
        char name[256];
        unsigned long long offset;
        while (std::fscanf(f, "%255s %llx", name, &offset) == 2)
            if (offset) map.emplace(name, sdk_base + offset);
        std::fclose(f);
    });
    return map;
}

// OOT_NVN_TRACE: logs every call the game makes into an NVN function (native or driver).
std::unordered_map<u64, std::string> traced_names;
void TraceNvnCall(Cpu* c, u64 target) {
    if (auto it = traced_names.find(target); it != traced_names.end())
        std::fprintf(stderr, "[nvn] %s(%#llx, %#llx, %#llx)\n", it->second.c_str(), (unsigned long long)c->x[0],
                     (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
}

u64 NvnProcResolve(const char* name);
u64 NvnProc(const char* name) {
    const u64 address = NvnProcResolve(name);
    static const bool trace = std::getenv("OOT_NVN_TRACE") != nullptr;
    if (trace && address) {
        traced_names.emplace(address, name);
        rt::g_call_tracer = TraceNvnCall;
    }
    return address;
}

u64 NvnProcResolve(const char* name) {
    if (!name) return 0;
    if (FindNative(name)) return RegisterNative(name, nullptr);
    if (PureDriverFunction(name)) {
        const auto& driver = DriverFunctions();
        if (auto it = driver.find(name); it != driver.end()) return it->second;
    }
    return RegisterNative(name, MissingNvnFunction);
}

} // namespace

namespace nvn {
// Address of the SDK driver's implementation of an NVN function.
u64 DriverFunction(const char* name) {
    const auto& driver = DriverFunctions();
    const auto it = driver.find(name);
    if (it == driver.end()) Fatal("no driver function %s", name);
    return it->second;
}
} // namespace nvn

NATIVE("nvnBootstrapLoader") { Ret(c, NvnProc(Ptr<const char>(c, 0))); }
NATIVE("nvnDeviceGetProcAddress") { Ret(c, NvnProc(Ptr<const char>(c, 1))); }

// nv:: graphics setup. The native NVN implementation allocates through the host, so the guest
// graphics allocator is only recorded.
namespace {
u64 graphics_alloc = 0, graphics_free = 0, graphics_realloc = 0, graphics_user = 0;
}
// Development aid: OOT_NVN_MAP=<names file> prints "<name> <sdk offset>" for every NVN function,
// using the SDK driver's own name lookup, so native implementations can be checked against the
// real driver. Runs once the SDK is initialized (here), then exits.
static void MapNvnFunctions(Cpu* c, const char* names) {
    u64 sdk_base = 0;
    for (const auto& m : Modules())
        if (m.name == "sdk") sdk_base = m.base;
    const u64 driver = *reinterpret_cast<u64*>(sdk_base + 0xe77fc0);
    const u64 buffer = ArenaAlloc(256);
    const auto call = [&](u64 fn, const std::string& name) {
        std::memcpy(reinterpret_cast<void*>(buffer), name.c_str(), name.size() + 1);
        return CallGuest(c, fn, {0, buffer});
    };
    const u64 get_proc = call(driver, "nvnDeviceGetProcAddress");
    FILE* in = std::fopen(names, "r");
    char line[256];
    while (in && std::fgets(line, sizeof line, in)) {
        std::string name{line};
        while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) name.pop_back();
        u64 fn = call(get_proc, name);
        if (!fn) fn = call(driver, name);
        std::printf("%s %llx\n", name.c_str(), (unsigned long long)(fn ? fn - sdk_base : 0));
    }
    std::fflush(stdout);
    std::_Exit(0);
}

NATIVE("_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_") {
    graphics_alloc = c->x[0];
    graphics_free = c->x[1];
    graphics_realloc = c->x[2];
    graphics_user = c->x[3];
    // The driver's own allocator setup (its pure data functions allocate through it).
    CallOriginal(c, "_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_",
                 {graphics_alloc, graphics_free, graphics_realloc, graphics_user});
    if (const char* names = std::getenv("OOT_NVN_MAP")) MapNvnFunctions(c, names);
    if (std::getenv("OOT_NVN_INFO")) {
        // Device info values from the driver's query switch with a zeroed and a patterned device:
        // equal results are constants, different ones depend on device state.
        u64 sdk_base = 0;
        for (const auto& m : Modules())
            if (m.name == "sdk") sdk_base = m.base;
        const u64 device = ArenaAlloc(0x4000);
        for (u32 pname = 0; pname <= 0xa9; ++pname) {
            std::memset(reinterpret_cast<void*>(device), 0, 0x4000);
            const u32 a = (u32)CallGuest(c, sdk_base + 0x656210, {pname, device});
            std::memset(reinterpret_cast<void*>(device), 0x11, 0x4000);
            const u32 b = (u32)CallGuest(c, sdk_base + 0x656210, {pname, device});
            std::printf("%u %#x %#x %s\n", pname, a, b, a == b ? "const" : "device");
        }
        std::fflush(stdout);
        std::_Exit(0);
    }
}

// The console's graphics system (gsv/gpu/vi2 sessions) is not used by the native NVN implementation.
// nv::InitCheck (used by the video module) only requires the graphics memory to be recorded.
static void RecordGraphicsMemory(u64 memory) {
    for (const auto& m : Modules())
        if (m.name == "sdk") *reinterpret_cast<u64*>(m.base + 0x104b058) = memory ? memory : 1;
}
NATIVE("_ZN2nv18InitializeGraphicsEPvm") { RecordGraphicsMemory(c->x[0]); }
NATIVE("_ZN2nv18InitializeGraphicsEv") { RecordGraphicsMemory(0); }
