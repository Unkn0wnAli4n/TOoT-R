// Native NVN device: creating a device brings up the Vulkan device and the game window.

#include <atomic>
#include <chrono>
#include "../../native.h"
#include "texture.h"
#include "vk.h"

using namespace rt;

namespace {

// nvnDeviceGetInteger values. Constants come from the SDK driver's own query table; entries
// that the driver reads from device state get explicit values.
constexpr u32 kDeviceInfo[] = {0xe7, 0x28, 0xe, 0x10000, 0x100, 0x8, 0x20, 0x4, 0x10, 0x20, 0x10, 0x4, 0x4, 0x4, 0x20, 0x20, 0x20, 0x100, 0x100, 0x4, 0x8, 0x200, 0x100, 0, 0, 0x1000, 0x4000, 0x10, 0x400, 0x400, 0x40, 0x400, 0x7fffffff, 0xffff, 0xffff, 0x8, 0x100000, 0x1000, 0x10, 0x1000, 0x1, 0x1, 0x1, 0, 0x10, 0, 0xc9, 0xc9, 0x14, 0, 0x1, 0x8, 0x8, 0x4, 0x20, 0, 0, 0x80, 0, 0x1, 0x4, 0, 0, 0, 0, 0, 0x1, 0x1, 0x1, 0x1, 0x5, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x10, 0x800, 0x20, 0x1000, 0x10000, 0x10000, 0x100, 0x4000, 0x40000, 0x1000, 0x1, 0x4, 0x2, 0, 0x1000, 0x4000, 0x1000, 0x80, 0x20, 0xffff, 0x1, 0, 0x100, 0x10, 0xffff, 0xffff, 0x100, 0x1, 0x10, 0x10, 0x10, 0x1, 0x1, 0x1, 0x100, 0, 0x10, 0x8, 0x10, 0x20, 0x1f, 0x20000000, 0x1000000, 0x1000000, 0x40000000, 0x20, 0x1, 0x40, 0, 0x1, 0x7fffffff, 0x20, 0x20, 0x3fffff, 0x200, 0x100, 0x20, 0x3fff, 0, 0x8, 0, 0, 0x4, 0x100, 0x10, 0, 0x1, 0x1f, 0, 0x40, 0x1, 0xe400, 0x6000, 0, 0x100, 0x40000, 0, 0, 0, 0xc9, 0xe, 0xc9, 0x14, 0x28000, 0x8, 0x4, 0x100, 0x100, 0x4, 0x10000, 0xc9, 0x5, 0x1};
u32 DeviceInfo(u32 pname) {
    switch (pname) {
    case 23: return 0x80;   // shader scratch memory scale factor (minimum)
    case 24: return 0x100;  // shader scratch memory scale factor (recommended)
    case 55: return 0x100;
    case 135: return 0x11;
    case 154: return 1;
    case 43: RT_LOG("nvnDeviceGetInteger: device-dependent info 43 queried"); return 0;
    case 45: RT_LOG("nvnDeviceGetInteger: device-dependent info 45 queried"); return 0;
    case 56: RT_LOG("nvnDeviceGetInteger: device-dependent info 56 queried"); return 0;
    case 58: RT_LOG("nvnDeviceGetInteger: device-dependent info 58 queried"); return 0;
    case 61: RT_LOG("nvnDeviceGetInteger: device-dependent info 61 queried"); return 0;
    case 62: RT_LOG("nvnDeviceGetInteger: device-dependent info 62 queried"); return 0;
    case 63: RT_LOG("nvnDeviceGetInteger: device-dependent info 63 queried"); return 0;
    case 64: RT_LOG("nvnDeviceGetInteger: device-dependent info 64 queried"); return 0;
    case 65: RT_LOG("nvnDeviceGetInteger: device-dependent info 65 queried"); return 0;
    case 98: RT_LOG("nvnDeviceGetInteger: device-dependent info 98 queried"); return 0;
    case 153: RT_LOG("nvnDeviceGetInteger: device-dependent info 153 queried"); return 0;
    default: return pname < sizeof kDeviceInfo / sizeof kDeviceInfo[0] ? kDeviceInfo[pname] : 0;
    }
}

} // namespace

namespace nvn {
u64 device_object = 0;
}

NATIVE("nvnDeviceInitialize") {
    auto& vk = nvn::Vk();
    // The driver's layout math gates sizes on bit 0 of its per-format support
    // cache. Native device creation replaces the routine that normally fills it.
    // Populate that bit from actual Vulkan support, retaining the driver's layout.
    u64 sdk_base = 0;
    for (const auto& module : Modules()) if (module.name == "sdk") sdk_base = module.base;
    if (sdk_base) {
        for (u32 format = 1; format <= 162; ++format) {
            if (format > 80 && format < 109) continue;
            if (format > 156 && format < 161) continue;
            const auto info = nvn::GetFormat(format);
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(vk.physical, info.vk, &properties);
            auto* flags = reinterpret_cast<u32*>(sdk_base + 0xeb4528 + format * 0x30);
            if (properties.optimalTilingFeatures || properties.bufferFeatures || (format >= 121 && format <= 148)) *flags |= 1u;
            else *flags &= ~1u;
        }
    }
    // +0 window origin mode and +4 depth mode (read by the pure Get/Set functions): NVN defaults,
    // LOWER_LEFT and NEAR_IS_MINUS_W.
    u32* modes = Ptr<u32>(c, 0);
    modes[0] = 0;
    modes[1] = 0;
    nvn::device_object = c->x[0];
    // Pure driver layout functions read the GPU page alignment and default memory
    // kind from the device. Leaving alignment at zero rounds every ordinary size
    // down to zero when ChooseKind selects the generic uncompressed kind.
    *reinterpret_cast<u32*>(c->x[0] + 0xd4) = 0x10000;
    *reinterpret_cast<u32*>(c->x[0] + 0xd8) = 0x10000;
    *reinterpret_cast<u32*>(c->x[0] + 0x1d98) = 0xfe;
    Ret(c, 1);
}
NATIVE("nvnDeviceFinalize") {}
NATIVE("nvnDeviceGetInteger") {
    const u32 pname = (u32)c->x[1];
    *Ptr<s32>(c, 2) = (s32)DeviceInfo(pname);
    static const bool trace = std::getenv("OOT_NVN_TRACE") != nullptr;
    if (trace) RT_LOG("nvnDeviceGetInteger(%u) = %#x", pname, DeviceInfo(pname));
}

// Fast-clear registration is an optional console compression optimization. Vulkan
// chooses clear compression itself; report registration unavailable so callers use
// ordinary clears, rather than entering the console driver's NvRm device path.
NATIVE("nvnDeviceRegisterFastClearColor") { Ret(c, 0); }
NATIVE("nvnDeviceRegisterFastClearColori") { Ret(c, 0); }
NATIVE("nvnDeviceRegisterFastClearColorui") { Ret(c, 0); }
NATIVE("nvnDeviceRegisterFastClearDepth") { Ret(c, 0); }
NATIVE("nvnDeviceRegisterFastClearStencil") { Ret(c, 0); }
NATIVE("nvnDeviceRegisterFastClearColorIndex") { Ret(c, 0xffffffffu); }
NATIVE("nvnDeviceRegisterFastClearColoriIndex") { Ret(c, 0xffffffffu); }
NATIVE("nvnDeviceRegisterFastClearColoruiIndex") { Ret(c, 0xffffffffu); }
NATIVE("nvnDeviceRegisterFastClearDepthIndex") { Ret(c, 0xffffffffu); }
NATIVE("nvnDeviceRegisterFastClearStencilIndex") { Ret(c, 0xffffffffu); }
// Handle packing matches the SDK driver's pure arithmetic entry points.
NATIVE("nvnDeviceGetTextureHandle") { Ret(c, c->x[1] ? (u64)(s64)(s32)c->x[1] | ((u64)(s64)(s32)c->x[2] << 20) : 0); }
NATIVE("nvnDeviceGetTexelFetchHandle") { Ret(c, (s64)(s32)c->x[1]); }
NATIVE("nvnDeviceGetSeparateTextureHandle") { Ret(c, (s64)(s32)c->x[1]); }
NATIVE("nvnDeviceGetSeparateSamplerHandle") { Ret(c, (u64)(s64)(s32)c->x[1] << 20); }
NATIVE("nvnDeviceGetImageHandle") { Ret(c, (s64)(s32)c->x[1]); }

// NvRm memory kind selection (used by the driver's texture layout code): always the generic
// uncompressed block-linear kind; compression is the host GPU's business.
NATIVE("NvRmGpuDeviceChooseKind") {
    *Ptr<u32>(c, 2) = 0xFE;
    Ret(c, 0);
}

// Display (window) texture requirements, normally computed by the graphics service: block-linear
// RGBA8 with 128-row blocks. TexturePreferenceInfoV2: +8 width, +0xc height. Result through x8:
// {u64 status, u64 size, u64 alignment}.
NATIVE("_ZN2nn3wsi5ounce6native12NativeWindow20CalculateRequirementERKNS0_2nv23TexturePreferenceInfoV2E") {
    const u32 width = *reinterpret_cast<u32*>(c->x[0] + 8), height = *reinterpret_cast<u32*>(c->x[0] + 0xc);
    const u64 pitch = ((u64)width * 4 + 63) & ~63ULL;
    const u64 size = pitch * (((u64)height + 127) & ~127ULL);
    u64* out = reinterpret_cast<u64*>(c->x[8]);
    out[0] = 0;
    out[1] = size;
    out[2] = 64 * 128;
}

namespace nvn { u64 DriverFunction(const char* name); }
// Texture layout queries run the driver's own layout math (see NvRmGpuDeviceChooseKind above).
NATIVE("nvnTextureBuilderGetStorageSize") {
    const u64 size = CallGuest(c, nvn::DriverFunction("nvnTextureBuilderGetStorageSize"), {c->x[0]});
    static const bool trace = std::getenv("OOT_NVN_TRACE") != nullptr;
    if (trace) RT_LOG("  storage size %#llx", (unsigned long long)size);
    Ret(c, size);
}
NATIVE("nvnTextureBuilderGetStorageAlignment") {
    u64 align = CallGuest(c, nvn::DriverFunction("nvnTextureBuilderGetStorageAlignment"), {c->x[0]});
    if (!align) align = 0x10000; // the driver reads this from console device state
    static const bool trace = std::getenv("OOT_NVN_TRACE") != nullptr;
    if (trace) RT_LOG("  storage alignment %#llx", (unsigned long long)align);
    Ret(c, align);
}

// ---- timestamps, debugging aids and device queries ------------------------------------------------
namespace nvn { u64 GuestTimestampNs(); }
// Counter reports are {u64 value, u64 timestamp in ns} (see ReportCounter).
NATIVE("nvnDeviceGetTimestampInNanoseconds") {
    static const bool trace = std::getenv("OOT_SYNC_TRACE") != nullptr;
    if (trace) { static u64 n = 0, last = 0; const u64 v = Ptr<const u64>(c, 1)[1]; if (++n % 50 == 0) RT_LOG("report timestamp read %#llx: %llu (delta to previous read %lld us, now-report %lld us)", (unsigned long long)c->x[1], (unsigned long long)v, (long long)(v - last) / 1000, (long long)(nvn::GuestTimestampNs() - v) / 1000); last = v; }
    Ret(c, Ptr<const u64>(c, 1)[1]);
}
NATIVE("nvnDeviceGetCurrentTimestampInNanoseconds") { Ret(c, nvn::GuestTimestampNs()); }
NATIVE("nvnDeviceGetFlags") { Ret(c, 0); }
NATIVE("nvnDeviceSetDebugLabel") {}
NATIVE("nvnDeviceInstallDebugCallback") {}
NATIVE("nvnInstallGlobalDebugCallback") {}
NATIVE("nvnDeviceSetIntermediateShaderCache") {}
NATIVE("nvnDeviceSetAutomaticCheckpointsByDebugGroup") {}
NATIVE("nvnDeviceSetAutomaticCheckpointsEnable") {}
NATIVE("nvnDeviceSetAutomaticCheckpointsMask") {}
NATIVE("nvnDeviceSetAutomaticCheckpointsSamplingInterval") {}
NATIVE("nvnDeviceSetCheckpointCallStacksEnable") {}
NATIVE("nvnDeviceSetCheckpointMaxCallStacksDepth") {}
NATIVE("nvnDeviceIsExternalDebuggerAttached") { Ret(c, 0); }
NATIVE("nvnDeviceGenerateDebugDomainId") {
    static std::atomic<u32> next{1};
    Ret(c, next++);
}
// DLSS networks need no host resources: evaluation is an upscale of the input colour.
NATIVE("nvnDlssNetworkInitialize") { Ret(c, 1); }
NATIVE("nvnDlssNetworkResize") { Ret(c, 1); }
NATIVE("nvnDlssNetworkFinalize") {}
