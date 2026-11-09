#ifdef __linux__
#include <cstdio>
#include <sys/mman.h>
#endif
#include "graphics_settings.h"
// Native replacements for SDK services backed by system processes on the console. The game runs
// as a docked, focused, offline application with one preselected user.

#include <chrono>
#include <mutex>
#include <thread>

#include "../kernel.h"
#include "../native.h"

using namespace rt;

namespace {
constexpr u32 ResultNotAvailable = 0x2307C; // account: network service account unavailable
constexpr u32 ResultNetworkUnavailable = 0x3226E; // nifm-style failure
const auto boot_time = std::chrono::steady_clock::now();
}

// ---- start-up (nninitInitializeSdkModule) ----------------------------------------------------
NATIVE("_ZN2nn2lm10InitializeEv") {}
NATIVE("_ZN2nn2oe10InitializeEv") {}
NATIVE("_ZN2nn3aoc10InitializeEv") {}
NATIVE("_ZN2nn3err26InitializeContextWriterApiEv") {}
NATIVE("_ZN2nn4diag34InitializeApplicationAbortObserverEv") {}
NATIVE("_ZN2nn4pctl10InitializeEv") {}
NATIVE("_ZN2nn2fs24InitializeForApplicationEv") {}
NATIVE("_ZN2nn2fs28SetLocalApplicationAccessLogEb") {}
NATIVE("_ZN2nn2fs29SetResultHandledByApplicationEb") {}
NATIVE("nntimeStartup") {}

// ---- nn::oe (operating environment) ----------------------------------------------------------
NATIVE("_ZN2nn2oe15SetCpuBoostModeENS0_12CpuBoostModeE") {}
// Docked (console, boost) by default; OOT_HANDHELD=1 reports handheld (normal performance).
static bool Handheld() { static const bool on = std::getenv("OOT_HANDHELD") != nullptr || gfx::Current().handheld; return on; } // (read at start: the game sizes its targets once)
// OOT_RENDER_SIZE=<w>x<h> (experiment): the renderer's size pair in main's data (main+0x3f08250: 1920x1080 for both
// modes as shipped), written before the game reads it.
static void RenderSizeOverride() {
    static bool done = false; if (done) return; done = true;
    const char* e = std::getenv("OOT_RENDER_SIZE"); unsigned w = 0, h = 0;
    if (!e || std::sscanf(e, "%ux%u", &w, &h) != 2) return;
    u32* table = reinterpret_cast<u32*>(0x800000000ull + 0x3f08250);
    if (table[0] != 1920 || table[1] != 1080) { RT_LOG("render size: the table is not where expected (%u x %u)", table[0], table[1]); return; }
#ifdef __linux__
    mprotect(reinterpret_cast<void*>(0x800000000ull + 0x3f08000), 0x1000, PROT_READ | PROT_WRITE);
#endif
    table[0] = table[2] = w; table[1] = table[3] = h;
    RT_LOG("render size: %u x %u", w, h);
}
NATIVE("_ZN2nn2oe16GetOperationModeEv") { RenderSizeOverride(); Ret(c, Handheld() ? 0 : 1); }
NATIVE("_ZN2nn2oe18GetPerformanceModeEv") { Ret(c, Handheld() ? 0 : 1); }
NATIVE("_ZN2nn2oe17FinishStartupLogoEv") {}
NATIVE("_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE") {
    char* out = Ptr<char>(c, 0);
    std::memset(out, 0, 16);
    std::strcpy(out, "1.0.1");
}
NATIVE("_ZN2nn2oe18GetDesiredLanguageEv") {
    u64 code = 0;
    std::memcpy(&code, "en-US", 5); // settings::LanguageCode is the string packed in a u64
    Ret(c, code);
}
NATIVE("_ZN2nn2oe20GetCurrentFocusStateEv") { Ret(c, 1); } // in focus
NATIVE("_ZN2nn2oe20SetFocusHandlingModeENS0_17FocusHandlingModeE") {}
NATIVE("_ZN2nn2oe25TryPopNotificationMessageEPj") { Ret(c, 0); }
NATIVE("_ZN2nn2oe22PopNotificationMessageEv") {
    rt::CoreRelease release;
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1)); // no applet messages ever arrive
}
NATIVE("_ZN2nn2oe23GetCurrentGraphicsScaleEv") { Ret(c, 10); }
NATIVE("_ZN2nn2oe25GetProgramTotalActiveTimeEv") {
    Ret(c, (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - boot_time).count());
}
NATIVE("_ZN2nn2oe28SetResumeNotificationEnabledEb") {}
NATIVE("_ZN2nn2oe38SetUserInactivityDetectionTimeExtendedEb") {}
NATIVE("_ZN2nn2oe42SetGraphicsScaleChangedNotificationEnabledEb") {}
NATIVE("_ZN2nn2oe42SetOperationModeChangedNotificationEnabledEb") {}
NATIVE("_ZN2nn2oe44SetPerformanceModeChangedNotificationEnabledEb") {}

// ---- add-on content, errors, parental controls ------------------------------------------------
NATIVE("_ZN2nn3aoc17CountAddOnContentEv") { Ret(c, 0); }
NATIVE("_ZN2nn3aoc16ListAddOnContentEPiii") { Ret(c, 0); }
NATIVE("_ZN2nn3err9ShowErrorENS_6ResultE") { RT_LOG("game reported error result %#x", (u32)c->x[0]); }
NATIVE("_ZN2nn3err11NplnContext4MakeEPKvm") {}
NATIVE("_ZN2nn4pctl33CheckAgcPermissionUnderMinimumAgeERKNS_7account3UidE") { Ret(c, 1); }

// ---- network (the game runs offline) ----------------------------------------------------------
NATIVE("_ZN2nn4nifm10InitializeEv") { Ret(c, 0); }
NATIVE("_ZN2nn4nifm18IsNetworkAvailableEv") { Ret(c, 0); }
NATIVE("_ZN2nn4nifm20CancelNetworkRequestEv") {}
NATIVE("_ZN2nn4nifm27SubmitNetworkRequestAndWaitEv") {}
NATIVE("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE") { Ret(c, ResultNetworkUnavailable); }
NATIVE("_ZN2nn4nifm31HandleNetworkRequestErrorResultEv") { Ret(c, ResultNetworkUnavailable); }
NATIVE("_ZN2nn6socket10InitializeERKNS0_6ConfigE") { Ret(c, 0); }
NATIVE("_ZN2nn6socket13IsInitializedEv") { Ret(c, 1); }
NATIVE("_ZN2nn6socket8FinalizeEv") { Ret(c, 0); }
NATIVE("_ZN2nn3ssl10InitializeEv") { Ret(c, 0); }
NATIVE("_ZN2nn3ssl8FinalizeEv") { Ret(c, 0); }
NATIVE("_ZN2nn3nsd24GetEnvironmentIdentifierEPNS0_21EnvironmentIdentifierE") {
    std::memset(Ptr(c, 0), 0, 8);
    std::strcpy(Ptr<char>(c, 0), "lp1");
}
NATIVE("_ZN2nn3nsd9ResolveExEPNS0_4FqdnERKS1_") { Ret(c, ResultNetworkUnavailable); }
NATIVE("_ZN2nn2ro12ProtectRelroEPKvS2_S2_S2_S2_") {}

// ---- account: one local user, no network service account --------------------------------------
namespace {
constexpr u64 UserIdLow = 0x0123456789abcdefULL, UserIdHigh = 0x7ec0de0123456789ULL;
}
NATIVE("_ZN2nn7account10InitializeEv") {}
NATIVE("_ZN2nn7account22TryOpenPreselectedUserEPNS0_10UserHandleE") {
    u64* handle = Ptr<u64>(c, 0); // UserHandle: opaque words; keep the uid in it
    handle[0] = UserIdLow;
    handle[1] = UserIdHigh;
    handle[2] = 1;
    Ret(c, 1);
}
NATIVE("_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE") {
    u64* uid = Ptr<u64>(c, 0);
    uid[0] = UserIdLow;
    uid[1] = UserIdHigh;
    Ret(c, 0);
}
NATIVE("_ZN2nn7account9CloseUserERKNS0_10UserHandleE") {}
NATIVE("_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE") {
    char* out = Ptr<char>(c, 0);
    std::memset(out, 0, 33);
    std::strcpy(out, "Player");
    Ret(c, 0);
}
NATIVE("_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE") { Ret(c, ResultNotAvailable); }
NATIVE("_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE") { Ret(c, ResultNotAvailable); }
NATIVE("_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE") { Ret(c, ResultNotAvailable); }
NATIVE("_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE") { Ret(c, ResultNotAvailable); }

// ---- shared fonts: a host TrueType font stands in for the system font ---------------------------
namespace {
u64 FontAddress(u64* size) {
    static u64 address = 0, length = 0;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* candidates[] = {"/usr/share/fonts/noto/NotoSans-Regular.ttf",
                                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                                    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"};
        for (const char* path : candidates) {
            FILE* f = std::fopen(path, "rb");
            if (!f) continue;
            std::fseek(f, 0, SEEK_END);
            length = (u64)std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            address = ArenaAlloc(length, 0x1000);
            std::fread(reinterpret_cast<void*>(address), 1, length, f);
            std::fclose(f);
            RT_LOG("shared font: %s", path);
            return;
        }
        Fatal("no TrueType font found for the shared font");
    });
    *size = length;
    return address;
}
}
NATIVE("_ZN2nn2pl22GetSharedFontLoadStateENS0_14SharedFontTypeE") { Ret(c, 1); } // loaded
NATIVE("_ZN2nn2pl20GetSharedFontAddressENS0_14SharedFontTypeE") { u64 size; Ret(c, FontAddress(&size)); }
NATIVE("_ZN2nn2pl17GetSharedFontSizeENS0_14SharedFontTypeE") { u64 size; FontAddress(&size); Ret(c, size); }
NATIVE("_ZN2nn2pl17RequestLoadNxFontEPvm") { Ret(c, 0); }

// ---- play reports: built by the recompiled SDK, never transmitted --------------------------------
namespace {
// A guest nn::os::SystemEvent created through the SDK's own os code (optionally signalled).
u64 MakeSystemEvent(Cpu* c, bool signalled) {
    const u64 event = ArenaAlloc(0x40, 16);
    CallExport(c, "_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb", {event, 0, 0});
    if (signalled) CallExport(c, "_ZN2nn2os17SignalSystemEventEPNS0_15SystemEventTypeE", {event});
    return event;
}
}
NATIVE("_ZN2nn5prepo10PlayReport4SaveEv") { Ret(c, 0); }
NATIVE("_ZN2nn5prepo10PlayReport4SaveERKNS_7account3UidE") { Ret(c, 0); }
NATIVE("_ZN2nn5prepo10PlayReport4SendERKNS_7account3UidE") { Ret(c, 0); }
NATIVE("_ZN2nn5prepo28RequestImmediateTransmissionEv") { Ret(c, 0); }
NATIVE("_ZN2nn5prepo27AddInternetConnectionStatusEPNS0_10PlayReportEPKcS4_S4_") { Ret(c, 0); }
NATIVE("_ZN2nn5prepo30GetOverSaveFrequencyLimitEventEv") {
    static u64 event = 0;
    if (!event) event = MakeSystemEvent(c, false);
    Ret(c, event);
}
NATIVE("_ZN2nn5prepo12AsyncContext9GetResultEv") { Ret(c, 0); }
NATIVE("_ZN2nn5prepo12AsyncContextD1Ev") {}
NATIVE("_ZN2nn5prepo12AsyncContext18GetCompletionEventEPNS_2os11SystemEventE") {
    // The SystemEvent wraps its SystemEventType at offset 0; create it in place, already signalled
    // (reports complete at once). The argument is read before the guest calls clobber x1.
    const u64 out = c->x[1];
    CallExport(c, "_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb", {out, 0, 0});
    CallExport(c, "_ZN2nn2os17SignalSystemEventEPNS0_15SystemEventTypeE", {out});
}

// ---- amiibo (nfp): no reader is present ------------------------------------------------------------
namespace {
constexpr u32 ResultNfcDeviceNotFound = 0x8073; // module 115 (nfp), description 64
}
NATIVE("_ZN2nn3nfp10InitializeEv") { Ret(c, 0); }
NATIVE("_ZN2nn3nfp8FinalizeEv") { Ret(c, 0); }
NATIVE("_ZN2nn3nfp11ListDevicesEPNS0_12DeviceHandleEPii") { *Ptr<s32>(c, 1) = 0; Ret(c, ResultNfcDeviceNotFound); }
NATIVE("_ZN2nn3nfp29AttachAvailabilityChangeEventEPNS_2os15SystemEventTypeE") {
    CallExport(c, "_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb", {c->x[0], 0, 0});
    Ret(c, 0);
}
NATIVE("_ZN2nn3nfp14GetDeviceStateERKNS0_12DeviceHandleE") { Ret(c, 0); }
#define NFP_NO_DEVICE(name) NATIVE(name) { Ret(c, ResultNfcDeviceNotFound); }
NFP_NO_DEVICE("_ZN2nn3nfp10GetTagInfoEPNS0_7TagInfoERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp13GetCommonInfoEPNS0_10CommonInfoERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp13StartRestorerEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE")
NFP_NO_DEVICE("_ZN2nn3nfp13StopDetectionERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp14StartDetectionERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp15GetRegisterInfoEPNS0_12RegisterInfoERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp18GetApplicationAreaEPvPmRKNS0_12DeviceHandleEm")
NFP_NO_DEVICE("_ZN2nn3nfp18SetApplicationAreaERKNS0_12DeviceHandleEPKvm")
NFP_NO_DEVICE("_ZN2nn3nfp19OpenApplicationAreaERKNS0_12DeviceHandleEj")
NFP_NO_DEVICE("_ZN2nn3nfp19StartGameDataEraserEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE")
NFP_NO_DEVICE("_ZN2nn3nfp21CreateApplicationAreaERKNS0_12DeviceHandleERKNS0_25ApplicationAreaCreateInfoE")
NFP_NO_DEVICE("_ZN2nn3nfp23RecreateApplicationAreaERKNS0_12DeviceHandleERKNS0_25ApplicationAreaCreateInfoE")
NFP_NO_DEVICE("_ZN2nn3nfp29StartNicknameAndOwnerSettingsEPNS0_12DeviceHandleEPbPNS0_12RegisterInfoERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE")
NFP_NO_DEVICE("_ZN2nn3nfp29StartNicknameAndOwnerSettingsEPNS0_12DeviceHandleEPbPNS0_12RegisterInfoERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoERKS4_")
NFP_NO_DEVICE("_ZN2nn3nfp5FlushERKNS0_12DeviceHandleE")
NFP_NO_DEVICE("_ZN2nn3nfp5MountERKNS0_12DeviceHandleENS0_9ModelTypeENS0_11MountTargetE")
NFP_NO_DEVICE("_ZN2nn3nfp9GetNpadIdEPjRKNS0_12DeviceHandleE")
#undef NFP_NO_DEVICE
