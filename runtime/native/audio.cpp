// nn::audio: Hookshot's audio renderer HLE (audio_hookshot.inc: the game's sound graph mixed in software
// and played through SDL3), with a small compatibility layer for this runtime, plus the device-side
// calls Hookshot has no version of. OOT_AUDIO=0 (or HOS_AUDIO=0): silence; HOS_AUDIO_DUMP=<file.wav>.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

#include <SDL3/SDL.h>

#include "../kernel.h"
#include "../native.h"
using namespace rt;

namespace {
// ---- what audio_hookshot.inc expects from its host -----------------------------------------------
using Ctx = Cpu;
#define ARG(n) (c->x[n])
#define RET(v) (c->x[0] = (uint64_t)(v))
#define GP(a) ((void*)(uintptr_t)(a))
#define G64(a) (*(uint64_t*)GP(a))
#define G32(a) (*(uint32_t*)GP(a))
#define HLE_CAT2(a, b) a##b
#define HLE_CAT(a, b) HLE_CAT2(a, b)
#define HLE_DEFINE(name, id)                                                                  \
    static void HLE_CAT(audio_fn_, id)(Cpu * c);                                              \
    static ::rt::NativeRegistration HLE_CAT(audio_reg_, id){name, HLE_CAT(audio_fn_, id)};    \
    static void HLE_CAT(audio_fn_, id)([[maybe_unused]] Cpu * c)
#define HLE(sym) HLE_DEFINE(#sym, __COUNTER__)
#define AUDIO_NATIVE(name) HLE_DEFINE(name, __COUNTER__)
#define hos_log(...) RT_LOG(__VA_ARGS__)
int64_t hos_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void hos_sleep_ns(int64_t ns) { if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); }
// System events keyed by the game's SystemEventType address (as Hookshot keys them): a kernel event
// attached to it, signalled from the mixer thread.
std::mutex events_mutex;
std::unordered_map<uint64_t, u32> event_handles;
void EventInit(Cpu* c, uint64_t key, int autoclear) {
    u32 writable, readable; CreateKernelEvent(&writable, &readable);
    CallExport(c, "_ZN2nn2os17AttachSystemEventEPNS0_15SystemEventTypeEjbjbNS0_14EventClearModeE",
               {key, readable, 1, writable, 1, (u64)(autoclear ? 1 : 0)});
    std::lock_guard lock{events_mutex}; event_handles[key] = writable;
}
void EventSignal(uint64_t key) {
    u32 handle = 0;
    { std::lock_guard lock{events_mutex}; auto it = event_handles.find(key); if (it != event_handles.end()) handle = it->second; }
    if (handle) SignalKernelEvent(handle);
}
#define hos_system_event_init(key, autoclear) EventInit(c, key, autoclear)
#define hos_system_event_signal(key) EventSignal(key)
std::atomic<unsigned long long> audio_updates{0};
void AudioStubLog(const char* name) {
    static std::mutex m; static std::set<std::string> told;
    std::lock_guard lock{m}; if (told.insert(name).second) RT_LOG("audio: %s is not modelled (answered 0)", name);
}
} // namespace

#include "audio_hookshot.inc"
#include "audio_stubs.inc"

// ---- device-side calls (no Hookshot version) -----------------------------------------------------------
AUDIO_NATIVE("_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleERKNS0_22AudioRendererParameterEPvm") { G64(ARG(0)) = 1; Ret(c, 0); }
AUDIO_NATIVE("_ZN2nn5audio21GetAudioRendererStateENS0_19AudioRendererHandleE") { Ret(c, renderer_running ? 0 : 1); }
AUDIO_NATIVE("_ZN2nn5audio34SetAudioRendererVoiceDropParameterENS0_19AudioRendererHandleEf") {}
AUDIO_NATIVE("_ZN2nn5audio34GetAudioRendererVoiceDropParameterENS0_19AudioRendererHandleE") { const float v = 1; std::memcpy(&c->v[0], &v, 4); }
AUDIO_NATIVE("_ZN2nn5audio29ExecuteAudioRendererRenderingENS0_19AudioRendererHandleE") { Ret(c, 0); }
AUDIO_NATIVE("_ZN2nn5audio26GetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameE") { const float v = 1; std::memcpy(&c->v[0], &v, 4); }
AUDIO_NATIVE("_ZN2nn5audio39IsDeviceOutputVolumeAutoTuneModeEnabledEv") { Ret(c, 0); }
AUDIO_NATIVE("_ZN2nn5audio40SetDeviceOutputVolumeAutoTuneModeEnabledEb") {}
AUDIO_NATIVE("_ZN2nn5audio44SetCallbackFuncForSetAudioDeviceOutputVolumeEPFvPKNS0_15AudioDeviceNameEfmEm") {}
AUDIO_NATIVE("_ZN2nn5audio58SetCallbackFuncForSetDeviceOutputVolumeAutoTuneModeEnabledEPFvbmEm") {}
AUDIO_NATIVE("_ZN2nn5audio36AcquireAudioOutputDeviceNotificationEPNS_2os11SystemEventE") { EventInit(c, c->x[0], 1); Ret(c, 0); }
AUDIO_NATIVE("_ZN2nn5audio36ReleaseAudioOutputDeviceNotificationEPNS_2os11SystemEventE") { CallExport(c, "_ZN2nn2os18DestroySystemEventEPNS0_15SystemEventTypeE", {c->x[0]}); }
AUDIO_NATIVE("_ZN2nn5audio24GetActiveAudioDeviceNameEPNS0_15AudioDeviceNameE") { std::memset(Ptr(c, 0), 0, 0x100); std::strcpy(Ptr<char>(c, 0), "AudioDefaultOutput"); }
// No capture/injection device is connected.
AUDIO_NATIVE("_ZN2nn5audio23ListVirtualInputDevicesEPNS0_15AudioDeviceNameEi") { Ret(c, 0); }
AUDIO_NATIVE("_ZN2nn5audio27ListVirtualInjectionDevicesEPNS0_15AudioDeviceNameEi") { Ret(c, 0); }
AUDIO_NATIVE("_ZN2nn5audio24ListVirtualOutputDevicesEPNS0_15AudioDeviceNameEi") { Ret(c, 0); }

// Hang diagnosis (kernel.cpp watchdog): renderer updates so far.
extern "C" unsigned long long AudioUpdateCount() { return (unsigned long long)elapsed_frames; }
