// nn::codec's hardware video decoders on FFmpeg (video_hookshot.inc, from Hookshot): the game's movie library
// (subsdk0 movie:: over the SDK's nn::movie) runs as recompiled code and decodes through these. AV1 (the game's
// movies) uses libdav1d. HOS_VIDEO_LOG=1: submissions and acquisitions.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <unordered_map>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include "../kernel.h"
#include "../native.h"
using namespace rt;

namespace {
using Ctx = Cpu;
#define ARG(n) (c->x[n])
#define RET(v) (c->x[0] = (uint64_t)(v))
#define GP(a) ((void*)(uintptr_t)(a))
#define G64(a) (*(uint64_t*)GP(a))
#define G32(a) (*(uint32_t*)GP(a))
#define VIDEO_CAT2(a, b) a##b
#define VIDEO_CAT(a, b) VIDEO_CAT2(a, b)
#define VIDEO_DEFINE(name, id)                                                                \
    static void VIDEO_CAT(video_fn_, id)(Cpu * c);                                            \
    static ::rt::NativeRegistration VIDEO_CAT(video_reg_, id){name, VIDEO_CAT(video_fn_, id)}; \
    static void VIDEO_CAT(video_fn_, id)([[maybe_unused]] Cpu * c)
#define HLE(sym) VIDEO_DEFINE(#sym, __COUNTER__)
#define hos_log(...) RT_LOG(__VA_ARGS__)
#define HOS_WRITTEN(p, n) ((void)0) // (CPU writes to tracked guest memory are seen by the write-protection handler)
void hos_sleep_ns(int64_t ns) { if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); }
int64_t hos_now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
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
// (Media Foundation's AV1 decoder is Windows-only: libdav1d decodes AV1 here, so these are never reached.)
typedef void (*mf_av1_picture_fn)(void* user, const uint8_t* y, const uint8_t* uv, int w, int h, int stride, int64_t pts);
void* mf_av1_open(int, int) { return nullptr; }
void mf_av1_decode(void*, const uint8_t*, size_t, int64_t, mf_av1_picture_fn, void*) {}
void mf_av1_flush(void*) {}
void mf_av1_close(void*) {}
} // namespace

#include "video_hookshot.inc"

// ---- nn::syncpt::sdk (from Hookshot, host/nn_syncpt.c): GPU sync points and their events. The SDK and the game's
// movie library import them from a system library the image does not carry (their calls went to address 0). GPU
// work completes in order here, so every sync point counts as reached; a sync point event is a system event,
// signalled when it is requested.
HLE(_ZN2nn6syncpt17GetLibraryVersionEv) { RET(1); }
HLE(_ZN2nn6syncpt3sdk12InitializeExEPvmNS1_14LibraryOptionsE) { RET(0); }
HLE(_ZN2nn6syncpt3sdk13WaitSyncPointERKNS1_15SyncPointHandleE) { RET(0); }
HLE(_ZN2nn6syncpt3sdk16TryWaitSyncPointERKNS1_15SyncPointHandleE) { RET(1); }
HLE(_ZN2nn6syncpt3sdk18TimedWaitSyncPointERKNS1_15SyncPointHandleENS_8TimeSpanE) { RET(1); }
HLE(_ZN2nn6syncpt3sdk20CreateSyncPointEventEPNS1_18SyncPointEventTypeE) { hos_system_event_init(ARG(0), 0); RET(0); }
HLE(_ZN2nn6syncpt3sdk21DestroySyncPointEventEPNS1_18SyncPointEventTypeE) {}
HLE(_ZN2nn6syncpt3sdk21RequestSyncPointEventEPNS1_18SyncPointEventTypeERKNS1_15SyncPointHandleE) { hos_system_event_signal(ARG(0)); RET(0); }
HLE(_ZN2nn6syncpt3sdk20CancelSyncPointEventEPNS1_18SyncPointEventTypeE) {}
HLE(_ZN2nn6syncpt3sdk21TryWaitSyncPointEventEPKNS1_18SyncPointEventTypeE) { RET(1); }
HLE(_ZN2nn6syncpt3sdk25InitializeMultiWaitHolderEPNS_2os19MultiWaitHolderTypeEPNS1_18SyncPointEventTypeE) {
    CallExport(c, "_ZN2nn2os25InitializeMultiWaitHolderEPNS0_19MultiWaitHolderTypeEPNS0_15SystemEventTypeE", {c->x[0], c->x[1]});
}
HLE(_ZN2nn6syncpt3sdk32MakeNvRmFenceFromSyncPointHandleEPjS2_RKNS1_15SyncPointHandleE) { if (ARG(0)) G32(ARG(0)) = 0; if (ARG(1)) G32(ARG(1)) = 0; }
HLE(_ZN2nn6syncpt3sdk32MakeSyncPointHandleFromNvRmFenceEjj) { RET(0); }
HLE(_ZN2nn6syncpt3sdk39MakeNativeSyncHandleFromSyncPointHandleERKNS1_15SyncPointHandleE) { RET(0); }
