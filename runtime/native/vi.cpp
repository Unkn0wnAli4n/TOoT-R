// nn::vi: one 60 Hz SDR display with one layer. The vsync event is a kernel event attached to the
// game's SystemEventType through the SDK's own os code and signalled by a host timer thread.

#include <chrono>
#include <thread>

#include "../kernel.h"
#include "../native.h"

using namespace rt;

namespace {
u64 display_object[4], layer_object[4], native_window[64];
}

NATIVE("_ZN2nn2vi10InitializeEv") {}
NATIVE("_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE") {
    *Ptr<u64>(c, 0) = reinterpret_cast<u64>(display_object);
    Ret(c, 0);
}
NATIVE("_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE") {
    *Ptr<u64>(c, 0) = reinterpret_cast<u64>(layer_object);
    Ret(c, 0);
}
NATIVE("_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE") {
    *Ptr<u64>(c, 0) = reinterpret_cast<u64>(native_window);
    Ret(c, 0);
}
NATIVE("_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE") {
    static u32 writable = 0, readable = 0;
    if (!writable) {
        CreateKernelEvent(&writable, &readable);
        std::thread([] {
            auto next = std::chrono::steady_clock::now();
            for (;;) {
                next += std::chrono::nanoseconds(16666667);
                std::this_thread::sleep_until(next);
                SignalKernelEvent(writable);
            }
        }).detach();
    }
    // AttachReadableHandleToSystemEvent(event, handle, managed=false, clear mode=auto clear)
    CallExport(c, "_ZN2nn2os33AttachReadableHandleToSystemEventEPNS0_15SystemEventTypeEjbNS0_14EventClearModeE",
               {c->x[0], readable, 0, 1});
    Ret(c, 0);
}
NATIVE("_ZN2nn2vi21QueryDisplayHdrStatusEPbPNS0_15HdrOutputFormatEPtPKNS0_7DisplayE") {
    *Ptr<u8>(c, 0) = 0;
    *Ptr<u32>(c, 1) = 0;
    *Ptr<u16>(c, 2) = 0;
    Ret(c, 0);
}
// (the engine's frame rate controller keeps a rate above 60 only with this: present.cpp's real-time step)
namespace nvn { bool AboveSixty(); }
NATIVE("_ZN2nn2vi24QueryDisplayVrrSupportedEPbPKNS0_7DisplayE") { *Ptr<u8>(c, 0) = nvn::AboveSixty() ? 1 : 0; Ret(c, 0); }
NATIVE("_ZN2nn2vi25QueryDisplayHdrPaperWhiteEPtPKNS0_7DisplayE") { *Ptr<u16>(c, 0) = 203; Ret(c, 0); }
NATIVE("_ZN2nn2vi30QueryDisplayCurrentRefreshRateEPNS0_11RefreshRateEPbPKNS0_7DisplayE") {
    *Ptr<u32>(c, 0) = 60; // RefreshRate: Hz
    *Ptr<u8>(c, 1) = 0;
    Ret(c, 0);
}
NATIVE("_ZN2nn2vi23SetDisplayVrrCapabilityEPNS0_7DisplayEb") { Ret(c, 0); }
NATIVE("_ZN2nn2vi36SetDisplay120HzRefreshRateCapabilityEPNS0_7DisplayEb") { Ret(c, 0); }
NATIVE("_ZN2nn2vi28SetLayerExtendedSwapIntervalEPNS0_5LayerEii") { Ret(c, 0); }
NATIVE("_ZN2nn2vi30ClearLayerExtendedSwapIntervalEPNS0_5LayerE") { Ret(c, 0); }
NATIVE("_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE") { Ret(c, 0); }
NATIVE("_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE") { Ret(c, 0); }
NATIVE("_ZN2nn2vi52SetLayerNextHdrExtendedContentLuminanceRangeMetadataEPNS0_5LayerEtt") { Ret(c, 0); }
NATIVE("_ZN2nn2vi52SetLayerNextHdrStandardContentLuminanceRangeMetadataEPNS0_5LayerEtt") { Ret(c, 0); }
