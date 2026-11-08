// movie::VideoDecoder / movie::AudioDecoder (subsdk0) without the console's hardware decoder.
//
// Movies play by default: every hook below forwards to the game's own (recompiled) movie library,
// which decodes through nn::codec (video.cpp). OOT_MOVIES=0 skips movies instead: the decoders
// accept input and hand back one of the game's registered frames marked end-of-stream, so playback
// finishes at once.

#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>

#include "../native.h"
#include <cstdlib>
#include <cstring>
// OOT_MOVIES=0: skip movies (end every stream at once) instead of playing them.
static bool MoviesOn() { static const bool on = !std::getenv("OOT_MOVIES") || std::atoi(std::getenv("OOT_MOVIES")) != 0; return on; }
// OOT_MOVIE_TRACE=1: every movie:: call (development).
static void MovieTrace(const char* name, Cpu* c) {
    static const bool on = std::getenv("OOT_MOVIE_TRACE") != nullptr;
    if (on && !std::strstr(name, "Acquire") && !std::strstr(name, "ReadSamples") && !std::strstr(name, "DecodeAsync") && !std::strstr(name, "Release")) RT_LOG("movie: %s(%#llx, %#llx)", name, (unsigned long long)c->x[0], (unsigned long long)c->x[1]);
}
// Every hook: the game's own implementation when movies play.
#define MOVIE_NATIVE(name) static void MOVIE_CAT(movie_skip_, __LINE__)(Cpu* c); \
    NATIVE(name) { if (MoviesOn()) { MovieTrace(name, c); c->x[0] = CallOriginal(c, name); return; } MOVIE_CAT(movie_skip_, __LINE__)(c); } \
    static void MOVIE_CAT(movie_skip_, __LINE__)([[maybe_unused]] Cpu* c)
#define MOVIE_CAT2(a, b) a##b
#define MOVIE_CAT(a, b) MOVIE_CAT2(a, b)

using namespace rt;

namespace {

struct Decoder {
    std::vector<u64> frames;
    u64 input_event = 0, output_event = 0;
    std::atomic<bool> end_delivered{false}; // the end-of-stream frame has been handed out
    std::atomic<bool> input_received{false};
};

std::mutex decoders_mutex;
std::unordered_map<u64, Decoder> decoders; // guest decoder object -> state

Decoder& Get(u64 object) {
    std::lock_guard lock{decoders_mutex};
    return decoders[object];
}

// A guest SystemEvent that stays signalled (manual clear), made by the SDK's own os code.
u64 SignalledEvent(Cpu* c) {
    const u64 event = ArenaAlloc(0x40, 16);
    CallExport(c, "_ZN2nn2os17CreateSystemEventEPNS0_15SystemEventTypeENS0_14EventClearModeEb", {event, 0, 0});
    CallExport(c, "_ZN2nn2os17SignalSystemEventEPNS0_15SystemEventTypeE", {event});
    return event;
}

// Arguments are read before any guest call, which clobbers the argument registers.
void InputEvent(Cpu* c) {
    const u64 object = c->x[0], out = c->x[1];
    Decoder& d = Get(object);
    if (!d.input_event) d.input_event = SignalledEvent(c);
    *reinterpret_cast<u64*>(out) = d.input_event;
    Ret(c, 0);
}
void OutputEvent(Cpu* c) {
    const u64 object = c->x[0], out = c->x[1];
    Decoder& d = Get(object);
    if (!d.output_event) d.output_event = SignalledEvent(c);
    *reinterpret_cast<u64*>(out) = d.output_event;
    Ret(c, 0);
}
void Register(Cpu* c) {
    Decoder& d = Get(c->x[0]);
    for (s32 i = 0; i < (s32)c->x[2]; ++i) d.frames.push_back(Ptr<u64>(c, 1)[i]);
    Ret(c, 0);
}
// Returns the first registered frame marked end-of-stream, once; afterwards no frame is ready
// (players drain the decoder until Acquire yields null).
u64 EndOfStreamFrame(Cpu* c, u64 object, const char* set_eos) {
    Decoder& d = Get(object);
    if (d.frames.empty()) Fatal("movie decoder has no registered frames");
    // Do not finish playback before the extractor has delivered its first
    // input. Otherwise shutdown can race an outstanding ReadSamples request.
    if (!d.input_received) return 0;
    if (d.end_delivered.exchange(true)) return 0;
    const u64 frame = d.frames[0];
    CallExport(c, set_eos, {frame, 1});
    return frame;
}

} // namespace

// The decoder boundary currently ends streams without displaying movie frames.
// End extraction at the same boundary so playback shutdown cannot leave a
// synchronous ReadSamples message waiting on the console extractor worker.
MOVIE_NATIVE("_ZN2nn5movie14MediaExtractor11ReadSamplesEPNS0_6BufferEi") {
    const u64 buffer=c->x[1];
    CallExport(c,"_ZN2nn5movie6Buffer19SetFilledBufferSizeEm",{buffer,0});
    CallExport(c,"_ZN2nn5movie6Buffer12SetTimeStampENS_8TimeSpanE",{buffer,0});
    CallExport(c,"_ZN2nn5movie6Buffer14SetBufferFlagsENS0_11BufferFlagsE",{buffer,8});
    Ret(c,0);
}

#define DECODER_OK(name) MOVIE_NATIVE(name) { Ret(c, 0); }
DECODER_OK("_ZN5movie12VideoDecoderC1Ev")
DECODER_OK("_ZN5movie12VideoDecoderD1Ev")
DECODER_OK("_ZN5movie12VideoDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24VideoDecoderOutputFormatENS_14VideoCodecModeE")
DECODER_OK("_ZN5movie12VideoDecoder9ConfigureEPKN2nn5movie13MediaPropertyEPKvm")
DECODER_OK("_ZN5movie12VideoDecoder17ReleaseVideoFrameEPNS_10VideoFrameE")
DECODER_OK("_ZN5movie12VideoDecoder4StopEv")
DECODER_OK("_ZN5movie12AudioDecoderC1Ev")
DECODER_OK("_ZN5movie12AudioDecoderD1Ev")
DECODER_OK("_ZN5movie12AudioDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24AudioDecoderOutputFormatENS_14AudioCodecModeE")
DECODER_OK("_ZN5movie12AudioDecoder9ConfigureEPKN2nn5movie13MediaPropertyEPKvm")
DECODER_OK("_ZN5movie12AudioDecoder17ReleaseAudioFrameEPNS_10AudioFrameE")
DECODER_OK("_ZN5movie12AudioDecoder25GetAudioFrameRequirementsEPN2nn5movie13MediaPropertyE")
DECODER_OK("_ZN5movie12AudioDecoder4StopEv")
#undef DECODER_OK
// Start and Flush begin a new (immediately ending) stream.
#define DECODER_RESTART(name) MOVIE_NATIVE(name) { auto& d=Get(c->x[0]);d.end_delivered=false;d.input_received=false;Ret(c,0); }
DECODER_RESTART("_ZN5movie12VideoDecoder5StartEv")
DECODER_RESTART("_ZN5movie12VideoDecoder5FlushEv")
DECODER_RESTART("_ZN5movie12AudioDecoder5StartEv")
DECODER_RESTART("_ZN5movie12AudioDecoder5FlushEv")
#undef DECODER_RESTART
MOVIE_NATIVE("_ZN5movie12VideoDecoder11DecodeAsyncEPKN2nn5movie6BufferE") { Get(c->x[0]).input_received=true;Ret(c,0); }
MOVIE_NATIVE("_ZN5movie12AudioDecoder11DecodeAsyncEPKN2nn5movie6BufferE") { Get(c->x[0]).input_received=true;Ret(c,0); }
MOVIE_NATIVE("_ZN5movie12VideoDecoder8FinalizeEv") {
    std::lock_guard lock{decoders_mutex};
    decoders.erase(c->x[0]);
    Ret(c, 0);
}
MOVIE_NATIVE("_ZN5movie12AudioDecoder8FinalizeEv") {
    std::lock_guard lock{decoders_mutex};
    decoders.erase(c->x[0]);
    Ret(c, 0);
}
MOVIE_NATIVE("_ZN5movie12VideoDecoder18GetInputReadyEventEPPN2nn2os15SystemEventTypeE") { InputEvent(c); }
MOVIE_NATIVE("_ZN5movie12VideoDecoder19GetOutputReadyEventEPPN2nn2os15SystemEventTypeE") { OutputEvent(c); }
MOVIE_NATIVE("_ZN5movie12AudioDecoder18GetInputReadyEventEPPN2nn2os15SystemEventTypeE") { InputEvent(c); }
MOVIE_NATIVE("_ZN5movie12AudioDecoder19GetOutputReadyEventEPPN2nn2os15SystemEventTypeE") { OutputEvent(c); }
MOVIE_NATIVE("_ZN5movie12VideoDecoder18RegisterVideoFrameEPPNS_10VideoFrameEi") { Register(c); }
MOVIE_NATIVE("_ZN5movie12AudioDecoder18RegisterAudioFrameEPPNS_10AudioFrameEi") { Register(c); }
MOVIE_NATIVE("_ZN5movie12VideoDecoder17AcquireVideoFrameEPPNS_10VideoFrameE") {
    const u64 object = c->x[0], out = c->x[1];
    *reinterpret_cast<u64*>(out) = EndOfStreamFrame(c, object, "_ZN5movie10VideoFrame19SetEndOfStreamStateEb");
    Ret(c, 0);
}
MOVIE_NATIVE("_ZN5movie12VideoDecoder17AcquireVideoFrameEPPNS_10VideoFrameEPi") {
    const u64 object = c->x[0], out = c->x[1], index = c->x[2];
    *reinterpret_cast<u64*>(out) = EndOfStreamFrame(c, object, "_ZN5movie10VideoFrame19SetEndOfStreamStateEb");
    if (index) *reinterpret_cast<s32*>(index) = 0;
    Ret(c, 0);
}
MOVIE_NATIVE("_ZN5movie12AudioDecoder17AcquireAudioFrameEPPNS_10AudioFrameEPi") {
    const u64 object = c->x[0], out = c->x[1], index = c->x[2];
    *reinterpret_cast<u64*>(out) = EndOfStreamFrame(c, object, "_ZN5movie10AudioFrame19SetEndOfStreamStateEb");
    if (index) *reinterpret_cast<s32*>(index) = 0;
    Ret(c, 0);
}
