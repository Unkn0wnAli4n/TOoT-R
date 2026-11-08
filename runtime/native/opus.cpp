#include <opus/opus.h>
#include "../native.h"
using namespace rt;
namespace {
struct Decoder { OpusDecoder* decoder; int rate, channels; };
Decoder*& HostDecoder(u64 object) { return *reinterpret_cast<Decoder**>(object); }
constexpr u32 Invalid = 0x7a73; // codec failure; consumers check the result
}
NATIVE("_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii") { Ret(c, opus_decoder_get_size((int)c->x[1]) + 0x1000); }
NATIVE("_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi") {
    int error; auto* decoder = opus_decoder_create((int)c->x[1], (int)c->x[2], &error);
    if (!decoder) { Ret(c, Invalid); return; }
    HostDecoder(c->x[0]) = new Decoder{decoder,(int)c->x[1],(int)c->x[2]}; Ret(c, 0);
}
NATIVE("_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE") { auto*& d=HostDecoder(c->x[0]); if(d){opus_decoder_destroy(d->decoder);delete d;d=nullptr;} }
NATIVE("_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE") { opus_decoder_ctl(HostDecoder(c->x[0])->decoder,OPUS_RESET_STATE); }
NATIVE("_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE") { Ret(c,HostDecoder(c->x[0])->rate); }
NATIVE("_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE") { Ret(c,HostDecoder(c->x[0])->channels); }
NATIVE("_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm") {
    auto* d=HostDecoder(c->x[0]); const auto* packet=Ptr<u8>(c,5);
    *Ptr<u64>(c,1)=0;*Ptr<s32>(c,2)=0;
    if(c->x[6]<8){Ret(c,Invalid);return;}
    // Nintendo packets carry an eight-byte big-endian size/final-range header.
    const u32 size=(u32(packet[0])<<24)|(u32(packet[1])<<16)|(u32(packet[2])<<8)|packet[3];
    if(size>c->x[6]-8){Ret(c,Invalid);return;}
    const int frames=(int)std::min<u64>(c->x[4]/(2*d->channels),d->rate*120/1000);
    const int samples=opus_decode(d->decoder,packet+8,size,Ptr<s16>(c,3),frames,0);
    if(samples<0){Ret(c,Invalid);return;}*Ptr<u64>(c,1)=size+8;*Ptr<s32>(c,2)=samples;Ret(c,0);
}
