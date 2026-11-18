// Hookshot's nn::codec video decoders (github.com/NotRexed/Hookshot, host/nn_video.c, GPL-3.0-or-later) on
// libavcodec (AV1 through dav1d), adapted by video.cpp's compatibility layer.
/* nn::codec's hardware video decoders (VP8/VP9 here) and their image pools, on FFmpeg's libavcodec (the
   system FFmpeg). MKW's movie library (subsdk0, movie::) plays
   its WebM movies through them: the cup openings, the courses' billboards, the emblems.

   How the library uses them (subsdk0, movie::detail::VideoDecoderBuffer): each picture buffer is an image pool of
   one image, sized by MakeDecodeImagePreference + CalculateDecoderImagePoolBuffer, registered with the decoder by
   index (RegisterXDecoderBuffer). It submits each compressed frame (SubmitXDecoderInputBuffer), takes finished
   pictures by buffer index (AcquireXDecoderOutputBuffer, GetXDecoderOutputBufferProperties), reads them back to
   its own memory as RGBA or NV12 (ReadDecoderOutputImageRgba / Nv12) and gives the buffer back (Release).

   Here a submission is decoded at once; a finished picture waits in a queue until an Acquire puts it in the first
   free registered buffer (host memory: the pool's guest memory is not used). The video service itself (nn::gsv) is
   not emulated: nn::codec::video::Initialize does nothing. */
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* (from Longshot ae7b1165) The decoder's input limit. The console's decoder takes only so much compressed data at a time: a
   submission it has no room for fails with RESULT_INPUT_FULL (the game keeps the data and submits again later), and its
   submission-ready event is clear until there is room. Here every submission was taken, so the movie player put a whole
   movie in at once, and the thread that does it is also the one that takes the pictures: while it was busy submitting
   (seconds, in step with the movie's length) it took a picture only every ~105 ms and the catch-up dropped the rest: a
   movie started late and showed ~10 pictures a second at first. Now a submission is refused while INPUT_LIMIT wait (not
   decoded, or decoded and not taken). The stream's last submission is always taken. HOS_VIDEO_INPUT_LIMIT=<n>, 0: none */
#define RESULT_INPUT_FULL 0x1c266f   /* (module 111, description 3603) */
#define INPUT_LIMIT 16
#define RESULT_NO_OUTPUT 0x2a1ca   /* (a failure: "no picture yet"; the library polls again) */
#define POOL_MAGIC 0x4c4f4f50u     /* 'POOL' */
#define DEC_MAGIC 0x43454456u      /* 'VDEC' */
#define PREF_MAGIC 0x46455250u     /* 'PREF' */
#define MAX_SLOTS 16
#define MAX_QUEUE 8

static int video_log(void) {   /* HOS_VIDEO_LOG=1: submissions and acquisitions */
    static int on = -1;
    if (on < 0) on = getenv("HOS_VIDEO_LOG") != NULL;
    return on;
}


/* an image pool: its pictures, w x h, as RGBA and as NV12 (both kept: the reader asks for either) */
typedef struct {
    int w, h, format, count;
    uint8_t *rgba, *nv12;
    uint64_t mem, mem_size;   /* (hookshot) the pool's guest memory: where the hardware decoder writes its pictures */
} Pool;
/* the preference MakeDecodeImagePreference returns (the caller's buffer is larger: only these fields are ours) */
typedef struct { uint32_t magic; uint16_t w, h; uint32_t format, tiling; } Pref;

typedef struct { uint8_t *rgba, *nv12; int w, h; int64_t pts; int last; } Picture;
/* a submission waiting to be decoded (pkt NULL: the stream's end) */
typedef struct Pending { AVPacket *pkt; struct Pending *next; } Pending;


typedef struct {
    AVCodecContext *ctx;
    void *mf;                        /* (hookshot) an AV1 decoder on Media Foundation instead of ctx */
    AVPacket *pkt;
    AVFrame *frame;
    struct SwsContext *sws;
    int sws_w, sws_h, sws_fmt;
    uint64_t slots[MAX_SLOTS];       /* registered pools (guest pointers), by buffer index */
    int busy[MAX_SLOTS];             /* acquired, not released */
    int64_t slot_pts[MAX_SLOTS];
    int slot_last[MAX_SLOTS];        /* the stream's last picture (the submission marked last, all decoded) */
    Picture queue[MAX_QUEUE];
    int qhead, qn;
    int eos;                         /* the last submission is in and decoded: an empty queue is the end */
    Pending *phead, *ptail;          /* submissions not decoded yet, oldest first */
    int64_t submits, acquires;
    /* (tootr) pictures decoded ahead of the game's asking, on a thread of the decoder's own: see ahead_main */
    pthread_t ahead; int ahead_on, ahead_stop, decoding; pthread_cond_t idle;
    Picture held[8]; int held_n;     /* (-1: frames go straight into the queue; else ahead_main collects them here) */
    int npending;                    /* submissions not decoded yet (the list's length): the input limit's count */
    int64_t refused;                 /* submissions refused while the decoder was full */
    uint64_t ready;                  /* its submission-ready event (0: not asked for); ready_clear: clear now, the decoder full */
    int ready_clear;
    int clock_set;                   /* (hookshot) the movie's clock: the picture at clock_pts was due at clock_ns */
    int64_t clock_ns, clock_pts;
    pthread_mutex_t lock;
} Dec;

static int input_limit(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("HOS_VIDEO_INPUT_LIMIT"); n = e && *e ? atoi(e) : INPUT_LIMIT; if (n < 0) n = 0; }
    return n;
}
static int input_full(int pending, int queued, int limit) { return limit > 0 && pending + queued >= limit; }
/* the decoder's submission-ready event follows the rule: clear while it is full, signalled once there is room. Holding d->lock */
static void input_room(Dec *d) {
    if (!d->ready) return;
    const int full = input_full(d->npending, d->qn, input_limit());
    if (full && !d->ready_clear) { d->ready_clear = 1; hos_system_event_clear(d->ready); }
    else if (!full && d->ready_clear) { d->ready_clear = 0; hos_system_event_signal(d->ready); }
}
static void *host_of(uint64_t obj, uint32_t magic) {
    if (!obj || G32(obj + 8) != magic) return NULL;
    return (void *)(uintptr_t)G64(obj);
}
static void set_host(uint64_t obj, uint32_t magic, void *p) {
    G64(obj) = (uint64_t)(uintptr_t)p;
    G32(obj + 8) = p ? magic : 0;
}

/* ---- the service ---- */
HLE(_ZN2nn5codec5video10InitializeEv) {
    static int logged;
    if (!logged++) hos_log("video: nn::codec::video on libavcodec");
    RET(0);
}
HLE(_ZN2nn5codec5video8FinalizeEv) { RET(0); }

/* ---- image pools ---- */
/* MakeDecodeImagePreference(u16 w, u16 h, OutputFormat, TilingMode) -> the preference (returned through x8) */
HLE(_ZN2nn5codec25MakeDecodeImagePreferenceEttNS0_24VideoDecoderOutputFormatENS0_15ImageTilingModeE) {
    const uint64_t out = c->x[8];
    Pref p = {PREF_MAGIC, (uint16_t)ARG(0), (uint16_t)ARG(1), (uint32_t)ARG(2), (uint32_t)ARG(3)};
    if (out) memcpy(GP(out), &p, sizeof p);
    RET(out);
}
/* CalculateDecoderImagePoolBuffer(u64* size, u64* alignment, Preference const&, u32 count) */
HLE(_ZN2nn5codec31CalculateDecoderImagePoolBufferEPmS1_RKNS0_27VideoDecoderImagePreferenceEj) {
    const Pref *p = (const Pref *)GP(ARG(2));
    const uint64_t one = p->magic == PREF_MAGIC ? (uint64_t)p->w * p->h * 4 : 0x400000;
    if (ARG(0)) G64(ARG(0)) = ((one * (uint32_t)ARG(3)) + 0xFFF) & ~0xFFFull;
    if (ARG(1)) G64(ARG(1)) = 0x1000;
    RET(0);
}
/* InitializeDecoderImagePool(pool, void* memory, size, Preference const&, u32 count) */
HLE(_ZN2nn5codec26InitializeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeEPvmRKNS0_27VideoDecoderImagePreferenceEj) {
    const Pref *p = (const Pref *)GP(ARG(3));
    Pool *o = (Pool *)calloc(1, sizeof *o);
    o->w = p->magic == PREF_MAGIC && p->w ? p->w : 1280, o->h = p->magic == PREF_MAGIC && p->h ? p->h : 720;
    o->format = p->magic == PREF_MAGIC ? (int)p->format : 2;
    o->count = (int)ARG(4) > 0 ? (int)ARG(4) : 1;
    o->rgba = (uint8_t *)calloc((size_t)o->count, (size_t)o->w * o->h * 4);
    o->nv12 = (uint8_t *)calloc((size_t)o->count, (size_t)o->w * o->h * 3 / 2);
    /* (from Longshot 67451f4e) video black, not zeros. A buffer the game takes before the decoder's first picture is shown as
       it is (dec_acquire: "nothing decoded, the buffer again", up to ~100 ms into a movie's start), and NV12 zeros are
       bright green (luma 0 with no chroma): one green frame where a movie opens. Luma 16, chroma 128 is black; the RGBA
       opaque black */
    if (o->nv12) {
        for (int i = 0; i < o->count; i++) {
            uint8_t *img = o->nv12 + (size_t)i * o->w * o->h * 3 / 2;
            memset(img, 0x10, (size_t)o->w * o->h);
            memset(img + (size_t)o->w * o->h, 0x80, (size_t)o->w * o->h / 2);
        }
    }
    if (o->rgba)
        for (size_t k = 3; k < (size_t)o->count * o->w * o->h * 4; k += 4) o->rgba[k] = 0xff;
    o->mem = ARG(1), o->mem_size = ARG(2);
    if (video_log())
        hos_log("video: image pool %#llx: %dx%d format %d, %d images, memory %#llx+%#llx", (unsigned long long)ARG(0),
                o->w, o->h, o->format, o->count, (unsigned long long)o->mem, (unsigned long long)o->mem_size);
    set_host(ARG(0), POOL_MAGIC, o);
    RET(0);
}
HLE(_ZN2nn5codec24FinalizeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeE) {
    Pool *o = (Pool *)host_of(ARG(0), POOL_MAGIC);
    if (o) free(o->rgba), free(o->nv12), free(o);
    set_host(ARG(0), POOL_MAGIC, NULL);
}
/* ReadDecoderOutputImageRgba(pool, u32 index, u16 x, u16 y, u16 w, u16 h, void* dst, size_t stride, size_t size) */
HLE(_ZN2nn5codec26ReadDecoderOutputImageRgbaEPKNS0_25VideoDecoderImagePoolTypeEjttttPvmm) {
    const Pool *o = (const Pool *)host_of(ARG(0), POOL_MAGIC);
    const uint32_t i = (uint32_t)ARG(1), x = (uint16_t)ARG(2), y = (uint16_t)ARG(3), w = (uint16_t)ARG(4), h = (uint16_t)ARG(5);
    const uint64_t dst = ARG(6), stride = ARG(7) ? ARG(7) : (uint64_t)w * 4;
    {   /* (hookshot, HOS_VIDEO_LOG) the first reads, with where they go */
        static int n;
        if (video_log() && __atomic_add_fetch(&n, 1, __ATOMIC_RELAXED) <= 4)
            hos_log("video: ReadDecoderOutputImageRgba(pool %s, index %u, %u,%u %ux%u, dst %#llx stride %llu)",
                    o ? "ok" : "unknown", i, x, y, w, h, (unsigned long long)dst, (unsigned long long)stride);
    }
    if (!o || i >= (uint32_t)o->count || !dst || x >= (uint32_t)o->w) { RET(0); return; }
    const uint8_t *src = o->rgba + (size_t)i * o->w * o->h * 4;
    const uint32_t cw = x + w <= (uint32_t)o->w ? w : o->w - x;
    for (uint32_t r = 0; r < h && y + r < (uint32_t)o->h; r++)
        memcpy(GP(dst + r * stride), src + ((size_t)(y + r) * o->w + x) * 4, (size_t)cw * 4);
    HOS_WRITTEN(GP(dst), (size_t)(h ? h - 1 : 0) * stride + (size_t)cw * 4);   /* (the game may give it to the GPU) */
    RET(0);
}
/* ReadDecoderOutputImageNv12(pool, u32 index, u16 x, u16 y, u16 w, u16 h, void* y_dst, size_t y_stride, [stack:]
   size_t y_size, void* uv_dst, size_t uv_stride, size_t uv_size) */
HLE(_ZN2nn5codec26ReadDecoderOutputImageNv12EPKNS0_25VideoDecoderImagePoolTypeEjttttPvmmS4_mm) {
    const Pool *o = (const Pool *)host_of(ARG(0), POOL_MAGIC);
    const uint32_t i = (uint32_t)ARG(1), x = (uint16_t)ARG(2), y = (uint16_t)ARG(3), w = (uint16_t)ARG(4), h = (uint16_t)ARG(5);
    const uint64_t ydst = ARG(6), ystride = ARG(7) ? ARG(7) : w;
    const uint64_t uvdst = G64(c->sp + 8), uvstride = G64(c->sp + 16) ? G64(c->sp + 16) : w;
    {   /* (HOS_VIDEO_LOG) each new destination, with the pool and image */
        static uint64_t seen[32];
        static int nseen;
        int k;
        for (k = 0; k < nseen && seen[k] != ydst; k++) {}
        if (video_log() && k == nseen && nseen < 32) {
            seen[__atomic_add_fetch(&nseen, 1, __ATOMIC_RELAXED) - 1] = ydst;
            hos_log("video: ReadDecoderOutputImageNv12(pool %#llx %s, index %u, %u,%u %ux%u, y %#llx/%llu, uv %#llx/%llu) "
                    "lr %#llx", (unsigned long long)ARG(0), o ? "ok" : "unknown", i, x, y, w, h, (unsigned long long)ydst,
                    (unsigned long long)ystride, (unsigned long long)uvdst, (unsigned long long)uvstride,
                    (unsigned long long)c->x[30]);
        }
    }
    if (!o || i >= (uint32_t)o->count || x >= (uint32_t)o->w) { RET(0); return; }
    const uint8_t *yp = o->nv12 + (size_t)i * o->w * o->h * 3 / 2, *uvp = yp + (size_t)o->w * o->h;
    const uint32_t cw = x + w <= (uint32_t)o->w ? w : o->w - x;
    for (uint32_t r = 0; ydst && r < h && y + r < (uint32_t)o->h; r++)
        memcpy(GP(ydst + r * ystride), yp + (size_t)(y + r) * o->w + x, cw);
    for (uint32_t r = 0; uvdst && r < h / 2 && y / 2 + r < (uint32_t)o->h / 2; r++)
        memcpy(GP(uvdst + r * uvstride), uvp + (size_t)(y / 2 + r) * o->w + (x & ~1u), cw);
    if (ydst) HOS_WRITTEN(GP(ydst), (size_t)(h ? h - 1 : 0) * ystride + cw);
    if (uvdst) HOS_WRITTEN(GP(uvdst), (size_t)(h / 2 ? h / 2 - 1 : 0) * uvstride + cw);
    RET(0);
}

/* ---- decoders ---- */
/* (tootr) Decoding ahead. The pictures were decoded when the game asked for one, on the game's thread: a step that
   took long - 60 to 110 ms now and then on a six-core laptop, where the average was 5 - put the movie behind its
   clock and pictures were skipped to catch up, 8 to 50 in every 100: a 60 fps movie that looked like 30 and
   stuttered. A thread of the decoder's own keeps a few pictures decoded ahead instead; the game takes the oldest
   and never waits for the decoder. The decoder itself is used outside the lock (`decoding` says so: a flush waits
   for the step to end), the queue only under it. libavcodec's decoders only; HOS_VIDEO_AHEAD=0: as before. */
#define AHEAD_PICTURES 5
static void ahead_start(Dec *d);
static void ahead_end(Dec *d);
static void dec_init_sized(Ctx *c, enum AVCodecID id, const char *what, int w, int h);
static void dec_init(Ctx *c, enum AVCodecID id, const char *what) { dec_init_sized(c, id, what, 0, 0); }
static void dec_init_sized(Ctx *c, enum AVCodecID id, const char *what, int w, int h) {
    const AVCodec *codec = NULL;
#ifndef _WIN32   /* (linux) FFmpeg's own AV1 decoder needs a hardware context, and there is no Media Foundation to fall
                    back on: dav1d, else libaom, by name */
    if (id == AV_CODEC_ID_AV1) codec = avcodec_find_decoder_by_name("libdav1d");
    if (!codec && id == AV_CODEC_ID_AV1) codec = avcodec_find_decoder_by_name("libaom-av1");
#endif
    if (!codec) codec = avcodec_find_decoder(id);
    Dec *d = (Dec *)calloc(1, sizeof *d);
    pthread_mutex_init(&d->lock, nullptr);
    pthread_cond_init(&d->idle, nullptr);
    d->held_n = -1;
    if (codec) {
        d->ctx = avcodec_alloc_context3(codec);
        d->ctx->thread_count = id == AV_CODEC_ID_AV1 ? 4 : 2;   /* (1080p AV1 in software) */
        if (avcodec_open2(d->ctx, codec, NULL) < 0) avcodec_free_context(&d->ctx);
    }
    if (!d->ctx && id == AV_CODEC_ID_AV1) d->mf = mf_av1_open(w, h);
    d->pkt = av_packet_alloc();
    d->frame = av_frame_alloc();
    ahead_start(d);
    set_host(ARG(0), DEC_MAGIC, d);
    static int logged;
    if (logged++ < 4)
        hos_log("video: %s decoder %s (%dx%d)", what, d->ctx ? "ready" : d->mf ? "ready on Media Foundation" :
                "not available in this FFmpeg", w, h);
    RET(0);
}
static void picture_free(Picture *p) {
    free(p->rgba), free(p->nv12);
    memset(p, 0, sizeof *p);
}
static void pending_free(Dec *d) {
    while (d->phead) {
        Pending *n = d->phead;
        d->phead = n->next;
        av_packet_free(&n->pkt);
        free(n);
    }
    d->ptail = NULL;
    d->npending = 0;
}
static void ready_event_give_back(uint64_t event);
static void dec_finalize(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(0), DEC_MAGIC);
    if (!d) return;
    ahead_end(d);
    ready_event_give_back(d->ready);
    pending_free(d);
    for (int i = 0; i < d->qn; i++) picture_free(&d->queue[(d->qhead + i) % MAX_QUEUE]);
    if (d->sws) sws_freeContext(d->sws);
    avcodec_free_context(&d->ctx);
    mf_av1_close(d->mf);
    av_packet_free(&d->pkt);
    av_frame_free(&d->frame);
    free(d);
    set_host(ARG(0), DEC_MAGIC, NULL);
}
static void queue_push(Dec *d, Picture p) {
    if (d->qn == MAX_QUEUE) {
        picture_free(&d->queue[d->qhead]);
        d->qhead = (d->qhead + 1) % MAX_QUEUE, d->qn--;
    }
    d->queue[(d->qhead + d->qn) % MAX_QUEUE] = p;
    d->qn++;
}
/* a decoded frame into the queue, as RGBA and NV12 (the oldest dropped when the reader falls behind) */
static void queue_frame(Dec *d, const AVFrame *f) {
    const int w = f->width, h = f->height;
    if (w <= 0 || h <= 0) return;
    if (!d->sws || d->sws_w != w || d->sws_h != h || d->sws_fmt != f->format) {
        if (d->sws) sws_freeContext(d->sws);
        d->sws = sws_getContext(w, h, (enum AVPixelFormat)f->format, w, h, AV_PIX_FMT_RGBA, SWS_BILINEAR, NULL, NULL, NULL);
        d->sws_w = w, d->sws_h = h, d->sws_fmt = f->format;
    }
    Picture p = {0};
    p.w = w, p.h = h, p.pts = f->pts;
    p.rgba = (uint8_t *)malloc((size_t)w * h * 4);
    p.nv12 = (uint8_t *)calloc(1, (size_t)w * h * 3 / 2);
    uint8_t *dst[4] = {p.rgba, NULL, NULL, NULL};
    int dst_stride[4] = {w * 4, 0, 0, 0};
    if (d->sws) sws_scale(d->sws, (const uint8_t *const *)f->data, f->linesize, 0, h, dst, dst_stride);
    if (f->format == AV_PIX_FMT_NV12) {   /* (hookshot: Media Foundation's pictures) */
        for (int y = 0; y < h; y++) memcpy(p.nv12 + (size_t)y * w, f->data[0] + (size_t)y * f->linesize[0], (size_t)w);
        for (int y = 0; y < h / 2; y++)
            memcpy(p.nv12 + (size_t)w * h + (size_t)y * w, f->data[1] + (size_t)y * f->linesize[1], (size_t)w);
    } else if (f->format == AV_PIX_FMT_YUV420P || f->format == AV_PIX_FMT_YUVA420P) {   /* (planar 4:2:0 -> NV12) */
        for (int y = 0; y < h; y++) memcpy(p.nv12 + (size_t)y * w, f->data[0] + (size_t)y * f->linesize[0], (size_t)w);
        uint8_t *uv = p.nv12 + (size_t)w * h;
        for (int y = 0; y < h / 2; y++)
            for (int x = 0; x < w / 2; x++) {
                uv[(size_t)y * w + 2 * x] = f->data[1][(size_t)y * f->linesize[1] + x];
                uv[(size_t)y * w + 2 * x + 1] = f->data[2][(size_t)y * f->linesize[2] + x];
            }
    }
    if (d->held_n >= 0) { if (d->held_n < 8) d->held[d->held_n++] = p; else picture_free(&p); return; }   /* (tootr: ahead_main's, pushed under the lock) */
    queue_push(d, p);
}
static void mf_picture(void *user, const uint8_t *y, const uint8_t *uv, int w, int h, int stride, int64_t pts) {
    AVFrame f;
    memset(&f, 0, sizeof f);
    f.width = w, f.height = h, f.format = AV_PIX_FMT_NV12, f.pts = pts;
    f.data[0] = (uint8_t *)y, f.data[1] = (uint8_t *)uv;
    f.linesize[0] = f.linesize[1] = stride;
    queue_frame((Dec *)user, &f);
}
static void drain(Dec *d) {
    while (avcodec_receive_frame(d->ctx, d->frame) == 0) {
        queue_frame(d, d->frame);
        av_frame_unref(d->frame);
    }
}
/* the oldest submissions decoded until a picture comes out (or none are left) */
static void decode_next(Dec *d);
/* (tootr) What a movie costs, always in the log (one line every 600 pictures asked for): how many pictures were shown,
   how often there was none ready and the last one stayed, how many were skipped to keep up with the sound, and the
   time one submission takes to decode. A movie that looks slower than the frame rate shows it here. */
static int64_t vstat_asked, vstat_repeated, vstat_skipped, vstat_decode_ns, vstat_decodes, vstat_worst_ns;
static void decode_timed(Dec *d) {
    const int64_t t0 = hos_now_ns();
    decode_next(d);
    const int64_t spent = hos_now_ns() - t0;
    vstat_decode_ns += spent; vstat_decodes++;
    if (spent > vstat_worst_ns) vstat_worst_ns = spent;
}
static void video_stats(void) {
    if (++vstat_asked < 600) return;
    hos_log("video: of 600 pictures asked for, %lld times none was ready (the last one stayed), %lld skipped to keep up; decoding took %.1f ms a step on average, %.1f at worst (%lld steps)",
            (long long)vstat_repeated, (long long)vstat_skipped, vstat_decodes ? vstat_decode_ns / 1e6 / (double)vstat_decodes : 0.0, vstat_worst_ns / 1e6, (long long)vstat_decodes);
    vstat_asked = vstat_repeated = vstat_skipped = vstat_decode_ns = vstat_decodes = vstat_worst_ns = 0;
}
static void decode_pending(Dec *d) {
    if (d->ahead_on) return;   /* (tootr: ahead_main decodes) */
    while (!d->qn && d->phead) decode_timed(d);
}
/* one submission decoded (its pictures into the queue, if any) */
static void decode_next(Dec *d) {
    movie_decoding();   /* (tootr) */
    if (d->phead) {
        Pending *n = d->phead;
        d->phead = n->next;
        if (!d->phead) d->ptail = NULL;
        if (d->npending > 0) d->npending--;
        if (d->mf) {   /* (hookshot) */
            if (n->pkt) {
                d->eos = 0;
                mf_av1_decode(d->mf, n->pkt->data, (size_t)n->pkt->size, n->pkt->pts, mf_picture, d);
                av_packet_free(&n->pkt);
            } else {
                mf_av1_decode(d->mf, NULL, 0, 0, mf_picture, d);
                d->eos = 1;
                if (d->qn) d->queue[(d->qhead + d->qn - 1) % MAX_QUEUE].last = 1;
            }
        } else if (n->pkt) {
            d->eos = 0;
            if (avcodec_send_packet(d->ctx, n->pkt) == AVERROR(EAGAIN)) {   /* (full: its pictures out first) */
                drain(d);
                avcodec_send_packet(d->ctx, n->pkt);
            }
            drain(d);
            av_packet_free(&n->pkt);
        } else {   /* (the end: the rest out, the last picture marked) */
            avcodec_send_packet(d->ctx, NULL);
            drain(d);
            d->eos = 1;
            if (d->qn) d->queue[(d->qhead + d->qn - 1) % MAX_QUEUE].last = 1;
            avcodec_flush_buffers(d->ctx);   /* (a stream after it starts again) */
        }
        free(n);
    }
}
static void *ahead_main(void *arg) {
    Dec *d = (Dec *)arg;
    pthread_mutex_lock(&d->lock);
    while (!d->ahead_stop) {
        if (!d->phead || d->qn >= AHEAD_PICTURES) {
            pthread_mutex_unlock(&d->lock);
            hos_sleep_ns(1000000);
            pthread_mutex_lock(&d->lock);
            continue;
        }
        Pending *n = d->phead;   /* (the oldest submission: taken out under the lock, decoded outside it) */
        d->phead = n->next;
        if (!d->phead) d->ptail = NULL;
        if (d->npending > 0) d->npending--;
        if (n->pkt) d->eos = 0;
        d->decoding = 1;
        d->held_n = 0;
        pthread_mutex_unlock(&d->lock);
        movie_decoding();
        const int64_t t0 = hos_now_ns();
        const int end = !n->pkt;
        if (n->pkt) {
            if (avcodec_send_packet(d->ctx, n->pkt) == AVERROR(EAGAIN)) {   /* (full: its pictures out first) */
                drain(d);
                avcodec_send_packet(d->ctx, n->pkt);
            }
            drain(d);
            av_packet_free(&n->pkt);
        } else {   /* (the end: the rest out) */
            avcodec_send_packet(d->ctx, NULL);
            drain(d);
            avcodec_flush_buffers(d->ctx);   /* (a stream after it starts again) */
        }
        free(n);
        const int64_t spent = hos_now_ns() - t0;
        pthread_mutex_lock(&d->lock);
        for (int i = 0; i < d->held_n; i++) queue_push(d, d->held[i]);
        d->held_n = -1;
        if (end) {
            d->eos = 1;
            if (d->qn) d->queue[(d->qhead + d->qn - 1) % MAX_QUEUE].last = 1;
        }
        vstat_decode_ns += spent; vstat_decodes++;
        if (spent > vstat_worst_ns) vstat_worst_ns = spent;
        d->decoding = 0;
        pthread_cond_broadcast(&d->idle);
        input_room(d);
    }
    pthread_mutex_unlock(&d->lock);
    return NULL;
}
static void ahead_start(Dec *d) {
    static int off = -1;
    if (off < 0) { const char *e = getenv("HOS_VIDEO_AHEAD"); off = e && e[0] == '0'; }
    if (off || !d->ctx) return;
    d->ahead_stop = 0;
    if (pthread_create(&d->ahead, NULL, ahead_main, d) == 0) d->ahead_on = 1;
}
static void ahead_end(Dec *d) {
    if (!d->ahead_on) return;
    pthread_mutex_lock(&d->lock);
    d->ahead_stop = 1;
    pthread_mutex_unlock(&d->lock);
    pthread_join(d->ahead, NULL);
    d->ahead_on = 0;
}
static void pending_push(Dec *d, AVPacket *pkt) {
    Pending *n = (Pending *)calloc(1, sizeof *n);
    n->pkt = pkt;
    if (d->ptail) d->ptail->next = n;
    else d->phead = n;
    d->ptail = n;
    d->npending++;
}
/* SubmitXDecoderInputBuffer(u64* id, decoder, data, size, TimeSpan pts, bool last): kept, decoded when a picture is
   acquired (the library submits a whole movie at once: its pictures decoded then would not fit any queue) */
static long long submissions;
static void dec_submit(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(1), DEC_MAGIC);
    if (ARG(0)) G64(ARG(0)) = (uint64_t)__atomic_add_fetch(&submissions, 1, __ATOMIC_RELAXED);
    if (!d || (!d->ctx && !d->mf)) { RET(0); return; }
    pthread_mutex_lock(&d->lock);
    /* (no room now: counted, the ready event cleared, nothing taken; never the stream's last submission. It returns at once,
       so the game's thread goes on to its Acquire, which makes the room) */
    if (!(ARG(5) & 1) && input_full(d->npending, d->qn, input_limit())) {
        d->refused++;
        input_room(d);
        pthread_mutex_unlock(&d->lock);
        RET(RESULT_INPUT_FULL);
        return;
    }
    if (ARG(2) && ARG(3)) {
        AVPacket *pkt = av_packet_alloc();
        if (pkt && av_new_packet(pkt, (int)ARG(3)) == 0) {
            memcpy(pkt->data, GP(ARG(2)), (size_t)ARG(3));
            pkt->pts = (int64_t)ARG(4);
            pending_push(d, pkt);
        } else {
            av_packet_free(&pkt);
        }
    }
    d->submits++;
    if (ARG(5) & 1) pending_push(d, NULL);
    if (video_log() && (d->submits <= 8 || (ARG(5) & 1)))
        hos_log("video: submit %lld (%llu bytes, pts %lld%s)", (long long)d->submits, (unsigned long long)ARG(3),
                (long long)ARG(4), (ARG(5) & 1) ? ", last" : "");
    input_room(d);
    pthread_mutex_unlock(&d->lock);
    RET(0);
}
static void dec_flush(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(0), DEC_MAGIC);
    if (d && (d->ctx || d->mf)) {
        pthread_mutex_lock(&d->lock);
        while (d->decoding) pthread_cond_wait(&d->idle, &d->lock);   /* (tootr: a step of ahead_main's ends first) */
        if (d->ctx) avcodec_flush_buffers(d->ctx);
        else mf_av1_flush(d->mf);
        pending_free(d);
        for (int i = 0; i < d->qn; i++) picture_free(&d->queue[(d->qhead + i) % MAX_QUEUE]);
        d->qhead = d->qn = 0;
        d->eos = 0;
        d->clock_set = 0;
        for (int i = 0; i < MAX_SLOTS; i++) d->slot_last[i] = 0;
        if (d->refused && video_log()) hos_log("video: %lld submissions were put off while %d waited", (long long)d->refused, input_limit());
        d->refused = 0;
        d->submits = 0;  /* (tootr: the next stream starts as the first did) */
        input_room(d);   /* (emptied: there is room again) */
        pthread_mutex_unlock(&d->lock);
    }
    RET(0);
}
/* RegisterXDecoderBuffer(decoder, u32 index, pool, u32 image) / UnregisterXDecoderBuffer(decoder, u32 index) */
static void dec_register(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(0), DEC_MAGIC);
    if (d && (uint32_t)ARG(1) < MAX_SLOTS) d->slots[(uint32_t)ARG(1)] = ARG(2), d->busy[(uint32_t)ARG(1)] = 0;
    RET(0);
}
static void dec_unregister(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(0), DEC_MAGIC);
    if (d && (uint32_t)ARG(1) < MAX_SLOTS) d->slots[(uint32_t)ARG(1)] = 0;
    RET(0);
}
/* AcquireXDecoderOutputBuffer(u32* index, NativeSyncHandle*, decoder): the oldest picture into a free buffer */
static void dec_acquire(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(2), DEC_MAGIC);
    if (ARG(1)) G64(ARG(1)) = 0;
    if (!d) { RET(RESULT_NO_OUTPUT); return; }
    /* (the library aborts on any failure here: the hardware call waits for a picture. Up to ~100 ms for one decoded
       by another thread's submission; then a free buffer as it was, so the caller is never refused) */
    pthread_mutex_lock(&d->lock);
    decode_pending(d);
    /* (tootr) not before the stream's first submission: the game takes its first buffers before it submits anything, and
       each of those waited the whole ~100 ms for a picture that could not come: a movie started ~0.3 s late */
    /* (100 ms by the clock: 50 sleeps of 2 ms take 250 ms where a sleep is 4-5 ms - Longshot cc04531b) */
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int tries = 0; !d->qn && !d->eos && d->submits > 0 && tries < 50; tries++) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        if ((now.tv_sec - t0.tv_sec) * 1000000000LL + (now.tv_nsec - t0.tv_nsec) > 100000000LL) break;
        pthread_mutex_unlock(&d->lock);
        hos_sleep_ns(2000000);
        pthread_mutex_lock(&d->lock);
        decode_pending(d);
    }
    int slot = -1;
    for (int i = 0; i < MAX_SLOTS && slot < 0; i++)
        if (d->slots[i] && !d->busy[i] && host_of(d->slots[i], POOL_MAGIC)) slot = i;
    if (slot < 0) {
        pthread_mutex_unlock(&d->lock);
        RET(RESULT_NO_OUTPUT);
        return;
    }
    d->acquires++;
    video_stats();
    if (!d->qn) {
        if (!d->eos && (d->phead || d->decoding)) vstat_repeated++;   /* (a movie is running and its next picture is late) */   /* (nothing decoded: the buffer's last picture again; after the last submission, as the last) */
        d->busy[slot] = 1;
        if (d->eos) d->slot_last[slot] = 1;
        if (video_log()) hos_log("video: acquire %lld: nothing decoded%s, buffer %d again", (long long)d->acquires,
                                 d->eos ? " (the end)" : "", slot);
        else if (d->acquires == 1 && !d->eos) hos_log("video: the first buffer went out before the first picture was decoded (shown black)");
        input_room(d);   /* (the pending ones were decoded on) */
        pthread_mutex_unlock(&d->lock);
        if (ARG(0)) G32(ARG(0)) = (uint32_t)slot;
        RET(0);
        return;
    }
    {   /* (hookshot) The game takes one picture a frame, and the sound plays on: under 60 fps the movie fell behind its
           sound (the intro "slow motion, audio ahead", issues 146 and 179). Pictures behind the movie's clock are skipped:
           the next ones decoded, as the whole movie is submitted at once. HOS_VIDEO_NO_CATCHUP=1: as before */
        static int off = -1;
        if (off < 0) off = getenv("HOS_VIDEO_NO_CATCHUP") != NULL;
        const int64_t now = hos_now_ns(), late = 25000000;   /* (a frame and a half at 60 fps) */
        Picture *h = &d->queue[d->qhead];
        if (!d->clock_set || h->pts < d->clock_pts) d->clock_set = 1, d->clock_ns = now, d->clock_pts = h->pts;   /* (a start, a loop) */
        int skipped = 0;
        for (int guard = 0; !off && guard < 600 && !d->queue[d->qhead].last; guard++) {
            const int64_t due = d->clock_pts + (now - d->clock_ns);
            if (d->queue[d->qhead].pts + late >= due) break;
            if (d->qn < 2) {
                if (!d->phead || d->ahead_on) break;   /* (tootr: with ahead_main the next ones come by themselves) */
                decode_timed(d);
                continue;
            }
            picture_free(&d->queue[d->qhead]);
            d->qhead = (d->qhead + 1) % MAX_QUEUE, d->qn--;
            skipped++;
        }
        vstat_skipped += skipped;
        if (skipped && video_log()) hos_log("video: %d pictures skipped to catch up with the movie's clock", skipped);
    }
    Picture *p = &d->queue[d->qhead];
    Pool *o = (Pool *)host_of(d->slots[slot], POOL_MAGIC);
    /* (the picture at the pool's size: rows and columns past either are left out) */
    const int w = p->w < o->w ? p->w : o->w, h = p->h < o->h ? p->h : o->h;
    for (int y = 0; y < h; y++) {
        memcpy(o->rgba + (size_t)y * o->w * 4, p->rgba + (size_t)y * p->w * 4, (size_t)w * 4);
        memcpy(o->nv12 + (size_t)y * o->w, p->nv12 + (size_t)y * p->w, (size_t)w);
    }
    for (int y = 0; y < h / 2; y++)
        memcpy(o->nv12 + (size_t)o->w * o->h + (size_t)y * o->w, p->nv12 + (size_t)p->w * p->h + (size_t)y * p->w,
               (size_t)w);
    /* (hookshot) and into the pool's guest memory, as the hardware decoder leaves it: OoT copies it from there
       straight into its luma and chroma textures (NV12, pitch = width, chroma right after the luma rows; RGBA when the
       memory has room for it and the pool's format is not NV12's) */
    if (o->mem && o->mem_size) {
        const size_t nv12_size = (size_t)o->w * o->h * 3 / 2, rgba_size = (size_t)o->w * o->h * 4;
        const int rgba = o->format != 0 && o->mem_size >= rgba_size;   /* (VideoDecoderOutputFormat 0: NV12) */
        const size_t need = rgba ? rgba_size : nv12_size;
        if (o->mem_size >= need) {
            memcpy(GP(o->mem), rgba ? o->rgba : o->nv12, need);
            HOS_WRITTEN(GP(o->mem), need);
        }
    }
    d->slot_pts[slot] = p->pts;
    d->slot_last[slot] = p->last;
    d->busy[slot] = 1;
    {   /* (hookshot) HOS_VIDEO_LOG: the picture's time against the time since the first, to see the picture fall behind the sound */
        static int64_t first_ns, first_pts;
        if (d->acquires == 1) first_ns = hos_now_ns(), first_pts = p->pts;
        if (video_log() && d->acquires % 60 == 0)
            hos_log("video: acquire %lld: picture at %.2f s, wall clock %.2f s (behind by %.2f s), %d decoded ahead", (long long)d->acquires,
                    (double)(p->pts - first_pts) / 1e9, (double)(hos_now_ns() - first_ns) / 1e9,
                    (double)(hos_now_ns() - first_ns - (p->pts - first_pts)) / 1e9, d->qn);
    }
    if (video_log() && (d->acquires <= 8 || p->last || d->acquires % 300 == 0)) {
        uint64_t sum = 0, n = 0;   /* (the picture's mean luma and RGB, sampled: a black or garbage decode shows) */
        uint64_t rgb[3] = {0, 0, 0};
        for (int y = 0; y < p->h; y += 37)
            for (int x = 0; x < p->w; x += 41, n++) {
                sum += p->nv12[(size_t)y * p->w + x];
                for (int k = 0; k < 3; k++) rgb[k] += p->rgba[((size_t)y * p->w + x) * 4 + k];
            }
        hos_log("video: acquire %lld: %dx%d pts %lld%s into buffer %d (mean Y %llu, RGB %llu %llu %llu)",
                (long long)d->acquires, p->w, p->h, (long long)p->pts, p->last ? " (last)" : "", slot,
                (unsigned long long)(n ? sum / n : 0), (unsigned long long)(n ? rgb[0] / n : 0),
                (unsigned long long)(n ? rgb[1] / n : 0), (unsigned long long)(n ? rgb[2] / n : 0));
    }
    picture_free(p);
    d->qhead = (d->qhead + 1) % MAX_QUEUE, d->qn--;
    input_room(d);   /* (a picture taken, the skipped ones gone: the game may submit again) */
    pthread_mutex_unlock(&d->lock);
    if (ARG(0)) G32(ARG(0)) = (uint32_t)slot;
    RET(0);
}
/* ReleaseXDecoderOutputBuffer(decoder, u32 index, NativeSyncHandle) */
static void dec_release(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(0), DEC_MAGIC);
    if (d && (uint32_t)ARG(1) < MAX_SLOTS) d->busy[(uint32_t)ARG(1)] = 0;
    RET(0);
}
/* GetXDecoderOutputBufferProperties(u16* w, u16* h, u16* display w, u16* display h, TimeSpan* pts, bool* last,
   decoder, u32 index) */
static void dec_properties(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(6), DEC_MAGIC);
    const uint32_t i = (uint32_t)ARG(7);
    const Pool *o = d && i < MAX_SLOTS ? (const Pool *)host_of(d->slots[i], POOL_MAGIC) : NULL;
    const uint16_t w = o ? (uint16_t)o->w : 0, h = o ? (uint16_t)o->h : 0;
    if (ARG(0)) *(uint16_t *)GP(ARG(0)) = w;
    if (ARG(1)) *(uint16_t *)GP(ARG(1)) = h;
    if (ARG(2)) *(uint16_t *)GP(ARG(2)) = w;
    if (ARG(3)) *(uint16_t *)GP(ARG(3)) = h;
    if (ARG(4)) G64(ARG(4)) = d && i < MAX_SLOTS ? (uint64_t)d->slot_pts[i] : 0;
    if (ARG(5)) *(uint8_t *)GP(ARG(5)) = d && i < MAX_SLOTS ? (uint8_t)d->slot_last[i] : 0;
    RET(0);
}
/* GetXDecoderSubmissionReadyEvent(decoder): the decoder's own event, signalled while it has room (input_room). It was one
   event for all, signalled for ever. The events are never freed, so a decoder that ends gives its event back, signalled,
   for the next one: a game still holding it sees "ready", as before */
static pthread_mutex_t ready_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t ready_spare[8];
static int nready_spare;
static void ready_event_give_back(uint64_t event) {
    if (!event) return;
    hos_system_event_signal(event);
    pthread_mutex_lock(&ready_lock);
    if (nready_spare < (int)(sizeof ready_spare / sizeof ready_spare[0])) ready_spare[nready_spare++] = event;
    pthread_mutex_unlock(&ready_lock);
}
static uint64_t ready_event_new(Ctx *c) {
    pthread_mutex_lock(&ready_lock);
    uint64_t event = nready_spare ? ready_spare[--nready_spare] : 0;
    pthread_mutex_unlock(&ready_lock);
    if (!event) {
        void *storage = NULL;
#ifdef _WIN32
        storage = _aligned_malloc(0x40, 16); if (!storage) return 0;   /* (zeroed, never freed) */
#else
        if (posix_memalign(&storage, 16, 0x40)) return 0;   /* (zeroed, never freed) */
#endif
        memset(storage, 0, 0x40);
        event = (uint64_t)(uintptr_t)storage;
        hos_system_event_init(event, 0);
    }
    hos_system_event_signal(event);
    return event;
}
static uint64_t ready_event(Ctx *c) {
    Dec *d = (Dec *)host_of(ARG(0), DEC_MAGIC);
    if (!d) {   /* (a decoder this file does not know: an event that is always ready, as before) */
        static uint64_t event;
        if (!event) event = ready_event_new(c);
        return event;
    }
    pthread_mutex_lock(&d->lock);
    uint64_t event = d->ready;
    pthread_mutex_unlock(&d->lock);
    if (!event) {   /* (made without the decoder's lock: making one calls into the SDK) */
        event = ready_event_new(c);
        pthread_mutex_lock(&d->lock);
        if (!d->ready) { d->ready = event; d->ready_clear = 0; input_room(d); event = 0; }
        const uint64_t have = d->ready;
        pthread_mutex_unlock(&d->lock);
        if (event) ready_event_give_back(event);   /* (another thread was first) */
        event = have;
    }
    return event;
}

/* VP9 */
HLE(_ZN2nn5codec36InitializeVp9DecoderByTransferMemoryEPNS0_14Vp9DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE) { dec_init(c, AV_CODEC_ID_VP9, "VP9"); }
HLE(_ZN2nn5codec34InitializeVp9DecoderByRemoteMemoryEPNS0_14Vp9DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE) { dec_init(c, AV_CODEC_ID_VP9, "VP9"); }
HLE(_ZN2nn5codec18FinalizeVp9DecoderEPNS0_14Vp9DecoderTypeE) { dec_finalize(c); }
HLE(_ZN2nn5codec24RegisterVp9DecoderBufferEPNS0_14Vp9DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj) { dec_register(c); }
HLE(_ZN2nn5codec26UnregisterVp9DecoderBufferEPNS0_14Vp9DecoderTypeEj) { dec_unregister(c); }
HLE(_ZN2nn5codec15StartVp9DecoderEPNS0_14Vp9DecoderTypeE) { RET(0); }
HLE(_ZN2nn5codec14StopVp9DecoderEPNS0_14Vp9DecoderTypeE) { dec_flush(c); }
HLE(_ZN2nn5codec15FlushVp9DecoderEPNS0_14Vp9DecoderTypeE) { dec_flush(c); }
HLE(_ZN2nn5codec33GetVp9DecoderSubmissionReadyEventEPNS0_14Vp9DecoderTypeE) { RET(ready_event(c)); }
HLE(_ZN2nn5codec27SubmitVp9DecoderInputBufferEPmPNS0_14Vp9DecoderTypeEPKvmNS_8TimeSpanEb) { dec_submit(c); }
HLE(_ZN2nn5codec36SubmitEncryptedVp9DecoderInputBufferEPmPNS0_14Vp9DecoderTypeEPKNS0_18EncryptionPropertyEPKvmNS_8TimeSpanEb) {
    if (ARG(0)) G64(ARG(0)) = (uint64_t)__atomic_add_fetch(&submissions, 1, __ATOMIC_RELAXED);
    RET(0);
}
HLE(_ZN2nn5codec29AcquireVp9DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Vp9DecoderTypeE) { dec_acquire(c); }
HLE(_ZN2nn5codec29ReleaseVp9DecoderOutputBufferEPNS0_14Vp9DecoderTypeEjNS_2gs16NativeSyncHandleE) { dec_release(c); }
HLE(_ZN2nn5codec35GetVp9DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_8TimeSpanEPbPNS0_14Vp9DecoderTypeEj) { dec_properties(c); }

/* (hookshot) AV1: OoT's movies. FFmpeg's AV1 decoder if the build has one, else Media Foundation (host/mf_av1.c).
   Initialize...ByTransferMemory(decoder, memory, size, ?, u16 width, u16 height, ...), ...ByRemoteMemory(decoder,
   size, u16 width, u16 height, ...) */
HLE(_ZN2nn5codec36InitializeAv1DecoderByTransferMemoryEPNS0_14Av1DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE) {
    dec_init_sized(c, AV_CODEC_ID_AV1, "AV1", (int)(uint16_t)ARG(4), (int)(uint16_t)ARG(5));
}
HLE(_ZN2nn5codec34InitializeAv1DecoderByRemoteMemoryEPNS0_14Av1DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE) {
    dec_init_sized(c, AV_CODEC_ID_AV1, "AV1", (int)(uint16_t)ARG(2), (int)(uint16_t)ARG(3));
}
HLE(_ZN2nn5codec18FinalizeAv1DecoderEPNS0_14Av1DecoderTypeE) { dec_finalize(c); }
HLE(_ZN2nn5codec24RegisterAv1DecoderBufferEPNS0_14Av1DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj) { dec_register(c); }
HLE(_ZN2nn5codec15StartAv1DecoderEPNS0_14Av1DecoderTypeE) { RET(0); }
HLE(_ZN2nn5codec14StopAv1DecoderEPNS0_14Av1DecoderTypeE) { dec_flush(c); }
HLE(_ZN2nn5codec15FlushAv1DecoderEPNS0_14Av1DecoderTypeE) { dec_flush(c); }
HLE(_ZN2nn5codec33GetAv1DecoderSubmissionReadyEventEPNS0_14Av1DecoderTypeE) { RET(ready_event(c)); }
HLE(_ZN2nn5codec27SubmitAv1DecoderInputBufferEPmPNS0_14Av1DecoderTypeEPKvmNS_8TimeSpanEb) { dec_submit(c); }
HLE(_ZN2nn5codec29AcquireAv1DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Av1DecoderTypeE) { dec_acquire(c); }
HLE(_ZN2nn5codec29ReleaseAv1DecoderOutputBufferEPNS0_14Av1DecoderTypeEjNS_2gs16NativeSyncHandleE) { dec_release(c); }
HLE(_ZN2nn5codec35GetAv1DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_8TimeSpanEPbPNS0_14Av1DecoderTypeEj) { dec_properties(c); }

/* VP8 */
HLE(_ZN2nn5codec36InitializeVp8DecoderByTransferMemoryEPNS0_14Vp8DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE) { dec_init(c, AV_CODEC_ID_VP8, "VP8"); }
HLE(_ZN2nn5codec34InitializeVp8DecoderByRemoteMemoryEPNS0_14Vp8DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE) { dec_init(c, AV_CODEC_ID_VP8, "VP8"); }
HLE(_ZN2nn5codec18FinalizeVp8DecoderEPNS0_14Vp8DecoderTypeE) { dec_finalize(c); }
HLE(_ZN2nn5codec24RegisterVp8DecoderBufferEPNS0_14Vp8DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj) { dec_register(c); }
HLE(_ZN2nn5codec26UnregisterVp8DecoderBufferEPNS0_14Vp8DecoderTypeEj) { dec_unregister(c); }
HLE(_ZN2nn5codec15StartVp8DecoderEPNS0_14Vp8DecoderTypeE) { RET(0); }
HLE(_ZN2nn5codec14StopVp8DecoderEPNS0_14Vp8DecoderTypeE) { dec_flush(c); }
HLE(_ZN2nn5codec15FlushVp8DecoderEPNS0_14Vp8DecoderTypeE) { dec_flush(c); }
HLE(_ZN2nn5codec33GetVp8DecoderSubmissionReadyEventEPNS0_14Vp8DecoderTypeE) { RET(ready_event(c)); }
HLE(_ZN2nn5codec27SubmitVp8DecoderInputBufferEPmPNS0_14Vp8DecoderTypeEPKvmNS_8TimeSpanEb) { dec_submit(c); }
HLE(_ZN2nn5codec29AcquireVp8DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Vp8DecoderTypeE) { dec_acquire(c); }
HLE(_ZN2nn5codec29ReleaseVp8DecoderOutputBufferEPNS0_14Vp8DecoderTypeEjNS_2gs16NativeSyncHandleE) { dec_release(c); }
HLE(_ZN2nn5codec35GetVp8DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_8TimeSpanEPbPNS0_14Vp8DecoderTypeEj) { dec_properties(c); }
