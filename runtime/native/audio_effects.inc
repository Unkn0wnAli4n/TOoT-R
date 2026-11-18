// nn::audio effects, this project's own (included by audio.cpp after the mixer, audio_hookshot.inc, whose mixes,
// voices and splitters they work on):
//   - Reverb (ReverbType): a pre-delay, early reflections by room kind, and a four-line feedback network for the
//     tail, its length, brightness and density from the game's parameters;
//   - Delay (DelayType): an echo per channel with feedback, a low pass in the loop and spread between channels;
//   - the BiquadFilter effect, and the float biquad filters of voices and of splitter destinations.
// An effect works on the buffers of the mix it was added to, in the order the game added them, before the mix goes
// on to its destination. They are models after the documented parameters, not copies of the console's processing:
// rooms sound like rooms, not sample for sample like the console's.
// OOT_AUDIO_EFFECTS=0: all left out (the dry sound, as before). HOS_AUDIO_TRACE=1 logs what the game sets.
enum { FX_REVERB = 1, FX_DELAY, FX_BIQUAD };
#define MAX_FX 48
#define FX_HANDLE 1000                 /* (their handles' indices start here: below are the mixer's I3DL2 reverbs) */
#define FX_PRE (RATE * 45 / 100)       /* pre-delay and early reflections: 0.45 s */
#define FX_FDN 8704                    /* a line of the tail: up to 181 ms */
#define FX_AP 512                      /* a diffusion all-pass */
typedef struct {
    int used, kind, mix, enabled, n, order;
    uint64_t buffer;                   /* the game's work buffer, given back when the effect is removed */
    int8_t in[MAX_CH], out[MAX_CH];
    /* reverb */
    int early_mode, late_mode;
    float early_gain, late_gain, hf_ratio, coloration, reverb_gain, out_gain, dry_gain;
    int64_t predelay_ns, decay_ns;
    float *pre[2];
    int pre_pos;
    float *line[4];
    int pos[4], len[4];
    float lp[4];
    float *ap[2][2];
    int ap_pos[2][2];
    /* delay */
    int64_t delay_ns, delay_max_ns;
    float in_gain, feedback, spread, lowpass;
    float *dl[MAX_CH];
    int dl_len, dl_pos;
    float dl_lp[MAX_CH];
    /* biquad */
    float b[3], a[2], z[MAX_CH][2];
} Fx;
static Fx fxs[MAX_FX];
static int fx_orders;

static int fx_off(void) {
    static int off = -1;
    if (off < 0) { const char *e = getenv("OOT_AUDIO_EFFECTS"); off = e && e[0] == '0'; }
    return off;
}
static Fx *fx_of(uint64_t obj, int kind) {
    const int i = get_handle(obj, K_EFFECT) - FX_HANDLE;
    return i >= 0 && i < MAX_FX && fxs[i].used && fxs[i].kind == kind ? &fxs[i] : NULL;
}
static void fx_free(Fx *x) {
    for (int k = 0; k < 2; k++) { free(x->pre[k]); for (int j = 0; j < 2; j++) free(x->ap[k][j]); }
    for (int k = 0; k < 4; k++) free(x->line[k]);
    for (int k = 0; k < MAX_CH; k++) free(x->dl[k]);
}
/* A new effect on the mix `mixobj` names; NULL when there is no room or no such mix. (with the lock held) */
static Fx *fx_new(int kind, uint64_t obj, uint64_t buffer, uint64_t mixobj, int channels) {
    int e = 0;
    while (e < MAX_FX && fxs[e].used) e++;
    const int m = mix_index(mixobj);
    if (e == MAX_FX || m < 0) return NULL;
    Fx *x = &fxs[e];
    memset(x, 0, sizeof *x);
    x->kind = kind; x->mix = m; x->enabled = 1; x->order = ++fx_orders; x->buffer = buffer;
    x->n = channels < 0 ? 0 : channels > MAX_CH ? MAX_CH : channels;
    for (int k = 0; k < MAX_CH; k++) x->in[k] = x->out[k] = (int8_t)k;
    put_handle(obj, K_EFFECT, FX_HANDLE + e);
    return x;
}
static void fx_remove(Ctx *c, int kind) {   /* Remove*(config, effect, mix): the work buffer goes back to the game */
    au_lock();
    Fx *x = fx_of(ARG(1), kind);
    uint64_t buffer = 0;
    if (x) { buffer = x->buffer; x->used = 0; fx_free(x); memset(x, 0, sizeof *x); put_handle(ARG(1), K_EFFECT, -1); }
    au_unlock();
    RET(buffer);
}
static void fx_io(Fx *x, uint64_t in, uint64_t out, int count) {
    if (!x) return;
    const int n = count < 0 ? 0 : count > MAX_CH ? MAX_CH : count;
    for (int k = 0; k < n; k++) { x->in[k] = *(int8_t *)GP(in + k); x->out[k] = *(int8_t *)GP(out + k); }
    x->n = n;
    AUT("audio: effect kind %d: %d channels, in %d %d out %d %d", x->kind, n, x->in[0], x->in[1], x->out[0], x->out[1]);
}

/* ---- Reverb ---------------------------------------------------------------------------------------------------- */
/* (which side of the room a channel is on, by the effect's channel count: 1, 2, 4 or 6 as FL FR FC LFE RL RR) */
enum { ROLE_L, ROLE_R, ROLE_C, ROLE_LFE, ROLE_RL, ROLE_RR, ROLE_MONO };
static int fx_role(int n, int c) {
    if (n == 1) return ROLE_MONO;
    if (n == 6) return c;
    if (n == 4) return c < 2 ? c : c + 2;
    return (c & 1) ? ROLE_R : ROLE_L;
}
static const float fx_early_span_ms[5] = {14.0f, 34.0f, 56.0f, 96.0f, 0.0f};   /* small room, large room, hall, cavern, none */
static const float fx_tap_l[5] = {0.00f, 0.21f, 0.36f, 0.57f, 0.83f}, fx_tap_r[5] = {0.07f, 0.27f, 0.43f, 0.66f, 1.00f};
static const float fx_tap_g[5] = {0.70f, -0.62f, 0.55f, -0.47f, 0.40f};
static const float fx_late_ms[5][4] = {   /* room, hall, metal corridor, cavern, the longest */
    {43.7f, 53.9f, 67.3f, 79.1f}, {53.9f, 79.1f, 116.2f, 130.6f}, {5.3f, 9.7f, 13.1f, 17.9f},
    {47.0f, 71.0f, 103.0f, 170.0f}, {53.9f, 79.1f, 116.2f, 170.6f}};
static const int fx_ap_len[2][2] = {{241, 83}, {263, 97}};
static float fx_allpass(float *buf, int *pos, int len, float g, float x) {
    const float d = buf[*pos], y = d - g * x;
    buf[*pos] = x + g * y;
    *pos = (*pos + 1) % len;
    return y;
}
static void fx_reverb_frame(Fx *x, Mix *m) {
    if (!x->pre[0]) return;
    const int em = x->early_mode < 0 || x->early_mode > 4 ? 0 : x->early_mode, lm = x->late_mode < 0 || x->late_mode > 4 ? 0 : x->late_mode;
    const int span = (int)(fx_early_span_ms[em] * RATE / 1000.0f);
    int pd = (int)((double)x->predelay_ns * RATE / 1e9);
    if (pd < 0) pd = 0;
    if (pd > FX_PRE - span - 2) pd = FX_PRE - span - 2;
    int tl[5], tr[5];
    for (int t = 0; t < 5; t++) { tl[t] = pd + (int)(fx_tap_l[t] * span); tr[t] = pd + (int)(fx_tap_r[t] * span); }
    const float decay = x->decay_ns > 0 ? fminf(fmaxf((float)x->decay_ns / 1e9f, 0.1f), 20.0f) : 1.5f;
    const float hf = fminf(fmaxf(x->hf_ratio, 0.1f), 1.0f);
    float g[4], damp[4];
    for (int k = 0; k < 4; k++) {
        const int L = (int)(fx_late_ms[lm][k] * RATE / 1000.0f);
        x->len[k] = L < 32 ? 32 : L >= FX_FDN ? FX_FDN - 1 : L;
        if (x->pos[k] >= x->len[k]) x->pos[k] = 0;
        g[k] = powf(10.0f, -3.0f * x->len[k] / (RATE * decay));
        /* (the high end fades in decay x ratio: a one-pole low pass in the loop whose gain at the top is that share) */
        const float r = powf(10.0f, -3.0f * x->len[k] / (RATE * decay) * (1.0f / hf - 1.0f));
        damp[k] = (1.0f - r) / (1.0f + r);
    }
    const float apg = 0.2f + 0.5f * fminf(fmaxf(x->coloration, 0.0f), 1.0f);
    const float early_gain = em == 4 ? 0.0f : x->early_gain * 0.5f, late_gain = x->late_gain;
    const int n = x->n;
    for (int i = 0; i < FRAME; i++) {
        float in_s[MAX_CH], xl = 0, xr = 0;
        for (int c = 0; c < n; c++) {
            const int b = x->in[c];
            float s = b >= 0 && b < m->buffers ? m->buf[b][i] : 0.0f;
            if (!isfinite(s)) s = 0;   /* (a NaN would stay in the feedback lines for good) */
            in_s[c] = s;
            switch (fx_role(n, c)) {
            case ROLE_L: case ROLE_RL: xl += s; break;
            case ROLE_R: case ROLE_RR: xr += s; break;
            case ROLE_C: xl += s * 0.5f; xr += s * 0.5f; break;
            case ROLE_MONO: xl += s; xr += s; break;
            default: break;
            }
        }
        const int p = x->pre_pos;
        x->pre[0][p] = xl; x->pre[1][p] = xr;
        float el = 0, er = 0;
        for (int t = 0; t < 5; t++) {
            el += fx_tap_g[t] * x->pre[t & 1][(p - tl[t] + FX_PRE) % FX_PRE];
            er += fx_tap_g[t] * x->pre[(t + 1) & 1][(p - tr[t] + FX_PRE) % FX_PRE];
        }
        float il = x->pre[0][(p - pd - span + FX_PRE) % FX_PRE] * x->reverb_gain, ir = x->pre[1][(p - pd - span + FX_PRE) % FX_PRE] * x->reverb_gain;
        x->pre_pos = (p + 1) % FX_PRE;
        for (int j = 0; j < 2; j++) {
            il = fx_allpass(x->ap[0][j], &x->ap_pos[0][j], fx_ap_len[0][j], apg, il);
            ir = fx_allpass(x->ap[1][j], &x->ap_pos[1][j], fx_ap_len[1][j], apg, ir);
        }
        float o[4];
        for (int k = 0; k < 4; k++) o[k] = x->line[k][x->pos[k]];
        const float hm[4] = {(o[0] + o[1] + o[2] + o[3]) * 0.5f, (o[0] - o[1] + o[2] - o[3]) * 0.5f,
                             (o[0] + o[1] - o[2] - o[3]) * 0.5f, (o[0] - o[1] - o[2] + o[3]) * 0.5f};
        for (int k = 0; k < 4; k++) {
            x->lp[k] = hm[k] * (1.0f - damp[k]) + x->lp[k] * damp[k];
            x->line[k][x->pos[k]] = ((k & 1) ? ir : il) * 0.7f + x->lp[k] * g[k];
            x->pos[k] = (x->pos[k] + 1) % x->len[k];
        }
        const float ll = (o[0] + o[2]) * 0.5f, lr = (o[1] + o[3]) * 0.5f, sl = (o[0] - o[2]) * 0.5f, sr = (o[1] - o[3]) * 0.5f;
        for (int c = 0; c < n; c++) {
            const int bo = x->out[c];
            if (bo < 0 || bo >= m->buffers) continue;
            float e = 0, l = 0;
            switch (fx_role(n, c)) {
            case ROLE_L: e = el; l = ll; break;
            case ROLE_R: e = er; l = lr; break;
            case ROLE_C: e = (el + er) * 0.35f; l = (ll + lr) * 0.35f; break;
            case ROLE_RL: e = el * 0.5f; l = sl; break;
            case ROLE_RR: e = er * 0.5f; l = sr; break;
            case ROLE_MONO: e = (el + er) * 0.5f; l = (ll + lr) * 0.5f; break;
            default: break;
            }
            m->buf[bo][i] = x->dry_gain * in_s[c] + x->out_gain * (early_gain * e + late_gain * l);
        }
    }
}

/* ---- Delay ----------------------------------------------------------------------------------------------------- */
static void fx_delay_frame(Fx *x, Mix *m) {
    if (x->dl_len < 2) return;
    int D = (int)((double)x->delay_ns * RATE / 1e9);
    if (D < 1) D = 1;
    if (D > x->dl_len - 1) D = x->dl_len - 1;
    const int n = x->n;
    const float lpa = fminf(fmaxf(x->lowpass, 0.0f), 0.99f), spread = fminf(fmaxf(x->spread, 0.0f), 1.0f);
    const float fb = fminf(fmaxf(x->feedback, 0.0f), 0.98f);
    for (int i = 0; i < FRAME; i++) {
        float in_s[MAX_CH], d[MAX_CH], sum = 0;
        const int p = x->dl_pos, rd = (p - D + x->dl_len) % x->dl_len;
        for (int c = 0; c < n; c++) {
            const int b = x->in[c];
            float s = b >= 0 && b < m->buffers ? m->buf[b][i] : 0.0f;
            if (!isfinite(s)) s = 0;
            in_s[c] = s;
            x->dl_lp[c] = x->dl[c][rd] * (1.0f - lpa) + x->dl_lp[c] * lpa;
            d[c] = x->dl_lp[c];
            sum += d[c];
        }
        for (int c = 0; c < n; c++) {
            const float others = n > 1 ? (sum - d[c]) / (float)(n - 1) : d[c];
            x->dl[c][p] = x->in_gain * in_s[c] + fb * ((1.0f - spread) * d[c] + spread * others);
            const int bo = x->out[c];
            if (bo >= 0 && bo < m->buffers) m->buf[bo][i] = x->dry_gain * in_s[c] + d[c];
        }
        x->dl_pos = (p + 1) % x->dl_len;
    }
}

/* ---- biquad filters (transposed direct form II, the denominator negated, as the voices' in the mixer) ----------- */
static void fx_biquad_run(const float *b, const float *a, float *z, float *s) {
    float z0 = z[0], z1 = z[1];
    for (int i = 0; i < FRAME; i++) {
        const float v = s[i], y = b[0] * v + z0;
        z0 = b[1] * v + a[0] * y + z1;
        z1 = b[2] * v + a[1] * y;
        s[i] = y;
    }
    if (!isfinite(z0) || !isfinite(z1)) { z0 = z1 = 0; memset(s, 0, FRAME * sizeof *s); }   /* (blown up: start over) */
    z[0] = z0; z[1] = z1;
}
static void fx_biquad_frame(Fx *x, Mix *m) {
    for (int c = 0; c < x->n; c++) {
        const int bi = x->in[c], bo = x->out[c];
        if (bi < 0 || bi >= m->buffers || bo < 0 || bo >= m->buffers) continue;
        float s[FRAME];
        memcpy(s, m->buf[bi], sizeof s);
        fx_biquad_run(x->b, x->a, x->z[c], s);
        memcpy(m->buf[bo], s, sizeof s);
    }
}
/* TBiquadFilterParameter<float>: bool enable @0, float numerator[3] @4, float denominator[2] @16 */
static int fx_read_biquad(uint64_t p, float *b, float *a) {
    memcpy(b, GP(p + 4), 12);
    memcpy(a, GP(p + 16), 8);
    for (int k = 0; k < 3; k++) if (!isfinite(b[k])) b[k] = 0;
    for (int k = 0; k < 2; k++) if (!isfinite(a[k])) a[k] = 0;
    return *(uint8_t *)GP(p) != 0;
}

/* (render_frame: a mix's effects, in the order the game added them) */
static void fx_test(int mix);
static void fx_frame(int mix) {
    if (fx_off()) return;
    fx_test(mix);
    int last = 0;
    for (;;) {
        Fx *next = NULL;
        for (int e = 0; e < MAX_FX; e++)
            if (fxs[e].used && fxs[e].mix == mix && fxs[e].order > last && (!next || fxs[e].order < next->order)) next = &fxs[e];
        if (!next) break;
        last = next->order;
        if (!next->enabled || next->n <= 0) continue;
        if (next->kind == FX_REVERB) fx_reverb_frame(next, &mixes[mix]);
        else if (next->kind == FX_DELAY) fx_delay_frame(next, &mixes[mix]);
        else if (next->kind == FX_BIQUAD) fx_biquad_frame(next, &mixes[mix]);
    }
}
/* (render_frame, after the voices: what they sent to splitter destinations that have a filter on) */
static void fx_splitters_frame(void) {
    for (int i = 0; i < MAX_SPLITTERS; i++) {
        Splitter *s = &splitters[i];
        if (!s->used || !s->fin_used) continue;
        s->fin_used = 0;
        const int nch = s->channels > 0 && s->channels < MAX_CH ? s->channels : MAX_CH;
        for (int d = 0; d < s->dests; d++) {
            if (!s->bq_on[d][0] && !s->bq_on[d][1]) continue;
            const int mi = s->dest_mix[d];
            if (mi < 0 || !mixes[mi].used) continue;
            Mix *m = &mixes[mi];
            for (int ch = 0; ch < nch; ch++) {
                float t[FRAME];
                memcpy(t, s->fin[ch], sizeof t);
                for (int f = 0; f < 2; f++)
                    if (s->bq_on[d][f] && !fx_off()) fx_biquad_run(s->bq_b[d][f], s->bq_a[d][f], s->bq_z[d][f][ch], t);
                for (int b = 0; b < m->buffers; b++) {
                    const float g0 = s->ramp[d][ch][b], g1 = s->vol[d][ch][b];
                    if (g0 == 0.0f && g1 == 0.0f) continue;
                    const float dg = (g1 - g0) / FRAME;
                    for (int k = 0; k < FRAME; k++) m->buf[b][k] += t[k] * (g0 + dg * (float)(k + 1));
                }
            }
        }
    }
}

/* ---- the game's calls -------------------------------------------------------------------------------------------- */
/* AddReverb(config, ReverbType*, buffer, size, mix, channels) */
static void fx_add_reverb(Ctx *c) {
    au_lock();
    Fx *x = fx_new(FX_REVERB, ARG(1), ARG(2), ARG(4), (int)ARG(5));
    if (x) {
        x->early_gain = 0.7f; x->late_gain = 0.5f; x->hf_ratio = 0.8f; x->coloration = 0.7f; x->reverb_gain = 0.2f;
        x->out_gain = 1.0f; x->dry_gain = 1.0f; x->predelay_ns = 15000000; x->decay_ns = 1500000000ll; x->late_mode = 1;
        for (int k = 0; k < 2; k++) {
            x->pre[k] = (float *)calloc(FX_PRE, sizeof(float));
            for (int j = 0; j < 2; j++) x->ap[k][j] = (float *)calloc(FX_AP, sizeof(float));
        }
        for (int k = 0; k < 4; k++) x->line[k] = (float *)calloc(FX_FDN, sizeof(float));
        x->used = 1;
    }
    au_unlock();
    AUT("audio: reverb %s on mix %d, %d channels", x ? "added" : "refused", mix_index(ARG(4)), (int)ARG(5));
    RET(x ? 0 : (0x1A9 << 9 | 153));
}
AUDIO_NATIVE("_ZN2nn5audio9AddReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPvmPNS0_10SubMixTypeEi") { fx_add_reverb(c); }
AUDIO_NATIVE("_ZN2nn5audio9AddReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPvmPNS0_12FinalMixTypeEi") { fx_add_reverb(c); }
AUDIO_NATIVE("_ZN2nn5audio12RemoveReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPNS0_10SubMixTypeE") { fx_remove(c, FX_REVERB); }
AUDIO_NATIVE("_ZN2nn5audio12RemoveReverbEPNS0_19AudioRendererConfigEPNS0_10ReverbTypeEPNS0_12FinalMixTypeE") { fx_remove(c, FX_REVERB); }
#define FX_FLOAT(SYMBOL, KIND, FIELD) \
    AUDIO_NATIVE(SYMBOL) { Fx *x = fx_of(ARG(0), KIND); if (x) x->FIELD = farg(c, 0); AUT("audio: %s = %.3f", #FIELD, farg(c, 0)); }
#define FX_TIME(SYMBOL, KIND, FIELD) \
    AUDIO_NATIVE(SYMBOL) { Fx *x = fx_of(ARG(0), KIND); if (x) x->FIELD = (int64_t)ARG(1); AUT("audio: %s = %.1f ms", #FIELD, (double)(int64_t)ARG(1) / 1e6); }
#define FX_INT(SYMBOL, KIND, FIELD) \
    AUDIO_NATIVE(SYMBOL) { Fx *x = fx_of(ARG(0), KIND); if (x) x->FIELD = (int)ARG(1); AUT("audio: %s = %d", #FIELD, (int)ARG(1)); }
FX_INT("_ZN2nn5audio18SetReverbEarlyModeEPNS0_10ReverbTypeENS1_9EarlyModeE", FX_REVERB, early_mode)
FX_INT("_ZN2nn5audio17SetReverbLateModeEPNS0_10ReverbTypeENS1_8LateModeE", FX_REVERB, late_mode)
FX_FLOAT("_ZN2nn5audio18SetReverbEarlyGainEPNS0_10ReverbTypeEf", FX_REVERB, early_gain)
FX_FLOAT("_ZN2nn5audio17SetReverbLateGainEPNS0_10ReverbTypeEf", FX_REVERB, late_gain)
FX_FLOAT("_ZN2nn5audio32SetReverbHighFrequencyDecayRatioEPNS0_10ReverbTypeEf", FX_REVERB, hf_ratio)
FX_FLOAT("_ZN2nn5audio19SetReverbColorationEPNS0_10ReverbTypeEf", FX_REVERB, coloration)
FX_FLOAT("_ZN2nn5audio19SetReverbReverbGainEPNS0_10ReverbTypeEf", FX_REVERB, reverb_gain)
FX_FLOAT("_ZN2nn5audio16SetReverbOutGainEPNS0_10ReverbTypeEf", FX_REVERB, out_gain)
FX_FLOAT("_ZN2nn5audio16SetReverbDryGainEPNS0_10ReverbTypeEf", FX_REVERB, dry_gain)
FX_TIME("_ZN2nn5audio21SetReverbPredelayTimeEPNS0_10ReverbTypeENS_8TimeSpanE", FX_REVERB, predelay_ns)
FX_TIME("_ZN2nn5audio18SetReverbDecayTimeEPNS0_10ReverbTypeENS_8TimeSpanE", FX_REVERB, decay_ns)
AUDIO_NATIVE("_ZN2nn5audio16SetReverbEnabledEPNS0_10ReverbTypeEb") { Fx *x = fx_of(ARG(0), FX_REVERB); if (x) x->enabled = (int)(ARG(1) & 1); AUT("audio: reverb %s", ARG(1) & 1 ? "on" : "off"); }
AUDIO_NATIVE("_ZN2nn5audio15IsReverbEnabledEPKNS0_10ReverbTypeE") { Fx *x = fx_of(ARG(0), FX_REVERB); RET(x ? x->enabled : 0); }
AUDIO_NATIVE("_ZN2nn5audio20SetReverbInputOutputEPNS0_10ReverbTypeEPKaS4_i") { fx_io(fx_of(ARG(0), FX_REVERB), ARG(1), ARG(2), (int)ARG(3)); }
AUDIO_NATIVE("_ZN2nn5audio20GetReverbInputOutputEPKNS0_10ReverbTypeEPaS4_Pii") {
    Fx *x = fx_of(ARG(0), FX_REVERB);
    const int n = x ? x->n : 0;
    for (int k = 0; k < n && k < (int)ARG(4); k++) { *(int8_t *)GP(ARG(1) + k) = x->in[k]; *(int8_t *)GP(ARG(2) + k) = x->out[k]; }
    if (ARG(3)) G32(ARG(3)) = (uint32_t)n;
}

/* AddDelay(config, DelayType*, buffer, size, mix, TimeSpan longest, channels) */
static void fx_add_delay(Ctx *c) {
    au_lock();
    Fx *x = fx_new(FX_DELAY, ARG(1), ARG(2), ARG(4), (int)ARG(6));
    if (x) {
        int64_t longest = (int64_t)ARG(5);
        if (longest < 1000000) longest = 1000000;
        if (longest > 5000000000ll) longest = 5000000000ll;   /* (5 s: 240k samples a channel) */
        x->delay_max_ns = x->delay_ns = longest;
        x->dl_len = (int)((double)longest * RATE / 1e9) + 2;
        for (int k = 0; k < x->n; k++) x->dl[k] = (float *)calloc((size_t)x->dl_len, sizeof(float));
        x->in_gain = 1.0f; x->feedback = 0.4f; x->dry_gain = 1.0f;
        x->used = 1;
    }
    au_unlock();
    AUT("audio: delay %s on mix %d, %d channels, up to %.0f ms", x ? "added" : "refused", mix_index(ARG(4)), (int)ARG(6), (double)(int64_t)ARG(5) / 1e6);
    RET(x ? 0 : (0x1A9 << 9 | 153));
}
AUDIO_NATIVE("_ZN2nn5audio8AddDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPvmPNS0_10SubMixTypeENS_8TimeSpanEi") { fx_add_delay(c); }
AUDIO_NATIVE("_ZN2nn5audio8AddDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPvmPNS0_12FinalMixTypeENS_8TimeSpanEi") { fx_add_delay(c); }
AUDIO_NATIVE("_ZN2nn5audio11RemoveDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPNS0_10SubMixTypeE") { fx_remove(c, FX_DELAY); }
AUDIO_NATIVE("_ZN2nn5audio11RemoveDelayEPNS0_19AudioRendererConfigEPNS0_9DelayTypeEPNS0_12FinalMixTypeE") { fx_remove(c, FX_DELAY); }
AUDIO_NATIVE("_ZN2nn5audio12SetDelayTimeEPNS0_9DelayTypeENS_8TimeSpanE") {
    Fx *x = fx_of(ARG(0), FX_DELAY);
    if (x) x->delay_ns = (int64_t)ARG(1) > x->delay_max_ns ? x->delay_max_ns : (int64_t)ARG(1);
    AUT("audio: delay time = %.1f ms", (double)(int64_t)ARG(1) / 1e6);
}
AUDIO_NATIVE("_ZN2nn5audio12GetDelayTimeEPKNS0_9DelayTypeE") { Fx *x = fx_of(ARG(0), FX_DELAY); RET(x ? (uint64_t)x->delay_ns : 0); }
FX_FLOAT("_ZN2nn5audio14SetDelayInGainEPNS0_9DelayTypeEf", FX_DELAY, in_gain)
FX_FLOAT("_ZN2nn5audio20SetDelayFeedbackGainEPNS0_9DelayTypeEf", FX_DELAY, feedback)
FX_FLOAT("_ZN2nn5audio15SetDelayDryGainEPNS0_9DelayTypeEf", FX_DELAY, dry_gain)
FX_FLOAT("_ZN2nn5audio21SetDelayChannelSpreadEPNS0_9DelayTypeEf", FX_DELAY, spread)
FX_FLOAT("_ZN2nn5audio21SetDelayLowPassAmountEPNS0_9DelayTypeEf", FX_DELAY, lowpass)
AUDIO_NATIVE("_ZN2nn5audio15SetDelayEnabledEPNS0_9DelayTypeEb") { Fx *x = fx_of(ARG(0), FX_DELAY); if (x) x->enabled = (int)(ARG(1) & 1); AUT("audio: delay %s", ARG(1) & 1 ? "on" : "off"); }
AUDIO_NATIVE("_ZN2nn5audio14IsDelayEnabledEPKNS0_9DelayTypeE") { Fx *x = fx_of(ARG(0), FX_DELAY); RET(x ? x->enabled : 0); }
AUDIO_NATIVE("_ZN2nn5audio16IsDelayRemovableEPKNS0_9DelayTypeE") { RET(1); }
AUDIO_NATIVE("_ZN2nn5audio19SetDelayInputOutputEPNS0_9DelayTypeEPKaS4_i") {
    au_lock();
    Fx *x = fx_of(ARG(0), FX_DELAY);
    fx_io(x, ARG(1), ARG(2), (int)ARG(3));
    if (x) for (int k = 0; k < x->n; k++) if (!x->dl[k]) x->dl[k] = (float *)calloc((size_t)x->dl_len, sizeof(float));
    au_unlock();
}

/* AddBiquadFilter(config, BiquadFilterType*, mix) */
static void fx_add_biquad(Ctx *c) {
    au_lock();
    Fx *x = fx_new(FX_BIQUAD, ARG(1), 0, ARG(2), 0);
    if (x) { x->b[0] = 1.0f; x->used = 1; }
    au_unlock();
    AUT("audio: biquad filter effect %s on mix %d", x ? "added" : "refused", mix_index(ARG(2)));
    RET(x ? 0 : (0x1A9 << 9 | 153));
}
AUDIO_NATIVE("_ZN2nn5audio15AddBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_10SubMixTypeE") { fx_add_biquad(c); }
AUDIO_NATIVE("_ZN2nn5audio15AddBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_12FinalMixTypeE") { fx_add_biquad(c); }
AUDIO_NATIVE("_ZN2nn5audio18RemoveBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_10SubMixTypeE") { fx_remove(c, FX_BIQUAD); }
AUDIO_NATIVE("_ZN2nn5audio18RemoveBiquadFilterEPNS0_19AudioRendererConfigEPNS0_16BiquadFilterTypeEPNS0_12FinalMixTypeE") { fx_remove(c, FX_BIQUAD); }
AUDIO_NATIVE("_ZN2nn5audio24SetBiquadFilterParameterEPNS0_16BiquadFilterTypeERKNS0_22TBiquadFilterParameterIfEE") {
    au_lock();
    Fx *x = fx_of(ARG(0), FX_BIQUAD);
    if (x && ARG(1)) {
        fx_read_biquad(ARG(1), x->b, x->a);
        static int logged;
        if (logged++ < 8) AUT("audio: biquad filter effect: b %.4f %.4f %.4f a %.4f %.4f", x->b[0], x->b[1], x->b[2], x->a[0], x->a[1]);
    }
    au_unlock();
}
AUDIO_NATIVE("_ZN2nn5audio22SetBiquadFilterEnabledEPNS0_16BiquadFilterTypeEb") {
    au_lock();
    Fx *x = fx_of(ARG(0), FX_BIQUAD);
    if (x) { if ((ARG(1) & 1) && !x->enabled) memset(x->z, 0, sizeof x->z); x->enabled = (int)(ARG(1) & 1); }
    au_unlock();
}
AUDIO_NATIVE("_ZN2nn5audio21IsBiquadFilterEnabledEPKNS0_16BiquadFilterTypeE") { Fx *x = fx_of(ARG(0), FX_BIQUAD); RET(x ? x->enabled : 0); }
AUDIO_NATIVE("_ZN2nn5audio26SetBiquadFilterInputOutputEPNS0_16BiquadFilterTypeEPKaS4_i") { fx_io(fx_of(ARG(0), FX_BIQUAD), ARG(1), ARG(2), (int)ARG(3)); }

/* SetVoiceBiquadFilterParameter(VoiceType*, int filter, const TBiquadFilterParameter<float>&): with the voice's other
   changes, at the next update (the mixer's queue) */
AUDIO_NATIVE("_ZN2nn5audio29SetVoiceBiquadFilterParameterEPNS0_9VoiceTypeEiRKNS0_22TBiquadFilterParameterIfEE") {
    au_lock();
    Voice *v = voice_of(ARG(0));
    const int f = (int)ARG(1);
    if (v && f >= 0 && f < 2 && ARG(2)) {
        Cmd k = {.op = OP_VOICE_BIQUAD, .idx = (int)(v - voices), .i0 = f};
        k.i1 = fx_read_biquad(ARG(2), k.b, k.a) && !fx_off();
        submit(&k);
        static int logged;
        if (k.i1 && logged++ < 8) AUT("audio: voice biquad %d (float): b %.4f %.4f %.4f a %.4f %.4f", f, k.b[0], k.b[1], k.b[2], k.a[0], k.a[1]);
    }
    au_unlock();
}
/* SetSplitterBiquadFilterParameter(SplitterType*, int destination, mix, int filter, const TBiquadFilterParameter<float>&) */
static void fx_split_biquad(Ctx *c) {
    au_lock();
    Splitter *s = splitter_of(ARG(0));
    const int d = (int)ARG(1), f = (int)ARG(3);
    if (s && d >= 0 && d < MAX_SPLIT_DESTS && f >= 0 && f < 2 && ARG(4)) {
        float b[3], a[2];
        const int on = fx_read_biquad(ARG(4), b, a);
        if (on && !s->bq_on[d][f]) memset(s->bq_z[d][f], 0, sizeof s->bq_z[d][f]);
        s->bq_on[d][f] = on;
        memcpy(s->bq_b[d][f], b, sizeof b);
        memcpy(s->bq_a[d][f], a, sizeof a);
        static int logged;
        if (on && logged++ < 8) AUT("audio: splitter destination %d biquad %d: b %.4f %.4f %.4f a %.4f %.4f", d, f, b[0], b[1], b[2], a[0], a[1]);
    }
    au_unlock();
}
AUDIO_NATIVE("_ZN2nn5audio32SetSplitterBiquadFilterParameterEPNS0_12SplitterTypeEiPNS0_10SubMixTypeEiRKNS0_22TBiquadFilterParameterIfEE") { fx_split_biquad(c); }
AUDIO_NATIVE("_ZN2nn5audio32SetSplitterBiquadFilterParameterEPNS0_12SplitterTypeEiPNS0_12FinalMixTypeEiRKNS0_22TBiquadFilterParameterIfEE") { fx_split_biquad(c); }

/* Development: OOT_AUDIO_TEST_REVERB=1 puts a hall reverb, OOT_AUDIO_TEST_DELAY=1 an echo, on the final mix (the game
   uses its own only in some places): to hear and measure them anywhere. */
static void fx_test(int mix) {
    static int state = -1;
    if (state < 0) state = (getenv("OOT_AUDIO_TEST_REVERB") ? 1 : 0) | (getenv("OOT_AUDIO_TEST_DELAY") ? 2 : 0);
    if (!(state & 3) || mix != sink_mix) return;
    int e = 0;
    while (e < MAX_FX && fxs[e].used) e++;
    if (e == MAX_FX) return;
    Fx *x = &fxs[e];
    memset(x, 0, sizeof *x);
    x->mix = mix; x->enabled = 1; x->order = ++fx_orders; x->n = 2;
    for (int k = 0; k < MAX_CH; k++) x->in[k] = x->out[k] = (int8_t)k;
    if (state & 1) {
        state &= ~1;
        x->kind = FX_REVERB; x->early_mode = 2; x->late_mode = 1; x->early_gain = 0.7f; x->late_gain = 0.7f; x->hf_ratio = 0.6f;
        x->coloration = 0.7f; x->reverb_gain = 0.6f; x->out_gain = 0.8f; x->dry_gain = 1.0f; x->predelay_ns = 20000000; x->decay_ns = 2500000000ll;
        for (int k = 0; k < 2; k++) { x->pre[k] = (float *)calloc(FX_PRE, sizeof(float)); for (int j = 0; j < 2; j++) x->ap[k][j] = (float *)calloc(FX_AP, sizeof(float)); }
        for (int k = 0; k < 4; k++) x->line[k] = (float *)calloc(FX_FDN, sizeof(float));
    } else {
        state &= ~2;
        x->kind = FX_DELAY; x->delay_ns = x->delay_max_ns = 285000000; x->dl_len = (int)(0.285 * RATE) + 2;
        for (int k = 0; k < 2; k++) x->dl[k] = (float *)calloc((size_t)x->dl_len, sizeof(float));
        x->in_gain = 0.85f; x->feedback = 0.4f; x->spread = 1.0f; x->lowpass = 0.3f; x->dry_gain = 1.0f;
    }
    x->used = 1;
}
