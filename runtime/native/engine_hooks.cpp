// Engine hooks: what the calls tools/oot_patches.py puts into the recompiled game do (oot_site, oot_call, oot_fn; the
// site numbers are that file's). Each reads the settings window's value every time, so a setting applies at once and
// its default leaves the game as it shipped. The sites and what the game's code does at each were found by Longshot
// (nx2/oot_patches.py, oot_shadow.py, oot_fov.py, host/oot_heap_retry.c: GPL-3) for this same game version; the code
// here is written for this runtime. Guest addresses are host addresses in this runtime.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <zstd.h>

#include <map>
#include <mutex>
#include "../ops.h"
#include "../rt.h"
#include "../addresses.h"
#include "graphics_settings.h"
#include "input_settings.h"
#include "debug_tools.h"
#include "dlss_math.h"   // (third_party/longshot-upscale)
#include "vk_temporal.h"
namespace nvn { u64 CurrentFrameNumber(); }

extern "C" {
extern volatile int oot_fixed_resolution;
extern volatile unsigned oot_fixed_scale_bits;
}

void oot_ensure_window(); // (nvn/vk.cpp)

namespace {
constexpr u64 Main = 0x800000000ull;
bool Readable(u64 a) { return a >= Main && a < (1ull << 39) - 0x100; }
template <typename T> T Load(u64 a) { T v; std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof v); return v; }
template <typename T> void Store(u64 a, T v) { std::memcpy(reinterpret_cast<void*>(a), &v, sizeof v); }
float LowFloat(const V128& v) { float f; std::memcpy(&f, &v, 4); return f; }
void SetLowFloat(V128& v, float f) { u64 lo = 0; std::memcpy(&lo, &f, 4); v = Q(lo, 0); } // (as a scalar load leaves the register)
bool Switch(const char* name, bool otherwise) { const char* e = std::getenv(name); return e ? std::atoi(e) != 0 : otherwise; }

// ---- render scale (the Graphics window's Render Resolution) -----------------------------------------------------------
// The engine's fixed-resolution path computes the 3D's scale (+0xe5c = scale * +0xe58 / +0xe50) only while +0xe2c bit 3
// is set, and nothing on that path sets it: it kept the dynamic path's last scale. Site 1 asks for the recompute while
// the scale in use is not the chosen one; site 2 clears the flags (+0xe20 bits 3, 4) that put a hard-coded 0.75 / 0.5 in
// its place.
void ScaleApply(Cpu* c) {
    if (!oot_fixed_resolution) return;
    const u64 module = c->x[20];
    if (!Readable(module)) return;
    float want; const unsigned bits = oot_fixed_scale_bits; std::memcpy(&want, &bits, 4);
    const float now = Load<float>(module + 0xe5c), e50 = Load<float>(module + 0xe50), e58 = Load<float>(module + 0xe58);
    const float d = now - (e50 != 0 ? want * e58 / e50 : want);
    if (d > 1e-4f || d < -1e-4f) c->x[8] |= 8;
}

// ---- the sun's shadow maps (Shadow Quality) ----------------------------------------------------------------------------
// w23: the cascades' size this frame, rounded to a multiple of 8 (1600 as shipped); x19: the shadow object.
void ShadowSize(Cpu* c) {
    {static const bool trace = std::getenv("OOT_SHADOW_TRACE") != nullptr; // (which shadow objects pass here, and their sizes)
     if (trace && Readable(c->x[19])) { static std::mutex m; static std::map<std::pair<u64, u32>, u64> seen; std::lock_guard l{m};
        if (++seen[{Load<u64>(c->x[19]), (u32)c->x[23]}] == 1) RT_LOG("shadow trace: object class %#llx (sun's is %#llx), size %u", (unsigned long long)Load<u64>(c->x[19]), (unsigned long long)OOT_AT(0x4b8a0a8), (u32)c->x[23]); }}
    const int choice = gfx::Current().shadow_quality;
    if (!choice || !Readable(c->x[19]) || Load<u64>(c->x[19]) != OOT_AT(0x4b8a0a8)) return; // (the sun's shadow object only)
    const u32 size = (u32)c->x[23];
    if (size < 64) return;
    const u32 game = 1600; // (the game's own level: smaller choices lower it, larger ones raise it)
    const u32 chosen = (u32)choice < game ? std::min(size, (u32)choice) : std::max(size, (u32)choice);
    if (chosen != size) { static int told = 0; if (told++ < 2) RT_LOG("shadow maps: %u texels a side (the game asked for %u)", chosen, size); }
    c->x[23] = chosen & ~7u;
}

// ---- field of view ------------------------------------------------------------------------------------------------------
// s0: the camera's vertical angle (radians) just loaded, before the game builds its projection and its culling from it.
// Only the gameplay view of the first camera, and not while an event camera runs.
void FieldOfView(Cpu* c) {
    const int percent = gfx::Current().fov;
    const u64 camera = c->x[19];
    if (percent == 100 || !Readable(camera)) return;
    const u64 views = Load<u64>(camera + 0x18);
    const u64 view = Readable(views) ? Load<u64>(views + 8) : 0;
    if (!Readable(view) || Load<u64>(view) != OOT_AT(0x4bf8a88)) return; // sead::LookAtCamera
    if (!OOT_AT(0x4c86bc0) || !OOT_AT(0x4c7e090)) return;
    const u64 singleton = Load<u64>(OOT_AT(0x4c86bc0));
    const u64 manager = Readable(singleton) ? Load<u64>(singleton) : 0;
    const u64 cameras = Readable(manager) && Load<u32>(manager + 32) ? Load<u64>(manager + 40) : 0;
    const u64 main_camera = Readable(cameras) ? Load<u64>(cameras) : 0;
    if (!Readable(main_camera) || Load<u64>(main_camera + 120) != view) return;
    const u64 event_singleton = Load<u64>(OOT_AT(0x4c7e090));
    const u64 events = Readable(event_singleton) ? Load<u64>(event_singleton) : 0;
    if (Readable(events) && Load<u64>(events + 104)) return;
    float angle = LowFloat(c->v[0]);
    if (!std::isfinite(angle) || angle <= 0.0f || angle >= 3.1415927f) return;
    angle = std::min(2.443461f, std::max(0.0872665f, angle * (percent / 100.0f))); // (5 to 140 degrees)
    SetLowFloat(c->v[0], angle);
}

void ScaleLowFloat(Cpu* c, int percent) {
    if (percent == 100) return;
    const float v = LowFloat(c->v[0]);
    if (std::isfinite(v)) SetLowFloat(c->v[0], v * (percent / 100.0f));
}

// ---- widescreen (Aspect Ratio at a shape of its own; read once, at start) ---------------------------------------------
// The game draws a 2560x1440 picture and a 1920x1080 scene with 16:9 cameras. With a wide ratio the picture keeps its
// height and takes the ratio's width (narrower than 16:9 too: the Steam Deck's 16:10 draws fewer pixels and has no bars),
// the scene three quarters of it, and the 16:9 cameras the picture's ratio. The HUD and menus keep their shape: in the
// middle 16:9 of a wider picture, scaled down to the width of a narrower one. Movies play in the middle 16:9.
// The sites are from NX2navi's patch list (credited there to Ivan0xFF, MIT) by way of Longshot's nx2/oot_wide.py.
struct Wide { float ratio = 0, squeeze = 1, margin = 0, qx = 1, qy = 1; u64 out = 0, scene = 0; u32 keep = 0xf; };
const Wide& WideValues() {
    static const Wide wide = [] {
        Wide w; const int choice = gfx::Current().aspect;
        double r = 0;
        if (const char* e = std::getenv("OOT_ASPECT")) r = std::atof(e);
        else if (choice == 2) { oot_ensure_window(); r = gfx::display_ratio.load(); }
        else if (choice == 3) r = 16.0 / 10.0; else if (choice == 4) r = 43.0 / 18.0; else if (choice == 5) r = 32.0 / 9.0; else if (choice == 6) r = 4.0 / 3.0;
        const double game = 16.0 / 9.0;
        // (the Output Resolution row: the picture's height, 1440 as the game ships; OOT_OUTPUT_HEIGHT for a run)
        const char* he = std::getenv("OOT_OUTPUT_HEIGHT");
        int height = he && *he ? std::atoi(he) : gfx::Current().output_height;
        if (height < 720 || height > 4320) height = 1440;
        const bool own_shape = !(r > 1.2) || (r < game * 1.01 && r > game * 0.99);
        if (own_shape && height == 1440) return w;
        if (own_shape) r = game;
        r = std::min(r, 48.0 / 9.0);
        unsigned oh = (unsigned)height / 8 * 8, ow = (unsigned)(oh * r / 8 + 0.5) * 8;
        const unsigned sh = oh * 3 / 4, sw = (unsigned)(ow * 0.75 / 8 + 0.5) * 8;
        w.out = (u64)oh << 32 | ow; w.scene = (u64)sh << 32 | sw; w.ratio = (float)ow / (float)oh;
        w.squeeze = (float)std::min(1.0, game / ((double)ow / oh)); w.margin = (1.0f - w.squeeze) / 2.0f;
        if (w.ratio < game) { w.qx = 1.0f; w.qy = (float)(w.ratio / game); w.keep &= ~2u; } // (narrower: the HUD at the picture's width, its shape kept)
        else { w.qx = w.squeeze; w.qy = 1.0f; }
        RT_LOG("widescreen: picture %ux%u, scene %ux%u, camera ratio %.4f, HUD scale %.3f x %.3f", ow, oh, sw, sh, w.ratio, w.qx, w.qy);
        return w; }();
    return wide;
}
// ---- the Output Resolution row (a picture of another height than the game's 1440) ----------------------------------------
// What has to follow the picture's size, as Longshot found (nx2/oot_patches.py OUT_EDITS, host/nn_applet.c
// hos_rt_pool_size): the 3D's largest size (above it the game takes a smaller upscale network: 1280 x 720 at 4K), the
// render targets' pool (0x13100000 bytes, sized for 2560 x 1440: a 4K picture ran out of it and the next target came
// back null) with the Graphics heap that holds it, the module budget that heap is counted against, and the heap made
// of the Graphics heap's rest, which keeps its shipped size.
u32 OutW() { const u64 o = WideValues().out; return o ? (u32)o : 2560; }
u32 OutH() { const u64 o = WideValues().out; return o ? (u32)(o >> 32) : 1440; }
u64 PoolSize() { // (the render targets' pool for this picture: by its pixels, never under 512 MiB once it is grown)
    constexpr u64 game = 0x13100000;
    const u64 px = (u64)OutW() * OutH();
    if (px <= 2560ull * 1440) return game;
    static const u64 size = [px] {
        const char* e = std::getenv("OOT_RT_POOL_MB");
        u64 v = e ? std::strtoull(e, nullptr, 0) << 20 : std::max<u64>((game * px / (2560 * 1440) + 0xfffff) & ~0xfffffull, 512ull << 20);
        if (!e && px > 3840ull * 2160) v = std::min<u64>(v, 900ull << 20);
        RT_LOG("output resolution: render target pool %llu MiB (the game's %llu MiB)", (unsigned long long)(v >> 20), (unsigned long long)(game >> 20));
        return v; }();
    return size;
}
u64 PoolGrowth() { return PoolSize() - 0x13100000ull; }
void OutputSite(Cpu* c, int id) {
    if (OutH() == 1440 && id != 104 && id != 105) return;
    switch (id) {
    case 101: c->x[24] = (u32)((u64)(u32)c->x[24] * OutW() / 2560); c->x[25] = (u32)((u64)(u32)c->x[25] * OutH() / 1440); break;
    case 102: { const float f = (float)(OutW() * 3 / 4); u32 b; std::memcpy(&b, &f, 4); c->x[8] = b; break; }
    case 103: { const float f = (float)(OutH() * 3 / 4); u32 b; std::memcpy(&b, &f, 4); c->x[8] = b; break; }
    case 104: if (PoolGrowth() && Readable(c->x[19]) && Load<u64>(c->x[19] + 1464) == 0x13100000ull) Store<u64>(c->x[19] + 1464, PoolSize()); break;
    case 105: if (const u64 g = PoolGrowth()) { const u64 budget = Readable(c->x[21]) ? Load<u64>(c->x[21] + (c->x[22] << 3)) : 0;
                  if ((budget == 0x34900000ull || budget == 0x348ffe20ull) && c->x[0] > g) c->x[0] -= g; } break;
    case 106: if (Readable(c->x[20]) && Load<u32>(c->x[20] + 8) == 2560) { Store<u32>(c->x[20] + 8, OutW()); Store<u32>(c->x[20] + 12, OutH()); } break;
    }
}
void OutputHeap(Cpu* c, u64 target, bool graphics) {
    const u64 g = PoolGrowth();
    if (g) { if (graphics) { if (c->x[0] == 0x34900000ull || c->x[0] == 0x348ffe20ull) c->x[0] += g; } else if (c->x[0] > g) c->x[0] -= g; }
    (rt_call)(c, target);
}
void WideRatio(Cpu* c, int vreg, int base) { // (a ratio just loaded: 16:9 becomes the picture's; with base, stored back at +0xb0)
    const Wide& w = WideValues();
    if (!(w.ratio > 0)) return;
    const float d = LowFloat(c->v[vreg]) - 16.0f / 9.0f;
    if (d > 0.01f || d < -0.01f) return;
    SetLowFloat(c->v[vreg], w.ratio);
    if (base >= 0 && Readable(c->x[base])) Store<float>(c->x[base] + 0xb0, w.ratio);
}
void WideHud(Cpu* c) { // (the 2D canvas's projection rows in q1 and q4; the canvas in x0, its screen 0x1b8 before it)
    const Wide& w = WideValues();
    if (!(w.ratio > 0)) return;
    float row0[4], row1[4]; std::memcpy(row0, &c->v[1], 16); std::memcpy(row1, &c->v[4], 16);
    if (row0[0] != 2.0f / 3840.0f || row1[1] != 2.0f / 2160.0f || !Readable(c->x[0] - 0x1b8)) return;
    const u64 table = Load<u64>(c->x[0] - 0x1b8);
    // (screens that cover the whole picture: the item wheel's and the pause menu's backgrounds, the event and loading fades)
    const u32 bit = table == OOT_AT(0x4bed098) ? 1u : table == OOT_AT(0x4be4fb0) ? 2u : table == OOT_AT(0x4be61c8) ? 4u : table == OOT_AT(0x4be7b30) ? 8u : table == OOT_AT(0x4bef4a0) ? 1u : 0u;
    if (bit & w.keep) return;
    for (float& v : row0) v *= w.qx;
    for (float& v : row1) v *= w.qy;
    std::memcpy(&c->v[1], row0, 16); std::memcpy(&c->v[4], row1, 16);
}
// ---- the HUD at a wide screen's edges (sites 17-22) -------------------------------------------------------------------
// On a picture wider than 16:9 the 2D canvas is drawn in the middle 16:9 (WideHud), so hearts, buttons and the minimap
// sat a bar's width in from the screen's edges. Here what belongs to a side of the HUD's layouts is moved out to it.
// The game's 2D layouts keep their panes in an array (0xb0 bytes each); a pane's final transform (3x4 at +0x28, its
// translation at +0x4c) is made from its own values and its parent's transform when its dirty bit 6 is set. The sites
// (found by ozwaldorf for Hookshot's ultrawide menus, by way of Longshot's nx2/oot_wide.py): 17 and 18 read a pane's
// dirty bits (set here while every pane is to be made again), 19 and 20 name the parent being multiplied by, 21 is
// where the translation was just stored (the pane in x19), 22 a pane's size as drawn (in s0, s1; the pane in x0).
// The rule, decided as each pane is placed: in a HUD layout, a pane wholly in the outer half of a side moves out by
// the bar's width; what is made from it follows by itself (its transform is made from its parent's) and is not moved
// again. Panes across the middle stay, and are looked into. Menus are not touched. The few panes that need telling
// by name are in Named(). Canvas units: 3840 x 2160, the origin in the middle.
// Graphics window: HUD Placement (Screen edges / Centred); OOT_HUD_EDGES=0|1; OOT_HUD_LOG=1 names each pane moved.
namespace hud {
enum Rule : u8 { Plain, Keep, Left, Frozen, Widen, MapIcons };
struct Pane { u64 layout = 0, root = 0; float moved = 0, wide = 1; bool pushed = false, frozen = false, tree = false; Rule rule = Plain; };
struct Root { u64 name = 0; bool hud = false, as_menu = false; };
std::mutex mutex;
std::unordered_map<u64, Pane> panes;
std::unordered_map<u64, Root> roots;
thread_local u64 parent_of_next = 0;
std::atomic<int64_t> force_until{0};

bool Is(u64 guest, const char* text) {
    if (!Readable(guest)) return false;
    for (int i = 0; i < 64; ++i) { const char ch = Load<char>(guest + i); if (ch != text[i]) return false; if (!ch) return true; }
    return false;
}
std::string Text(u64 guest) { std::string t; for (int i = 0; Readable(guest) && i < 48; ++i) { const char ch = Load<char>(guest + i); if (!ch) break; t += ch; } return t; }
u64 LayoutName(u64 layout) { const u64 res = Readable(layout) ? Load<u64>(layout + 0xf8) : 0; return Readable(res) ? Load<u64>(res + 0x18) : 0; }
u64 PaneName(u64 layout, u64 pane) {
    const u64 res = Readable(layout) ? Load<u64>(layout + 0xf8) : 0;
    const u64 names = Readable(res) ? Load<u64>(res + 0x20) : 0, base = Readable(layout) ? Load<u64>(layout + 0x100) : 0;
    return Readable(names) && pane >= base ? Load<u64>(names + (pane - base) / 0xb0 * 8) : 0;
}
// (the hearts, magic, rupees and buttons; the place's name; the minimap; the text box's button guide)
bool HudLayout(u64 name) { for (const char* n : {"PlayerStatus_00", "Location_00", "MapMenu_00", "Talk_00", "MessageLog_01"}) if (Is(name, n)) return true; return false; }
Rule Named(u64 layout_name, u64 pane_name) {
    if (!pane_name) return Plain;
    if (Is(layout_name, "PlayerStatus_00")) {
        // (the left gauges' group is also placed by the pause menu: it stays, its gauges move)
        if (Is(pane_name, "N_ScaleL_00") || Is(pane_name, "N_ScaleL_01")) return Keep;
        if (Is(pane_name, "Pa_HeartGauge_00") || Is(pane_name, "Pa_MpGauge_00") || Is(pane_name, "Pa_AgonyStone_00")) return Left;
    } else if (Is(layout_name, "MapMenu_00")) {
        // (the minimap's layout is the pause menu's map page too: there the icons' group is at the canvas's origin, and
        // nothing of the layout moves. The icons and the world map are placed by code: whole, with what is under them)
        if (Is(pane_name, "N_MapIcon_00")) return MapIcons;
        if (Is(pane_name, "Pa_MapIcon_00") || Is(pane_name, "Pa_WorldMap_00")) return Frozen;
    } else if (Is(layout_name, "Location_00")) {
        for (const char* n : {"P_DecoFrameLT_00", "P_DecoFrameLTSh_00", "P_DecoFrameRT_00", "P_DecoFrameRTSh_00", "P_DecoFrameLB_00", "P_DecoFrameLBSh_00", "P_DecoFrameRB_00", "P_DecoFrameRBSh_00"})
            if (Is(pane_name, n)) return Widen;
    }
    return Plain;
}
// How far a side pane moves out, canvas units (0: nothing to do).
float Out() {
    static const bool allowed = Switch("OOT_HUD_EDGES", true);
    const Wide& w = WideValues();
    if (!allowed || !gfx::Current().hud_edges || !(w.ratio > 16.0f / 9.0f) || !(w.squeeze < 0.995f)) return 0;
    return 1920.0f * (1.0f / w.squeeze - 1.0f);
}
void DirtyBits(Cpu* c, bool store) {
    if (!force_until.load(std::memory_order_relaxed)) return;
    if (input::NowNs() > force_until.load(std::memory_order_relaxed)) { force_until = 0; return; }
    c->x[8] |= 0x40;
    if (store && Readable(c->x[0])) Store<u32>(c->x[0], (u32)c->x[8]);
}
void Placed(u64 pane) {
    const u64 parent = parent_of_next; parent_of_next = 0;
    const float out = Out();
    if (out == 0 || !Readable(pane)) return;
    const u64 layout = Load<u64>(pane + 0x18);
    std::lock_guard lock{mutex};
    Pane up; bool have_parent = false;
    if (parent) {
        const auto it = panes.find(parent);
        if (it == panes.end() || !it->second.tree || !Readable(parent) || Load<u64>(parent + 0x18) != it->second.layout) { panes.erase(pane); return; } // (a menu's pane)
        up = it->second; have_parent = true;
    }
    Pane& p = panes[pane];
    if (p.layout != layout || (!have_parent && !p.root) || (have_parent && p.root != up.root)) {
        const u64 name = LayoutName(layout);
        p = Pane{}; p.layout = layout; p.root = have_parent ? up.root : layout; p.rule = Named(name, PaneName(layout, pane));
        if (!have_parent) { Root& r = roots[layout]; if (r.name != name) { r = Root{}; r.name = name; r.hud = HudLayout(name); } }
    }
    Root& root = roots[p.root];
    p.tree = root.hud;
    if (!p.tree) return;
    const u32 flags = Load<u32>(pane + 0xa8), type = (flags >> 22) & 0xf, anchor = ((flags >> 11) & 0x7f) % 3; // (0 left, 1 centre, 2 right)
    const float x = Load<float>(pane + 0x4c) - up.moved, y = Load<float>(pane + 0x50);
    const float w = Load<float>(pane + 0x58) * std::fabs(Load<float>(pane + 0x28));
    const float x0 = x - (anchor == 1 ? w * 0.5f : anchor == 2 ? w : 0.0f), x1 = x0 + w;
    if (p.rule == MapIcons) {
        const bool menu = std::fabs(x) < 2.0f && std::fabs(y) < 2.0f;
        if (menu != root.as_menu) { root.as_menu = menu; force_until = input::NowNs() + 200'000'000; } // (its panes placed before this one: again)
    }
    float side = 0;
    p.frozen = up.frozen || p.rule == Frozen;
    if (!up.pushed && !p.frozen && !root.as_menu && p.rule != Keep && p.rule != Widen) {
        const bool group = type == 0 || type == 6 || type == 7 || w > 960.0f; // (no picture of its own, or a backdrop: by where it is put)
        if (p.rule == Left) side = -out;
        else if (std::fabs(x) > 2400.0f) side = 0; // (off the canvas: something waiting to slide in)
        else if (group) side = x > 1100.0f ? out : x < -1100.0f ? -out : 0;
        else side = x0 > 700.0f ? out : x1 < -700.0f ? -out : 0;
    }
    p.wide = p.rule == Widen && !root.as_menu ? 1.0f / WideValues().squeeze : 1.0f;
    if (side != 0 && p.moved != up.moved + side) {
        static const bool log = Switch("OOT_HUD_LOG", false); static int shown = 0;
        if (log && shown++ < 400) RT_LOG("hud: %s:%s type %u x %.0f [%.0f %.0f] moves %.0f", Text(LayoutName(layout)).c_str(), Text(PaneName(layout, pane)).c_str(), type, x, x0, x1, side);
    }
    p.moved = up.moved + side; p.pushed = up.pushed || side != 0;
    if (p.moved != up.moved) Store<float>(pane + 0x4c, x + p.moved);
    if (panes.size() > 60000) { panes.clear(); roots.clear(); force_until = input::NowNs() + 200'000'000; } // (menus opened and closed leave theirs behind)
}
void Size(Cpu* c) {
    if (Out() == 0) return;
    float wide = 1;
    { std::lock_guard lock{mutex}; const auto it = panes.find(c->x[0]); if (it == panes.end() || !it->second.tree) return; wide = it->second.wide; }
    if (wide != 1.0f) SetLowFloat(c->v[0], LowFloat(c->v[0]) * wide);
}
} // namespace hud

// ---- the temporal upscalers' camera jitter (sites 62-64) --------------------------------------------------------------
// A temporal upscaler (the vendored chain's DLSS or FSR: nvn/longshot_upscale.cpp) resolves detail finer than a pixel
// from frames whose camera moved by a different fraction of a pixel each. The game fills each view's uniform buffers in
// one function; these sites are where it loads the view's projection rows (the view in x28, its rows at +0x1a8), and
// the frame's offset goes on the third column of rows 0 and 1 in the registers: the GPU's copy of the projection
// alone, so the game's own culling and shadow fitting see the camera as it is. (Longshot's nx2/oot_jitter.py and
// host/rx_backend.c hos_dlss_jitter_clip; the offset and the check that the projection is the picture's are its
// dlss_math.h.) Nothing while no engine draws: the chain's sample is all zero then.
void UpscalerJitter(Cpu* c, int xreg, int yreg) {
    static std::mutex m; static u64 at = ~0ull; static MkwVkJitter sample{};
    const u64 frame = nvn::CurrentFrameNumber();
    MkwVkJitter now;
    { std::lock_guard lock{m}; if (at != frame) { at = frame; sample = mkwvk::temporal_jitter_sample(); } now = sample; }
    if (!now.width || !now.height || !Readable(c->x[28] + 0x1a8)) return;
    float p[16]; std::memcpy(p, reinterpret_cast<const void*>(c->x[28] + 0x1a8), sizeof p);
    float j[2];
    if (!dlss_jitter_clip(p, WideValues().ratio, (unsigned)frame, (int)now.phases, (float)now.width, (float)now.height, j)) return;
    const auto add = [&](int reg, float by) { if (reg < 0) return; float v; std::memcpy(&v, reinterpret_cast<const u8*>(&c->v[reg]) + 8, 4); v += by; std::memcpy(reinterpret_cast<u8*>(&c->v[reg]) + 8, &v, 4); };
    add(xreg, j[0]); add(yreg, j[1]);
}

void WideMovie(Cpu* c) { // (the movie layer's viewport: min x at x29 - 32, max x at x29 - 24)
    const Wide& w = WideValues();
    if (!(w.margin > 0) || !Readable(c->x[29] - 40)) return;
    const u64 at = c->x[29] - 40;
    float a = Load<float>(at + 8), b = Load<float>(at + 16);
    const float in = (b - a) * w.margin; a += in; b -= in;
    Store<float>(at + 8, a); Store<float>(at + 16, b);
}
void WideThumb(Cpu* c) { // (the save's picture, 543 x 306: the middle 16:9 of a wider picture, the middle band of a narrower one)
    const Wide& w = WideValues();
    if (!(w.ratio > 0) || (u32)c->x[1] != 0 || (u32)c->x[2] != 0 || (u32)c->x[3] != 543 || (u32)c->x[4] != 306) return;
    if (w.ratio > 16.0f / 9.0f) { const int wide = (int)(543.0f * w.ratio / (16.0f / 9.0f) + 0.5f); c->x[1] = (u32)((543 - wide) / 2); c->x[3] = (u32)wide; }
    else { const int tall = (int)(306.0f * (16.0f / 9.0f) / w.ratio + 0.5f); c->x[2] = (u32)((306 - tall) / 2); c->x[4] = (u32)tall; }
}

// ---- allocations of the model thread tried again when their heap is full (stability) -----------------------------------
// The game's "OptimizeChangeBank" thread rebuilds a model's material block from small per-model heaps and checks none of
// these allocations: a full heap left a null pointer the next calls wrote through (a crash a few frames after a load).
// A failed one is tried again in the heap's parents. OOT_HEAP_RETRY=0: as shipped.
std::atomic<u32> retries{0};
u64 GuestCall(Cpu* c, u64 fn, u64 a0, u64 a1, u64 a2, u64 lr) { // (every register as before but x0)
    u64 x[32]; std::memcpy(x, c->x, sizeof x); const u32 nzcv = c->nzcv;
    c->x[0] = a0; c->x[1] = a1; c->x[2] = a2; c->x[30] = lr;
    (rt_call)(c, fn);
    const u64 r = c->x[0];
    std::memcpy(c->x, x, sizeof x); c->nzcv = nzcv;
    return r;
}
void CpuAlloc(Cpu* c, u64 target) {
    const u64 heap = c->x[0], size = c->x[1], align = c->x[2], lr = c->x[30];
    (rt_call)(c, target);
    static const bool on = Switch("OOT_HEAP_RETRY", true);
    if (c->x[0] || !on) return;
    u64 h = heap, got = 0;
    for (int up = 0; up < 8 && Readable(h) && !got; ++up) {
        const u64 parent = Load<u64>(h + 0x38);
        if (!Readable(parent) || parent == h) break;
        h = parent;
        const u64 table = Load<u64>(h);
        const u64 try_alloc = Readable(table) ? Load<u64>(table + 0x30) : 0;
        if (!Readable(try_alloc)) continue;
        got = GuestCall(c, try_alloc, h, size, align, lr);
    }
    if (retries++ < 20) RT_LOG("model buffer of %#llx bytes: its heap %#llx was full; %s", (unsigned long long)size, (unsigned long long)heap, got ? "taken from a parent heap" : "no parent had room either");
    c->x[0] = got;
}
void GpuAlloc(Cpu* c, u64 target) {
    const u64 a0 = c->x[0], a1 = c->x[1], a2 = c->x[2], lr = c->x[30];
    (rt_call)(c, target);
    static const bool on = Switch("OOT_HEAP_RETRY", true);
    if ((c->x[0] & 1) || !on || !Readable(a1)) return;
    const u64 heap = Load<u64>(a1 + 8);
    bool ok = false;
    const auto in = [&](u64 other) { Store<u64>(a1 + 8, other); c->x[0] = a0; c->x[1] = a1; c->x[2] = a2; c->x[30] = lr; (rt_call)(c, target); Store<u64>(a1 + 8, heap); return (c->x[0] & 1) != 0; };
    u64 h = heap;
    for (int up = 0; up < 8 && Readable(h) && !ok; ++up) {
        const u64 parent = Load<u64>(h + 0x38);
        if (!Readable(parent) || parent == h) break;
        h = parent;
        ok = in(h);
    }
    if (!ok && heap) ok = in(0); // (the game's default heap)
    if (retries++ < 20) RT_LOG("GPU memory of a model: its heap %#llx was full; %s", (unsigned long long)heap, ok ? "made in a parent heap" : "no heap had room");
}
// ---- the Voice Language option kept after a restart (Longshot issue 430) ------------------------------------------------
// The voice resource applies the option only when the game's data manager lists it as written; option.sav's value
// replaces it without being listed, so a saved choice stood in the Options screen while the voices stayed the text
// language's. The question "was it written?" (x0 the manager, x1 the option's hash; bit 0 of x0 the answer) is also
// answered yes when the option's value is not the one it had at the update before. OOT_VOICE_FOLLOW=0: as shipped.
bool VoiceOption(u64 manager, u32* value) {
    constexpr u32 Option = 0xa2dcdc9du; // (MurmurHash3 of "OptionData.VoiceLanguage")
    if (!Readable(manager)) return false;
    const u64 table = Load<u64>(manager + 0x260); const u32 size = Readable(table) ? Load<u32>(manager + 0x26c) : 0;
    if (!size) return false;
    u32 at = Option % size;
    for (u32 walked = 0; walked < size; ++walked, at = at + 1 < size ? at + 1 : 0) {
        const u32 hash = Load<u32>(table + 8ull * at);
        if (!hash) return false;
        if (hash != Option) continue;
        u32 entry = Load<u32>(table + 8ull * at + 4);
        if (entry >= Load<u32>(manager + 0x250)) entry = 0;
        const u64 entries = Load<u64>(manager + 0x258);
        if (!Readable(entries)) return false;
        *value = Load<u32>(entries + 0x38ull * entry + 0x1c);
        return true;
    }
    return false;
}
void VoiceWritten(Cpu* c, u64 target) {
    const u64 manager = c->x[0]; const bool ours = (u32)c->x[1] == 0xa2dcdc9du;
    (rt_call)(c, target);
    static const bool on = Switch("OOT_VOICE_FOLLOW", true);
    if (!ours || !on) return;
    static u32 seen = 0; static bool have = false; // (the updates run one at a time)
    u32 now = 0;
    if (!VoiceOption(manager, &now)) return;
    if (have && now != seen && !(c->x[0] & 1)) { c->x[0] |= 1; RT_LOG("voice language: the saved option is applied"); }
    seen = now; have = true;
}

// ---- text size (the Game tab's Text Size) -----------------------------------------------------------------------------------
// x19 at site 5: a LetterAnimControl (the text panes that reveal their letters: dialogue, Navi, signs). Its +0x28 is the
// game's pane wrapper, whose +0x28 and +0x38 are the diagonal of the text's 2D scale (0.8 each for dialogue). Each keeps
// the game's own value, taken again whenever the game writes it, and gets that value times the row's scale; at 100% a
// field never scaled is left alone and one scaled before gets the game's value back. (Longshot's host/oot_text_size.c.)
void TextSize(u64 control) {
    const float scale = gfx::Current().text_size / 100.0f;
    const u64 wrapper = Readable(control) ? Load<u64>(control + 0x28) : 0;
    if (!Readable(wrapper) || !Load<u64>(wrapper + 8)) return;
    struct Field { u64 at; u32 game, written, seen; };
    static Field kept[64]; static u32 tick = 0; static std::mutex mutex;
    std::lock_guard lock{mutex};
    ++tick;
    for (const u32 offset : {0x28u, 0x38u}) {
        const u64 at = wrapper + offset;
        const u32 now = Load<u32>(at);
        Field* f = nullptr; Field* oldest = &kept[0];
        for (Field& k : kept) { if (k.at == at) { f = &k; break; } if (k.seen < oldest->seen) oldest = &k; }
        if (!f) { if (scale == 1.0f) continue; f = oldest; f->at = at; f->game = f->written = now; }
        f->seen = tick;
        if (now != f->written) f->game = now; // (the game wrote it since)
        float own; std::memcpy(&own, &f->game, 4);
        const float v = own * scale; std::memcpy(&f->written, &v, 4);
        if (f->written != now) Store<u32>(at, f->written);
    }
}

// ---- above 60 frames a second ----------------------------------------------------------------------------------------------
// The engine steps by 60 / its frame rate (present.cpp writes the measured rate): 0.42 of a frame at 144 fps. Most of the
// game follows the step; these places need whole 60 Hz frames. What each is and why was found by Longshot (nx2/
// oot_patches.py patch_whole_frames, patch_sound_job, patch_bgm_time, patch_bgm_flags, patch_solver_steps,
// patch_turn_rate, patch_movie_frames; host/oot_high_fps.h, frame_whole.h); the code is this runtime's.
namespace highfps {
std::mutex mutex;
float StepNow() { if (!OOT_AT(0x4d41fc8)) return 1.0f; const float s = Load<float>(OOT_AT(0x4d41fc8)); return std::isfinite(s) ? s : 1.0f; } // (the engine's global step)
bool Partial(float step) { return step > 0.0f && step < 0.999f; }

// Counters that add the step (or the step times a whole number) cut to whole frames: (int)0.42 is 0 every frame, and the
// counter never moves: the rupee count above Link's head stays, water rings stand still, the day's clock stops above 120
// fps (and runs slow below), the Scarecrow's Song waits on one note, the Deku Tree's web stays rigid. Each counter keeps
// the part the cut dropped, so it moves at the game's rate at every frame rate. At a step of 1 this changes nothing.
struct Carry { u64 at; float rest; u32 used; };
Carry carries[64]; u32 calls = 0;
u32 Whole(u64 at, float amount) {
    if (!(amount > 0.0f)) return 0;
    if (amount >= 2147483648.0f) return 0x7fffffffu;
    std::lock_guard lock{mutex};
    ++calls;
    Carry* slot = nullptr; Carry* oldest = &carries[0];
    for (Carry& k : carries) { if (k.at == at) { slot = &k; break; } if (k.used < oldest->used) oldest = &k; }
    if (!slot) { slot = oldest; slot->at = at; slot->rest = 0.0f; }
    slot->used = calls;
    const float sum = slot->rest + amount;
    const u32 whole = (u32)sum;
    slot->rest = sum - (float)whole;
    return whole;
}
void WholeSite(Cpu* c, int reg, u64 at) { c->x[reg] = Whole(at, LowFloat(c->v[0])); }

// The physics solver's steps, (int)(step * the steps of a 60 Hz frame + 0.001): rounded up while the step is a part of a
// frame, at least 1, never under the game's own: a solver step then never covers more time than at 60 fps, and Link stays
// on moving platforms.
void SolverSteps(Cpu* c) {
    const float scaled = LowFloat(c->v[0]) - 0.001f, per_frame = LowFloat(c->v[1]);
    if (!(scaled > 0.0f) || !(per_frame > 0.0f) || !(scaled < per_frame) || per_frame > 1000.0f) return;
    const s32 game = (s32)(u32)c->x[1];
    const s32 steps = std::max(1, (s32)std::ceil(scaled - 0.1f));
    if (steps > game) c->x[1] = (u32)steps;
}

// The sound manager's job (its phase in w2; phase 0 hands the step to the sound engine, whose fades, delays and ramps
// count updates of at least one whole step). Fed parts of a step, sounds are not released when an area unloads, the voices
// in use climb until the budget is full, and new sounds, then the music, stop until a restart. While the step is a part
// of a frame the job runs at 60 Hz: phase 0 adds the step up and runs once a whole one is there, with a step of 1; the
// other phases follow that frame's choice.
float part = 0.0f; bool run = true;
u64 flags_at = 0; u8 held = 0, flags_run = 0;
bool SoundGate(Cpu* c) { // true: the job returns at once
    const u32 phase = (u32)c->x[2];
    float step = 1.0f;
    if (phase == 0) {
        const u64 object = OOT_AT(0x4c7e890) ? Load<u64>(OOT_AT(0x4c7e890)) : 0;
        const u64 time = Readable(object) ? Load<u64>(object) : 0;
        if (Readable(time)) step = Load<float>(time + 0x10);
    }
    std::lock_guard lock{mutex};
    if (phase == 0) {
        if (!(step > 0.0f)) { part = 0.0f; run = true; }
        else if (step >= 0.999f) run = true;
        else { part += step; run = part >= 1.0f; if (run) part -= 1.0f; }
        static u64 n = 0, ran = 0; ran += run;
        if (++n % 3600 == 0) { RT_LOG("sound job: ran in %llu of the last 3600 frames (step %.3f)", (unsigned long long)ran, step); ran = 0; }
    }
    // (the music manager's change flags of a frame the job is held: what its update would have read now)
    if (!run && flags_at && Readable(flags_at)) held |= Load<u8>(flags_at);
    return !run;
}
void SoundStep(V128& v) { if (Partial(LowFloat(v))) SetLowFloat(v, 1.0f); }
// The music manager (run by that job, so at 60 Hz too) reads the step and the frame's seconds itself: a whole step and
// that step's seconds while the job is held to whole steps (handed the part, its clock ran at 60 / fps of the real one).
void MusicTime(Cpu* c) {
    const float step = LowFloat(c->v[8]);
    if (!Partial(step)) return;
    SetLowFloat(c->v[9], LowFloat(c->v[9]) / step);
    SetLowFloat(c->v[8], 1.0f);
}
// It also reads a byte of change flags the battle manager sets for one frame; the frames the job was held are kept
// (SoundGate) and read with the next run's: the battle music ends after a fight at every frame rate.
void MusicBegin() { std::lock_guard lock{mutex}; flags_run = held; held = 0; }
void MusicFlags(Cpu* c) { { std::lock_guard lock{mutex}; flags_at = c->x[26] + 2200; } c->x[8] |= flags_run; }

// The steering update clamps the saved turn rate to the turn's size times the step, but the turn is scaled by the step
// already: a rate-limited update keeps rate * step * step, and above 60 fps Gohma circles the ceiling without reaching
// her next point. While the step is a part of a frame both bounds are the turn back as a rate (divided, not multiplied).
// Here s1 = turn * b and s0 = turn * step were just computed (the step in s8).
void TurnRate(Cpu* c) {
    const float step = LowFloat(c->v[8]);
    if (!(step > 0.0f) || !(step < 1.0f)) return;
    const float turn = LowFloat(c->v[0]) / step;
    if (!std::isfinite(turn) || turn == 0.0f) return;
    const float b = LowFloat(c->v[1]) / turn;
    if (std::isfinite(b) && b != 0.0f) SetLowFloat(c->v[1], turn / b);
    SetLowFloat(c->v[0], turn / step);
}

// The movie player counts its updates, and a cutscene with a movie places its narration, rumble and end by that count:
// above 60 fps they ran ahead of the picture. The count grows by the frames' time. (w8: the count plus one.)
void MovieFrames(Cpu* c) { c->x[8] = (u32)((u32)c->x[8] - 1 + Whole(c->x[20] + 760, StepNow())); }

// A character on a moving floor is carried by the floor's velocity; three update modules first test the physics' 1/dt
// against 100 and above it take a tenth of that velocity (meant for slowed time: at 60 fps 1/dt is 60). Above 100 fps
// that fired every frame, and Link stood still on the Shadow Temple's barge while it sailed off under him. While the
// frame's step is a part of one, the test is the game's as it would be at 60 fps: 1/dt times the step. Slowed time still
// takes the tenth. Sites 107-109, at the tenth's block (1/dt in s0): true goes on as if the test had failed.
// (OOT_FPS_LOG=1: how often each of these changed the game's way, every 600 times)
void Fired(const char* what) {
    static const bool on = Switch("OOT_FPS_LOG", false);
    if (!on) return;
    static std::mutex m; static std::unordered_map<std::string, u64> n;
    std::lock_guard lock{m};
    if (++n[what] % 600 == 1) RT_LOG("above 60 fps: %s (%llu times)", what, (unsigned long long)n[what]);
}
bool FloorCarryFull(Cpu* c) {
    const float step = StepNow();
    if (!Partial(step)) return false;
    const float rate = LowFloat(c->v[0]);
    const bool full = std::isfinite(rate) && rate * step <= 100.0f;
    if (full) Fired("a moving floor's full carry kept");
    return full;
}

// A horse's jump sets the blend into its jump animation each frame so the blend ends at takeoff; on a step under 1
// (slow motion at 60 fps) it adds the step to last frame's blend instead. Above 60 fps every step is under 1, the blend
// never ended, and Epona galloped through fences while her body jumped. A step that is the frame's own is not slow
// motion: site 110 (the slow-motion path; the horse's step in s10) goes on to the normal one.
bool HorseWholeStep(Cpu* c) {
    const float frame = StepNow(), step = LowFloat(c->v[10]);
    const bool whole = Partial(frame) && std::fabs(step - frame) <= 0.001f * frame;
    if (whole) Fired("a horse's jump blend at the normal rate");
    return whole;
}

// An animation that loops over one frame or less is one pose (the frozen Zoras under the ice: start 0, end 1). At 60 fps
// every update lands on the same phase; above it the phase moved by a part of a frame each update and the pose twitched.
// Such a loop's clock takes whole frames of the step (the rest kept per entry), and its drawn time is its start.
bool OneFramePose(float start, float end) { return end > start && end - start <= 1.0f; }
struct LoopCarry { u64 at; float rest; };
LoopCarry loop_carries[4096];
// Sites 111, 112: a clock advance, step (s8) times the speed in s1 / s0; x20 the entry (start +16, end +20).
void LoopStep(Cpu* c, int speed_reg) {
    const float step = LowFloat(c->v[8]);
    if (!Partial(StepNow()) || !(step > 0.0f)) return;
    const u64 entry = c->x[20];
    if (!Readable(entry) || !OneFramePose(Load<float>(entry + 16), Load<float>(entry + 20))) return;
    float whole;
    {
        std::lock_guard lock{mutex};
        u32 h = (u32)((entry * 0x9E3779B97F4A7C15ull) >> 40) % 4096, k = 0;
        while (k < 32 && loop_carries[(h + k) % 4096].at && loop_carries[(h + k) % 4096].at != entry) ++k;
        LoopCarry& slot = loop_carries[(h + (k < 32 ? k : 0)) % 4096];
        if (slot.at != entry) { slot.at = entry; slot.rest = 0.0f; }
        const float sum = slot.rest + step;
        whole = std::floor(sum);
        slot.rest = sum - whole;
    }
    V128& speed = c->v[speed_reg];
    SetLowFloat(speed, LowFloat(speed) * (whole / step)); // (step * this = whole * the speed)
    Fired("a one-frame loop's clock in whole frames");
}
// Site 113: the time a skeleton is drawn at, in s0, about to be stored at x22 (its range at +4, +8).
void LoopPose(Cpu* c) {
    if (!Partial(StepNow()) || !Readable(c->x[22])) return;
    const float start = Load<float>(c->x[22] + 4), end = Load<float>(c->x[22] + 8);
    if (OneFramePose(start, end)) { SetLowFloat(c->v[0], start); Fired("a one-frame loop drawn at its start"); }
}

// Pulling up onto a ledge, Link's animation drives his body: each update the game sets a target (his position plus this
// frame's move) and asks the physics for the velocity that reaches it. At 60 fps he lands on it; above 60 the solver
// delivers only part of the move (0.86 at 120 fps), the next target starts from where he ended up, and the shortfall is
// never made up: he slid back off the lip. While the step is a part of a frame Link's target also carries what the physics
// left of the last one (at most three of this frame's moves). A climb starts from the game's own target: the carry needs
// the update before. Site 95 notes Link's body; sites 114, 115 (the two copies of the driven path, the controller in x20,
// its body at +0xf0, the target at +0x3c, the flags at +0x100).
std::atomic<u64> link_body{0};
std::atomic<u32> link_updates{0};
struct DriveCarry { float target[3], game[3]; u64 body; u32 update; bool held; } drive;
void DriveTarget(Cpu* c) {
    const u64 controller = c->x[20];
    if (!Readable(controller)) return;
    const u64 body = Load<u64>(controller + 0xf0);
    if (!body || body != link_body.load(std::memory_order_relaxed) || !Readable(body)) return;
    const float step = StepNow();
    if (!Partial(step) || (Load<u64>(controller + 0x100) & 0x180) != 0x180) return; // (the gap: the next driven call starts again)
    float at[3] = {Load<float>(body + 0x8c), Load<float>(body + 0x9c), Load<float>(body + 0xac)};
    float target[3], game[3];
    std::memcpy(target, reinterpret_cast<const void*>(controller + 0x3c), sizeof target);
    std::memcpy(game, target, sizeof game);
    for (int i = 0; i < 3; ++i) if (!std::isfinite(target[i]) || !std::isfinite(at[i]) || std::fabs(target[i]) > 1e30f || std::fabs(at[i]) > 1e30f) return;
    const u32 update = link_updates.load(std::memory_order_relaxed);
    {
        std::lock_guard lock{mutex};
        const bool same = drive.held && drive.body == body && update - drive.update <= 1u;
        drive.update = update;
        if (same && (!std::memcmp(drive.game, target, sizeof target) || !std::memcmp(drive.target, target, sizeof target))) {
            std::memcpy(target, drive.target, sizeof target); // (run again, or not renewed: the target given)
        } else {
            std::memcpy(drive.game, target, sizeof target);
            if (same) {
                float carry[3], carried = 0.0f, move = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    carry[i] = drive.target[i] - at[i];
                    carried += carry[i] * carry[i];
                    move += (target[i] - at[i]) * (target[i] - at[i]);
                }
                const float scale = carried > 9.0f * move ? 3.0f * std::sqrt(move / carried) : 1.0f;
                for (int i = 0; i < 3 && carried > 0.0f; ++i) target[i] += carry[i] * scale;
            }
            std::memcpy(drive.target, target, sizeof target);
            drive.body = body; drive.held = true;
        }
    }
    if (std::memcmp(game, target, sizeof target)) { std::memcpy(reinterpret_cast<void*>(controller + 0x3c), target, sizeof target); Fired("Link's driven body given its shortfall"); }
}

// The texture data manager asks for a compaction round every frame it has gaps: each round moves textures and makes them
// again (11-22 re-made per 300 frames standing in Castle Town). Site 116 (the request) lets it ask at most once a second.
// OOT_TEX_COMPACT_MS=<ms>: another gap (0: every frame, as the game has it).
bool CompactNotDue() {
    static const long long gap_ms = std::getenv("OOT_TEX_COMPACT_MS") ? std::atoll(std::getenv("OOT_TEX_COMPACT_MS")) : 1000;
    if (gap_ms <= 0) return false;
    static std::atomic<long long> last{0};
    const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    long long was = last.load(std::memory_order_relaxed);
    if (was && now - was < gap_ms) { Fired("a texture compaction request held"); return true; }
    return !last.compare_exchange_strong(was, now);
}
} // namespace highfps
} // namespace

// ---- the neighbouring map preloaded (site 117) -----------------------------------------------------------------------
// The game preloads one neighbouring map on a thread of its own: its preload manager's update picks, each frame, among up
// to 16 neighbours whose radius Link is in, the one he moves towards, else the nearest; asking for another one than the
// one loading or loaded throws that one away and loads the other from the start. In the Market's plaza Link stands in
// Hyrule Field's and Hyrule Castle's radius at once, and walking or turning there flipped the choice back and forth: the
// loader read and set up the two maps over and over, most of a core on a 4-core CPU. A different choice is now taken only
// once it has stayed the choice for OOT_MAP_PRELOAD_HOLD_MS (2000; 0: the game's own at every frame); until then the
// update goes on with the map already asked for. A first choice and the game's direct requests stay the game's.
// x21: the chosen unit (bit 31: none), x19: the manager (its units at +0x408, 0x128 bytes each: +0 in use, +0x10 the
// layout's path; +0x28 bit 0 a direct request, +0x2c its state, +0x70 the current unit's path, +0x1e8 the next one's).
namespace mapload {
std::mutex mutex; int candidate = -2; s64 since_ns = 0;
int Held(u64 mgr) {
    if (Load<u8>(mgr + 0x28) & 1) return -1;
    const u64 next = Load<u64>(mgr + 0x1e8), current = Load<u64>(mgr + 0x70); const u32 state = Load<u32>(mgr + 0x2c);
    u64 want = 0;
    if (Readable(next) && Load<u8>(next)) want = next;
    else if (state >= 1 && state <= 4 && Readable(current) && Load<u8>(current)) want = current;
    if (!want) return -1;
    for (int k = 0; k < 16; ++k) {
        const u64 slot = mgr + 0x408 + 0x128ull * k;
        if (Load<u8>(slot) != 1) continue;
        const u64 path = Load<u64>(slot + 0x10);
        if (!Readable(path)) continue;
        if (std::strncmp(reinterpret_cast<const char*>(path), reinterpret_cast<const char*>(want), 0x100) == 0) return k;
    }
    return -1;
}
void Pick(Cpu* c) {
    static const s64 hold_ns = (std::getenv("OOT_MAP_PRELOAD_HOLD_MS") ? std::atoll(std::getenv("OOT_MAP_PRELOAD_HOLD_MS")) : 2000) * 1000000ll;
    const u64 mgr = c->x[19];
    if (hold_ns <= 0 || !Readable(mgr)) return;
    const int chosen = (u32)c->x[21] >> 31 ? -1 : (int)(u32)c->x[21];
    const int held = Held(mgr);
    std::lock_guard lock{mutex};
    if (held < 0 || chosen == held) { candidate = -2; return; }
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (candidate != chosen) { candidate = chosen; since_ns = now; }
    if (now - since_ns >= hold_ns) { candidate = -2; return; }
    c->x[21] = (c->x[21] & ~0xffffffffull) | (u32)held; // (the held map goes on being the one asked for)
    static u64 n = 0; if (++n % 600 == 1) RT_LOG("map preload: the next map kept while the choice flickers (%llu times)", (unsigned long long)n);
}
} // namespace mapload
// (debug_tools.cpp's player hook: Link's components, at each of his updates)
void oot_note_link(unsigned long long components) {
    highfps::link_updates.fetch_add(1, std::memory_order_relaxed);
    u64 p = components;
    for (u64 off : {0x50ull, 0x20ull, 0x10ull, 8ull, 0x20ull}) p = Readable(p) ? Load<u64>(p + off) : 0;
    if (Readable(p)) highfps::link_body.store(p, std::memory_order_relaxed);
}

extern "C" void oot_site(Cpu* c, int id) {
    switch (id) {
    case 1: ScaleApply(c); break;
    case 2: if (oot_fixed_resolution) c->x[8] &= ~0x18ull; break;
    case 3: ShadowSize(c); break;
    case 4: FieldOfView(c); break;
    case 5: ScaleLowFloat(c, gfx::Current().text_speed); TextSize(c->x[19]); break;
    case 40: case 41: case 42: case 43: ScaleLowFloat(c, gfx::Current().camera_speed); break;
    case 6: case 7: if (const u64 size = WideValues().out) c->x[8] = size; break;
    case 8: if (const u64 size = WideValues().scene) c->x[8] = size; break;
    case 9: WideRatio(c, 0, -1); break;
    case 10: WideRatio(c, 1, 0); break;
    case 11: WideRatio(c, 3, 21); break;
    case 12: WideRatio(c, 3, 0); break;
    case 13: WideRatio(c, 1, 2); break;
    case 14: WideHud(c); break;
    case 15: WideMovie(c); break;
    case 16: WideThumb(c); break;
    case 17: hud::DirtyBits(c, true); break;
    case 18: hud::DirtyBits(c, false); break;
    case 19: hud::parent_of_next = 0; break;
    case 20: hud::parent_of_next = c->x[0]; break;
    case 21: hud::Placed(c->x[19]); break;
    case 22: hud::Size(c); break;
    case 62: UpscalerJitter(c, 19, 20); break;
    case 63: UpscalerJitter(c, 26, -1); break;
    case 64: UpscalerJitter(c, -1, 6); break;
    case 70: highfps::WholeSite(c, 12, c->x[19] + 864); break;
    case 71: highfps::WholeSite(c, 9, c->x[21]); break;
    case 72: highfps::WholeSite(c, 8, c->x[19] + 60); break;
    case 73: case 75: highfps::WholeSite(c, 8, c->x[19] + 0x265fc); break;
    case 74: highfps::WholeSite(c, 8, c->x[9]); break;
    case 76: highfps::WholeSite(c, 22, c->x[19] + 2296); break;
    case 77: case 78: highfps::SolverSteps(c); break;
    case 79: highfps::SoundStep(c->v[0]); break;
    case 80: highfps::MusicTime(c); break;
    case 81: case 82: case 83: case 84: highfps::MusicFlags(c); break;
    case 85: highfps::TurnRate(c); break;
    case 86: highfps::MovieFrames(c); break;
    case 111: highfps::LoopStep(c, 1); break;
    case 112: highfps::LoopStep(c, 0); break;
    case 113: highfps::LoopPose(c); break;
    case 114: case 115: highfps::DriveTarget(c); break;
    case 117: mapload::Pick(c); break;
    case 87: tools::Life(c, true); break;
    case 88: tools::Life(c, false); break;
    case 89: case 90: tools::Speed(c, c->x[0], false); break;
    case 91: tools::Speed(c, c->x[0], true); break;
    case 95: tools::PlayerUpdate(c, c->x[19]); break;
    case 96: tools::FlightPose(c, c->x[19], c->x[25]); break;
    case 97: tools::FlightRotation(c, c->x[0], c->x[1]); break;
    case 98: tools::FreeView(c, c->x[20]); break;
    case 101: case 102: case 103: case 104: case 105: case 106: OutputSite(c, id); break;
    case 100: break; // (the scene buffer's format: for a native HDR picture, not used yet)
    default: break;
    }
}
extern "C" void oot_call(Cpu* c, unsigned long long target, int id) {
    switch (id) {
    case 30: case 31: CpuAlloc(c, target); break;
    case 32: case 33: GpuAlloc(c, target); break;
    case 34: VoiceWritten(c, target); break;
    case 35: OutputHeap(c, target, true); break;
    case 36: OutputHeap(c, target, false); break;
    default: (rt_call)(c, target); break;
    }
}
// ---- the mouse as the camera (Keyboard & Mouse: Mouse Camera at Direct) ----------------------------------------------
// Site 50 is the head of the player camera's update, the camera in x0, once a frame on the game's thread. The camera
// keeps its pitch limits at +0x178 (two floats, degrees), its pitch at +0x18c, its yaw at +0x1a4 and its state at
// +0x204: 3 while it follows Link by itself, 4 while the player turns it (Longshot's findings, host/mouse_native.c).
// The mouse's motion since the last frame is added to angles kept here and those are written into the camera before
// its update reads them, every frame: the update still eases, follows and collides inside the frame, but the game's
// own drift back behind Link never adds up while the mouse is the camera. While the stick (or the camera keys) turns
// the camera, the game's angles become the mouse's new start. In any other state, and while the game takes the right
// stick for something else (hid.cpp), this leaves the motion alone and the mouse is the right stick.
void MouseCameraHook(u64 camera) {
    const input::Config& cfg = input::Current();
    const bool mouse = cfg.mouse_camera == 2, gyro = cfg.gyro == 2; // (the motion sensor set to Always turns these angles too)
    static const bool camera_log = Switch("OOT_CAMERA_LOG", false);
    if ((!mouse && !gyro && !camera_log) || !Readable(camera)) return;
    const float low = Load<float>(camera + 0x178), high = Load<float>(camera + 0x17c);
    const float pitch = Load<float>(camera + 0x18c), yaw = Load<float>(camera + 0x1a4);
    const u32 state = Load<u32>(camera + 0x204);
    static u64 own = 0;
    static float target_yaw = 0, target_pitch = 0;
    static bool have = false;
    const bool camera_like = std::isfinite(low) && std::isfinite(high) && low < high && low >= -90.0f && high <= 90.0f &&
                             std::isfinite(pitch) && std::isfinite(yaw) && std::fabs(yaw) <= 360.0f;
    { static const bool log = Switch("OOT_CAMERA_LOG", false); static int64_t at = 0; // (tests: the follow camera's angles twice a second)
      if (log && camera_like && input::NowNs() - at > 500'000'000) { at = input::NowNs(); RT_LOG("camera: state %u yaw %.1f pitch %.1f", state, yaw, pitch); } }
    if (!camera_like || (state != 3 && state != 4)) { input::camera_live_ns = 0; have = false; return; }
    input::camera_live_ns = input::NowNs();
    if (!mouse && !gyro) return;
    const bool mouse_now = mouse && input::captured.load(std::memory_order_relaxed);
    if (!input::camera_free.load(std::memory_order_relaxed) || (!mouse_now && !gyro)) { have = false; return; }
    if (!have || own != camera || input::stick_turning.load(std::memory_order_relaxed)) { own = camera; target_yaw = yaw; target_pitch = pitch; have = true; }
    float dx = 0, dy = 0, gyaw = 0, gpitch = 0;
    const bool moved = mouse_now && input::TakeMotion(dx, dy) && (dx != 0 || dy != 0);
    const bool turned = gyro && input::TakeGyro(gyaw, gpitch);
    if (!moved && !turned && !mouse_now) { have = false; return; } // (the sensor at rest: the game's camera is its own)
    if (moved || turned) {
        const float k = 0.1f * cfg.mouse_sens / 100.0f; // (degrees a count)
        target_yaw -= (cfg.mouse_invert_x ? -dx : dx) * k;
        target_pitch += (cfg.mouse_invert_y ? -dy : dy) * k;
        const float kg = cfg.gyro_sens / 100.0f; // (the camera turns as far as the controller)
        target_yaw += (cfg.gyro_invert_x ? -gyaw : gyaw) * kg;
        target_pitch += (cfg.gyro_invert_y ? gpitch : -gpitch) * kg;
        while (target_yaw > 180.0f) target_yaw -= 360.0f;
        while (target_yaw < -180.0f) target_yaw += 360.0f;
        target_pitch = std::min(high, std::max(low, target_pitch));
    }
    Store<float>(camera + 0x1a4, target_yaw);
    Store<float>(camera + 0x18c, target_pitch);
}

// ---- the mouse in the game's own menus (sites 60, 61) -----------------------------------------------------------------
// Site 61 is the head of every 2D screen's update (x0 the screen), run each frame on the game's threads; 60 the pause
// menu's handler of a pane getting the focus (x0 the screen, x1 the pane). A screen keeps its buttons in a list (+136
// the head, +144 the first node, +156 an entry's offset from its node, a node's next at +8; entry+24 the button). A
// button has a state byte at +56 (not 0: it has the focus) and at +0x58 its hit box: a 3x4 matrix at +0x28 (x scale
// [0], y scale [4], the translation [9], [10], in the layout's 3840 x 2160 units about the middle, y up) and a
// rectangle x, y, w, h at +0x84. The game's own focus setter is main+0x86ffc0 (screen, button, 0). (All of this is
// Longshot's finding, host/oot_menu.c; the code here is this runtime's.)
// The screen in front - the one first seen last among those running with a focus manager (*(screen+32)+96) and a button
// shown - gets the pointer: the button under it gets the focus as the pad would give it, a left click is A once the
// focus is there, a right click B, the wheel the D-pad's up and down. A click on the pause menu's tab bar steps to
// that page with L or R; on the "Back" prompt it is B. While such a screen runs the window lets the mouse go
// (input::menu_live_ns). Keyboard & Mouse tab: Mouse in Menus. OOT_MENU_LOG=1; OOT_MENU_POINTER_FILE=<file> holding
// "x y [l|r|u|d]" (the picture's -1..1; a press when the line changes) drives it in tests.
namespace menu {
struct Rect { float x0, y0, x1, y1; };
struct Seen { u64 screen = 0; int64_t first = 0, seen = 0, lost = 0; bool buttons = false; unsigned n = 0; };
std::mutex seen_mutex, mouse_mutex;
Seen seen[24];
std::atomic<u64> pause_screen{0};
std::atomic<int> page_now{-1};
bool Log() { static const bool on = Switch("OOT_MENU_LOG", false); return on; }

bool ButtonRect(u64 button, Rect& r) {
    const u64 box = Readable(button) ? Load<u64>(button + 0x58) : 0;
    if (!Readable(box)) return false;
    float m[12], b[4]; std::memcpy(m, reinterpret_cast<void*>(box + 0x28), sizeof m); std::memcpy(b, reinterpret_cast<void*>(box + 0x84), sizeof b);
    const float sx = m[0], sy = m[4], tx = m[9], ty = m[10];
    if (!(sx > 1e-3f) || !(sy > 1e-3f) || !(b[2] > 0) || !(b[3] > 0) || !std::isfinite(tx) || !std::isfinite(ty)) return false;
    r = {tx + sx * b[0], ty + sy * b[1], tx + sx * (b[0] + b[2]), ty + sy * (b[1] + b[3])};
    // (a slot whose hit box covers one side of its picture, the picture's middle at +0xfc: both sides)
    const float s2 = Load<float>(box + 0xd8), cx2 = Load<float>(box + 0xfc), cy2 = Load<float>(box + 0x100);
    if (s2 > 1e-3f && std::isfinite(cx2) && std::isfinite(cy2)) {
        const float w = r.x1 - r.x0, h = r.y1 - r.y0, cx = (r.x0 + r.x1) / 2, cy = (r.y0 + r.y1) / 2;
        if (std::fabs(cx2 - cx) < w && std::fabs(cy2 - cy) < h) {
            r.x0 = std::min(r.x0, 2 * cx2 - r.x1); r.x1 = std::max(r.x1, 2 * cx2 - r.x0);
            r.y0 = std::min(r.y0, 2 * cy2 - r.y1); r.y1 = std::max(r.y1, 2 * cy2 - r.y0);
        }
    }
    return true;
}
template <typename F> int EachButton(u64 screen, F&& f) {
    if (!Readable(screen)) return 0;
    const u64 head = screen + 136; const s32 off = Load<s32>(screen + 156);
    u64 node = Load<u64>(screen + 144); int i = 0;
    for (; Readable(node) && node != head && i < 400; ++i) {
        const u64 entry = node - (u64)(s64)off;
        if (!Readable(entry)) break;
        f(Load<u64>(entry + 24));
        node = Load<u64>(node + 8);
    }
    return i;
}
struct Hit { std::vector<u64> button; std::vector<Rect> rect; u64 hit = 0, focused = 0; };
void Collect(u64 screen, Hit& h) {
    h.button.clear(); h.rect.clear(); h.hit = h.focused = 0;
    EachButton(screen, [&](u64 button) { Rect r; if (!ButtonRect(button, r)) return; h.button.push_back(button); h.rect.push_back(r); if (Load<u8>(button + 56)) h.focused = button; });
    // A text line's mark (a box under 100 x 100 at the line's middle) is taken 1100 wide; boxes one above another grow
    // to meet halfway across a gap under 240, the column's ends by as much: no dead lines between a list's rows.
    for (Rect& r : h.rect) if (r.x1 - r.x0 < 100 && r.y1 - r.y0 < 100) { const float cx = (r.x0 + r.x1) / 2; r.x0 = cx - 550; r.x1 = cx + 550; }
    std::vector<Rect> out = h.rect;
    for (size_t i = 0; i < h.rect.size(); ++i) {
        const Rect& r = h.rect[i]; const float cx = (r.x0 + r.x1) / 2; float above = -1, below = -1;
        for (size_t j = 0; j < h.rect.size(); ++j) {
            const Rect& o = h.rect[j];
            if (j == i || std::fabs((o.x0 + o.x1) / 2 - cx) > 60) continue;
            const float up = o.y0 - r.y1, down = r.y0 - o.y1;
            if (up >= 0 && up < 240 && (above < 0 || up < above)) above = up;
            if (down >= 0 && down < 240 && (below < 0 || down < below)) below = down;
        }
        if (above >= 0) out[i].y1 += above / 2;
        if (below >= 0) out[i].y0 -= below / 2;
        if (above >= 0 && below < 0) out[i].y0 -= above / 2;
        if (below >= 0 && above < 0) out[i].y1 += below / 2;
    }
    h.rect = out;
}
void Find(Hit& h, float x, float y) {
    float best = 0;
    for (size_t i = 0; i < h.rect.size(); ++i) {
        const Rect& r = h.rect[i];
        if (x < r.x0 || x > r.x1 || y < r.y0 || y > r.y1) continue;
        const float area = (r.x1 - r.x0) * (r.y1 - r.y0);
        if (!h.hit || area < best) { h.hit = h.button[i]; best = area; } // (the smallest: a slot inside a panel)
    }
}
bool SetFocus(Cpu* c, u64 screen, u64 button) {
    static const GuestFn fn = rt::LookupFunction(OOT_AT(0x86ffc0));
    if (!fn) { static bool told = false; if (!told) { told = true; RT_LOG("menus: the focus setter is not in this build: the mouse only clicks"); } return false; }
    const Cpu saved = *c;
    c->x[0] = screen; c->x[1] = button; c->x[2] = 0; c->x[30] = 0;
    fn(c);
    const bool taken = c->x[0] & 1;
    *c = saved;
    return taken;
}
// The pause menu's tab bar (its pages' names and the L and R signs are not buttons): the layout's units, as shown.
int TabAt(float x, float y) {
    if (y < 900 || y > 1080) return -1;
    if (x >= -1100 && x < -926) return 4;  // L
    if (x >= -926 && x < 942) return x < -462 ? 0 : x < 6 ? 1 : x < 474 ? 2 : 3;
    if (x >= 942 && x <= 1100) return 5;   // R
    return -1;
}
void Focused(u64 screen, u64 pane) { // (site 60)
    if (!Readable(pane)) return;
    pause_screen = screen;
    const u64 at = (Load<u8>(pane + 65) & 0x20) ? Load<u64>(pane + 32) + 48 : pane + 32;
    const u64 obj = Readable(at) ? Load<u64>(at) : 0, a = Readable(obj) ? Load<u64>(obj + 128) : 0, b = Readable(a) ? Load<u64>(a + 248) : 0, name = Readable(b) ? Load<u64>(b + 24) : 0;
    static const char* const pages[] = {"PaCollectMenu_00", "PaEquipmentMenu_00", "PaItemMenu_00", "PaSystemMenu_00"};
    for (int i = 0; i < 4; ++i) if (hud::Is(name, pages[i])) page_now = i;
}
void Mouse(Cpu* c, u64 screen) {
    std::unique_lock lock{mouse_mutex, std::try_to_lock};
    if (!lock.owns_lock()) return;
    input::menu_live_ns = input::NowNs();
    input::Pointer p = input::PointerNow();
    unsigned pressed = input::TakePointerPresses(); int wheel = input::TakePointerWheel();
    {   // (tests: a pointer from a file)
        static const char* file = std::getenv("OOT_MENU_POINTER_FILE"); static int64_t read_ns = 0; static std::string last; static input::Pointer from_file;
        if (file) {
            const int64_t now = input::NowNs();
            if (now - read_ns > 50'000'000) { read_ns = now;
                if (FILE* f = std::fopen(file, "r")) { char line[64]{}; if (std::fgets(line, sizeof line, f) && last != line) { last = line; char what = 0; float x = 0, y = 0;
                    if (std::sscanf(line, "%f %f %c", &x, &y, &what) >= 2) { from_file = {x, y, true}; if (what == 'l') pressed |= 2; if (what == 'r') pressed |= 8; if (what == 'u') wheel = 1; if (what == 'd') wheel = -1; } } std::fclose(f); } }
            p = from_file;
        }
    }
    static float last_x = 9, last_y = 9; static u64 asked = 0; static int64_t asked_ns = 0;
    static int tab_goal = -1; static int64_t tab_ns = 0; static int tab_steps = 0;
    const int64_t now = input::NowNs();
    if (tab_goal >= 0) { // (a clicked tab: L or R until its page shows)
        const int at = page_now.load();
        if (at == tab_goal || tab_steps >= 4 || at < 0) tab_goal = -1;
        else if (now - tab_ns > 350'000'000) { input::Tap(tab_goal < at ? input::L : input::R, 0, 80); tab_ns = now; ++tab_steps; }
    }
    if (!p.inside) return;
    const Wide& w = WideValues();
    const float x = p.x * 1920.0f / (w.ratio > 0 ? w.qx : 1.0f), y = p.y * 1080.0f / (w.ratio > 0 ? w.qy : 1.0f);
    const bool moved = p.x != last_x || p.y != last_y; last_x = p.x; last_y = p.y;
    static Hit hit; Collect(screen, hit);
    const bool back_prompt = y > -1080 && y < -960 && x >= 1600 && x < 1900;
    if (!back_prompt) Find(hit, x, y);
    bool just = false;
    if (hit.hit && hit.hit != hit.focused && (moved || (pressed & 2)) && !(asked == hit.hit && now - asked_ns < 150'000'000)) {
        const bool taken = SetFocus(c, screen, hit.hit);
        asked = hit.hit; asked_ns = now; just = true;
        if (Log()) RT_LOG("menu: pointer %.0f, %.0f over button %#llx of screen %#llx: focus %s", x, y, (unsigned long long)hit.hit, (unsigned long long)screen, taken ? "set" : "not taken");
    }
    if (wheel) for (int i = 0; i < std::min(std::abs(wheel), 4); ++i) input::Tap(wheel > 0 ? input::Up : input::Down, i * 130, 70);
    if (pressed & 2) { // (the left button)
        if (hit.hit) input::Tap(input::A, just ? 60 : 0, 80);
        else if (back_prompt) input::Tap(input::B, 0, 80);
        else if (screen == pause_screen.load()) {
            const int tab = TabAt(x, y);
            if (tab == 4) input::Tap(input::L, 0, 80); else if (tab == 5) input::Tap(input::R, 0, 80);
            else if (tab >= 0 && tab != page_now.load()) { tab_goal = tab; tab_steps = 0; tab_ns = 0; }
        }
        if (Log()) RT_LOG("menu: click at %.0f, %.0f: %s", x, y, hit.hit ? "the button's A" : back_prompt ? "Back" : "nothing under it");
    }
    if (pressed & 8) input::Tap(input::B, 0, 80); // (the right button: back)
}
void Tick(Cpu* c, u64 screen) { // (site 61)
    if (!input::Current().mouse_menus || !Readable(screen)) return;
    const int64_t now = input::NowNs();
    bool mine = false;
    {
        std::lock_guard lock{seen_mutex};
        Seen* me = nullptr; Seen* oldest = &seen[0];
        for (Seen& s : seen) { if (s.screen == screen) { me = &s; break; } if (s.seen < oldest->seen) oldest = &s; }
        if (!me) { me = oldest; *me = Seen{}; me->screen = screen; me->first = now;
            if (Log()) RT_LOG("menu: screen %#llx (vtable main+%#llx)", (unsigned long long)screen, (unsigned long long)(Load<u64>(screen) - Main)); }
        if (now - me->seen > 200'000'000) me->first = now; // (shown again: in front of what was there)
        me->seen = now;
        if (me->n++ % 6 == 0) { // (ten times a second: does it show a button?)
            const bool had = me->buttons; bool any = false;
            const u64 holder = Load<u64>(screen + 32);
            if (Readable(holder) && Load<u64>(holder + 96)) EachButton(screen, [&](u64 button) { Rect r; if (!any && Load<u8>(button + 56) && ButtonRect(button, r)) any = true; });
            me->buttons = any; if (had && !any) me->lost = now;
        }
        const Seen* front = nullptr;
        for (const Seen& s : seen) if (s.screen && now - s.seen < 100'000'000 && (s.buttons || now - s.lost < 500'000'000) && (!front || s.first > front->first)) front = &s;
        mine = front == me;
    }
    if (mine) Mouse(c, screen);
}
} // namespace menu

// ---- the resource loaders' zstd decode on the host's libzstd ---------------------------------------------------------
// Site 52 is the head of zstd's decompressMultiFrame as the game has it inlined (x0 the game's context, x1 and x2 the
// output and its room, x3 and x4 the compressed frames, x5 and x6 a raw dictionary, x7 the game's DDict or 0): what the
// ARes_Loading threads decode every resource through while an area loads. Recompiled it is a fifth of a core then; the
// host's libzstd does the same decode several times faster, so loads finish sooner and the loaders get out of the
// renderer's way. The game's DDict (+8 its content, +16 the size, +0x6840 the dictID, +0x6844 whether it carries entropy
// tables) is made again as a libzstd one from the same bytes, kept while the game's stays the same. A raw dictionary or
// another frame format runs as recompiled. (The function's interface and the DDict's fields are Longshot's findings,
// host/hos_zstd.c.) OOT_NATIVE_ZSTD=0: off; =2: both decoders run and their outputs are compared (the log counts).
void main_8000eb380(Cpu* c); // (the recompiled decoder, for the comparison)
struct HostDict { u64 guest = 0, content = 0, size = 0; u32 id = 0, entropy = 0; std::vector<u8> head; ZSTD_DDict* dd = nullptr; };
std::mutex zstd_mutex;
std::vector<HostDict> zstd_dicts;
ZSTD_DDict* HostDictFor(u64 guest) {
    const u64 content = Load<u64>(guest + 8), size = Load<u64>(guest + 16);
    const u32 id = Load<u32>(guest + 0x6840), entropy = Load<u32>(guest + 0x6844);
    if (!Readable(content) || !size || size > (64u << 20)) return nullptr;
    const size_t head = (size_t)std::min<u64>(size, 4096);
    std::lock_guard lock{zstd_mutex};
    for (HostDict& d : zstd_dicts)
        if (d.guest == guest && d.content == content && d.size == size && d.id == id && d.entropy == entropy &&
            !std::memcmp(d.head.data(), reinterpret_cast<const void*>(content), head)) return d.dd;
    ZSTD_DDict* dd = ZSTD_createDDict(reinterpret_cast<const void*>(content), (size_t)size); // (a copy: the game may free its own)
    if (!dd || ZSTD_getDictID_fromDDict(dd) != id) { ZSTD_freeDDict(dd); return nullptr; }
    if (zstd_dicts.size() >= 32) { ZSTD_freeDDict(zstd_dicts.front().dd); zstd_dicts.erase(zstd_dicts.begin()); }
    HostDict d{guest, content, size, id, entropy, {}, dd};
    d.head.assign(reinterpret_cast<const u8*>(content), reinterpret_cast<const u8*>(content) + head);
    zstd_dicts.push_back(std::move(d));
    RT_LOG("zstd: the game's dictionary at %#llx (%llu bytes, id %u) decodes on the host's libzstd %s", (unsigned long long)guest, (unsigned long long)size, id, ZSTD_versionString());
    return dd;
}
bool ZstdFrames(Cpu* c) {
    static const int mode = std::getenv("OOT_NATIVE_ZSTD") ? std::atoi(std::getenv("OOT_NATIVE_ZSTD")) : 1;
    if (!mode) return false;
    const u64 context = c->x[0], dst = c->x[1], room = c->x[2], src = c->x[3], size = c->x[4], ddict = c->x[7];
    if (c->x[5] || c->x[6] || !Readable(context) || Load<u32>(context + 0x7110) || !Readable(dst) || !Readable(src)) return false;
    ZSTD_DDict* dd = nullptr;
    if (ddict && (!Readable(ddict) || !(dd = HostDictFor(ddict)))) return false;
    thread_local ZSTD_DCtx* z = ZSTD_createDCtx();
    if (!z) return false;
    static std::atomic<u64> calls{0}, bytes{0}, nanos{0}, checked{0}, differed{0};
    if (mode == 2) {
        // Both decoders: libzstd into a buffer of its own, then the recompiled function into the game's, compared at once.
        thread_local bool inside = false;
        if (inside) return false;
        thread_local std::vector<u8> expected;
        expected.resize(room);
        const size_t r = dd ? ZSTD_decompress_usingDDict(z, expected.data(), room, reinterpret_cast<const void*>(src), size, dd)
                            : ZSTD_decompressDCtx(z, expected.data(), room, reinterpret_cast<const void*>(src), size);
        inside = true; main_8000eb380(c); inside = false;
        const u64 game = c->x[0];
        ++checked;
        const bool same = ZSTD_isError(r) ? ZSTD_isError((size_t)game) : (game == r && !std::memcmp(reinterpret_cast<const void*>(dst), expected.data(), r));
        if (!same && ++differed <= 8) RT_LOG("zstd check: a decode of %llu bytes differs (the game's result %#llx, libzstd's %#zx)", (unsigned long long)size, (unsigned long long)game, r);
        if ((++calls & 1023) == 0) RT_LOG("zstd check: %llu decodes compared, %llu differed", (unsigned long long)checked.load(), (unsigned long long)differed.load());
        return true;
    }
    const auto begin = std::chrono::steady_clock::now();
    const size_t r = dd ? ZSTD_decompress_usingDDict(z, reinterpret_cast<void*>(dst), room, reinterpret_cast<const void*>(src), size, dd)
                        : ZSTD_decompressDCtx(z, reinterpret_cast<void*>(dst), room, reinterpret_cast<const void*>(src), size);
    nanos += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
    if (!ZSTD_isError(r)) bytes += r;
    if ((++calls & 4095) == 0) RT_LOG("zstd: %llu resources (%llu MiB) decoded on the host in %.2f s", (unsigned long long)calls.load(), (unsigned long long)(bytes.load() >> 20), nanos.load() / 1e9);
    c->x[0] = r;
    return true;
}

// ---- the new-game opening (Game: Skip Opening) -------------------------------------------------------------------------
// The opening is two event requests (site 53, a one-shot event request: the event's name is the pointer at the request
// argument + 6800), "Opening_Introduction" and "Opening_LoadingKokiriForest". Their flowcharts set the save's state on
// both of their paths and start two movie timelines, "Dm_Kokiri_0030" and "Dm_Kokiri_0000" (site 54, the timelines'
// only starter: x0 the timeline player, x1 a pointer to the name's pointer); the introduction then asks whether it was
// skipped (site 55) and takes its short path if so. With the setting on the requests are still sent, so the game sets
// everything it would; the two timelines are not started (bit 0 of the player + 0xf25 cleared makes the caller finish
// its node), and the skip query is answered yes while one of the two requests is the current one. Any other event
// request ends that, as does a minute passing. (Longshot's findings and method, host/hos_main.c hos_oot_opening.)
bool Opening(Cpu* c, int what) {
    static std::atomic<s64> armed_until{0};
    if (!gfx::Current().skip_opening) { armed_until = 0; return false; }
    const auto name_at = [](u64 p) { std::string name; if (Readable(p)) for (int i = 0; i < 79; ++i) { const char ch = Load<char>(p + i); if (!ch) break; name += ch; } return name; };
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (what == 0) {
        const u64 argument = c->x[1];
        const std::string name = Readable(argument) ? name_at(Load<u64>(argument + 6800)) : std::string();
        if (name == "Opening_Introduction" || name == "Opening_LoadingKokiriForest") {
            if (armed_until.exchange(now + 60'000) < now) RT_LOG("opening skip: %s requested", name.c_str());
        } else if (!name.empty()) armed_until = 0;
        return false; // (the request is still sent)
    }
    if (armed_until.load() < now) return false;
    if (what == 2) { c->x[0] = 1; return true; }
    const std::string name = Readable(c->x[1]) ? name_at(Load<u64>(c->x[1])) : std::string();
    if (name != "Dm_Kokiri_0000" && name != "Dm_Kokiri_0030") return false;
    if (Readable(c->x[0])) Store<u8>(c->x[0] + 0xf25, Load<u8>(c->x[0] + 0xf25) & ~1u);
    RT_LOG("opening skip: %s not started", name.c_str());
    return true;
}

extern "C" bool oot_fn(Cpu* c, int id) {
    switch (id) {
    case 53: return Opening(c, 0);
    case 54: return Opening(c, 1);
    case 55: return Opening(c, 2);
    case 52: return ZstdFrames(c);
    case 50: MouseCameraHook(c->x[0]); return false;
    case 60: menu::Focused(c->x[0], c->x[1]); return false;
    case 61: menu::Tick(c, c->x[0]); return false;
    case 92: tools::Speed(c, c->x[0], true); return false;
    case 99: return tools::SaveRequest(c);
    case 93: return highfps::SoundGate(c);
    case 94: highfps::MusicBegin(); return false;
    case 107: case 108: case 109: return highfps::FloorCarryFull(c);
    case 110: return highfps::HorseWholeStep(c);
    case 116: return highfps::CompactNotDue();
    // Site 56: the camera's turn back behind Link as he walks (its follow state's movement path) is left out when the
    // Auto-centre Camera row is Off: x25 gets the parameter base that path would restore, and the game goes on where
    // the camera's other yaw paths meet (Longshot's patch_camera_centering). Pitch, collision and Z-targeting stay.
    case 56: if (!input::Current().no_centering) return false; c->x[25] = c->x[26]; return true;
    default: return false;
    }
}
