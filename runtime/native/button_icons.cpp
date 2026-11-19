// The game's on-screen A, B, X and Y icons drawn for the player's controller (Controls > Button Icons).
//
// The game draws button icons as characters of its own icon font (Font2/Scalable/External...bfotf.zs: a CFF
// OpenType font, zstd-packed with the game's dictionary and its words XORed with a key). When the font is opened
// it is handed over changed in memory: each of the four face-button glyphs is made to show the label of the
// controller button that presses that game button - the letters moved between the four glyphs (Xbox and Nintendo
// pads, any Button Layout), or the PlayStation shapes drawn here inside the font's own button circle. Nothing of
// the game's is stored or shipped: the change is made from the player's game files at each start.
#include "../native.h"
#include "input_settings.h"
#include <SDL3/SDL.h>
#include <zstd.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

bool ReadRomFile(const std::string& path, std::vector<unsigned char>& out);

namespace {
using Bytes = std::vector<u8>;

u32 Be32(const u8* p) { return (u32)p[0] << 24 | (u32)p[1] << 16 | (u32)p[2] << 8 | p[3]; }
u16 Be16(const u8* p) { return (u16)(p[0] << 8 | p[1]); }
void Put32(u8* p, u32 v) { p[0] = (u8)(v >> 24); p[1] = (u8)(v >> 16); p[2] = (u8)(v >> 8); p[3] = (u8)v; }

// The dictionary the game's .zs files are packed with (Pack/ZsDic.pack.zs: a zstd-packed SARC holding zs.zsdic).
bool Dictionary(Bytes& out) {
    Bytes packed;
    if (!ReadRomFile("/Pack/ZsDic.pack.zs", packed)) return false;
    const u64 size = ZSTD_getFrameContentSize(packed.data(), packed.size());
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN || size > (64u << 20)) return false;
    Bytes sarc(size);
    if (ZSTD_isError(ZSTD_decompress(sarc.data(), sarc.size(), packed.data(), packed.size()))) return false;
    if (sarc.size() < 0x20 || std::memcmp(sarc.data(), "SARC", 4) != 0) return false;
    auto le32 = [&](u64 at) { u32 v; std::memcpy(&v, sarc.data() + at, 4); return v; };
    auto le16 = [&](u64 at) { u16 v; std::memcpy(&v, sarc.data() + at, 2); return v; };
    const u32 data = le32(0x0c);
    const u64 sfat = 0x14;
    if (std::memcmp(sarc.data() + sfat, "SFAT", 4) != 0) return false;
    const u32 count = le16(sfat + 6);
    const u64 names = sfat + 12 + 16ull * count + 8;
    for (u32 i = 0; i < count; ++i) {
        const u64 node = sfat + 12 + 16ull * i;
        if (node + 16 > sarc.size()) return false;
        const u64 name = names + (u64)(le32(node + 4) & 0xffff) * 4;
        const u32 begin = le32(node + 8), end = le32(node + 12);
        if (name >= sarc.size() || (u64)data + end > sarc.size() || begin > end) return false;
        if (std::strcmp(reinterpret_cast<const char*>(sarc.data() + name), "zs.zsdic") == 0) {
            out.assign(sarc.begin() + data + begin, sarc.begin() + data + end);
            return true;
        }
    }
    return false;
}

// ---- Type 2 charstrings ----
void Number(Bytes& s, int v) {
    if (v >= -107 && v <= 107) s.push_back((u8)(v + 139));
    else if (v >= 108 && v <= 1131) { v -= 108; s.push_back((u8)(v / 256 + 247)); s.push_back((u8)(v % 256)); }
    else if (v >= -1131 && v <= -108) { v = -v - 108; s.push_back((u8)(v / 256 + 251)); s.push_back((u8)(v % 256)); }
    else { s.push_back(28); s.push_back((u8)((v >> 8) & 0xff)); s.push_back((u8)(v & 0xff)); }
}
struct Pen {
    Bytes s; int x = 0, y = 0;
    void Move(double ax, double ay) { const int nx = (int)std::lround(ax), ny = (int)std::lround(ay); Number(s, nx - x); Number(s, ny - y); s.push_back(21); x = nx; y = ny; }
    void Line(double ax, double ay) { const int nx = (int)std::lround(ax), ny = (int)std::lround(ay); Number(s, nx - x); Number(s, ny - y); s.push_back(5); x = nx; y = ny; }
    void Curve(double x1, double y1, double x2, double y2, double x3, double y3) {
        const int p[6] = {(int)std::lround(x1), (int)std::lround(y1), (int)std::lround(x2), (int)std::lround(y2), (int)std::lround(x3), (int)std::lround(y3)};
        Number(s, p[0] - x); Number(s, p[1] - y); Number(s, p[2] - p[0]); Number(s, p[3] - p[1]); Number(s, p[4] - p[2]); Number(s, p[5] - p[3]);
        s.push_back(8); x = p[4]; y = p[5];
    }
    // A closed polygon (contours close themselves at the next moveto or endchar).
    void Polygon(const std::vector<std::array<double, 2>>& points) { Move(points[0][0], points[0][1]); for (size_t i = 1; i < points.size(); ++i) Line(points[i][0], points[i][1]); }
    // A circle: clockwise cuts out of the filled button, counter-clockwise fills again.
    void Circle(double cx, double cy, double r, bool clockwise) {
        const double k = 0.5523 * r, d = clockwise ? -1 : 1;
        Move(cx, cy + r);
        Curve(cx - d * k, cy + r, cx - d * r, cy + k, cx - d * r, cy);
        Curve(cx - d * r, cy - k, cx - d * k, cy - r, cx, cy - r);
        Curve(cx + d * k, cy - r, cx + d * r, cy - k, cx + d * r, cy);
        Curve(cx + d * r, cy + k, cx + d * k, cy + r, cx, cy + r);
    }
};

// One token of a charstring: a number or an operator.
struct Token { bool op; int value; };
bool Tokens(const u8* p, size_t n, std::vector<Token>& out) {
    for (size_t i = 0; i < n;) {
        const u8 b = p[i];
        if (b == 28) { if (i + 3 > n) return false; out.push_back({false, (int)(int16_t)Be16(p + i + 1)}); i += 3; }
        else if (b == 255) { if (i + 5 > n) return false; out.push_back({false, (int)Be32(p + i + 1) >> 16}); i += 5; }
        else if (b >= 32 && b <= 246) { out.push_back({false, b - 139}); i += 1; }
        else if (b >= 247 && b <= 250) { if (i + 2 > n) return false; out.push_back({false, (b - 247) * 256 + p[i + 1] + 108}); i += 2; }
        else if (b >= 251 && b <= 254) { if (i + 2 > n) return false; out.push_back({false, -(b - 251) * 256 - p[i + 1] - 108}); i += 2; }
        else if (b == 12) { if (i + 2 > n) return false; out.push_back({true, 1200 + p[i + 1]}); i += 2; }
        else if (b == 1 || b == 3 || b == 18 || b == 23 || b == 19 || b == 20) return false; // (hints: not in these glyphs)
        else { out.push_back({true, b}); i += 1; }
    }
    return true;
}

// The face-button glyph's frame, read from the font's A: its width operand, where the subroutine that draws the
// button circle (and ends the glyph) starts, and that subroutine's number.
struct Button { int width = 0, subr = 0; double cx = 0, cy = 0, r = 0; };
bool ReadButton(const u8* glyph, size_t n, int lsb, Button& b) {
    std::vector<Token> t;
    if (!Tokens(glyph, n, t) || t.size() < 5) return false;
    if (t.back().op != true || t.back().value != 10) return false; // callsubr
    std::vector<int> stack;
    int x = 0, y = 0;
    bool first = true;
    for (size_t i = 0; i + 1 < t.size(); ++i) {
        if (!t[i].op) { stack.push_back(t[i].value); continue; }
        const int op = t[i].value;
        if (first) { // (the width comes first, ahead of the first operator's own operands)
            const size_t expected = op == 21 ? 2 : (op == 22 || op == 4) ? 1 : 99;
            if (stack.size() != expected + 1) return false;
            b.width = stack[0]; stack.erase(stack.begin()); first = false;
        }
        if (op == 21 && stack.size() == 2) { x += stack[0]; y += stack[1]; }
        else if (op == 22 && stack.size() == 1) x += stack[0];
        else if (op == 4 && stack.size() == 1) y += stack[0];
        else if (op == 5 && stack.size() % 2 == 0) for (size_t k = 0; k < stack.size(); k += 2) { x += stack[k]; y += stack[k + 1]; }
        else if (op == 6 || op == 7) { bool h = op == 6; for (int v : stack) { (h ? x : y) += v; h = !h; } }
        else return false;
        stack.clear();
    }
    if (stack.size() != 3) return false;
    b.subr = stack[2];
    const double top_x = x + stack[0], top_y = y + stack[1]; // (the circle is drawn from its top)
    b.r = top_x - lsb; b.cx = top_x; b.cy = top_y - b.r;
    return b.r > 50 && b.r < 2000;
}

// The PlayStation shapes: 0 cross, 1 circle, 2 square, 3 triangle, cut out of the font's button.
Bytes PlayStationGlyph(const Button& b, int shape) {
    Pen p;
    Number(p.s, b.width);
    const double cx = b.cx, cy = b.cy, r = b.r, stroke = 0.17 * r;
    if (shape == 0) {
        const double a = 0.50 * r, w = stroke / 2;
        // a plus, clockwise, turned 45 degrees
        const std::vector<std::array<double, 2>> plus = {{w, a}, {w, w}, {a, w}, {a, -w}, {w, -w}, {w, -a}, {-w, -a}, {-w, -w}, {-a, -w}, {-a, w}, {-w, w}, {-w, a}};
        std::vector<std::array<double, 2>> cross;
        const double c = std::sqrt(0.5);
        for (const auto& q : plus) cross.push_back({cx + (q[0] + q[1]) * c, cy + (q[1] - q[0]) * c});
        p.Polygon(cross);
    } else if (shape == 1) {
        p.Circle(cx, cy, 0.44 * r, true);
        p.Circle(cx, cy, 0.44 * r - stroke, false);
    } else if (shape == 2) {
        const double o = 0.36 * r, i = o - stroke;
        p.Polygon({{cx - o, cy + o}, {cx + o, cy + o}, {cx + o, cy - o}, {cx - o, cy - o}});
        p.Polygon({{cx - i, cy + i}, {cx - i, cy - i}, {cx + i, cy - i}, {cx + i, cy + i}});
    } else {
        const double o = 0.50 * r, i = o - 2 * stroke, down = 0.06 * r;
        auto corner = [&](double radius, int k) { const double pi = 3.14159265358979, a = pi / 2 + k * 2 * pi / 3; return std::array<double, 2>{cx + radius * std::cos(a), cy - down + radius * std::sin(a)}; };
        p.Polygon({corner(o, 0), corner(o, 2), corner(o, 1)}); // clockwise
        p.Polygon({corner(i, 0), corner(i, 1), corner(i, 2)});
    }
    // to the circle's top, then the font's own circle subroutine (which ends the glyph)
    Number(p.s, (int)std::lround(cx) - p.x); Number(p.s, (int)std::lround(cy + r) - p.y); Number(p.s, b.subr); p.s.push_back(10);
    return p.s;
}

// A CFF INDEX at `at`: its item boundaries (absolute offsets into the table) and its end.
bool ReadIndex(const Bytes& cff, size_t at, std::vector<size_t>& bounds, size_t& end) {
    if (at + 2 > cff.size()) return false;
    const u32 count = Be16(&cff[at]);
    if (!count) { bounds.clear(); end = at + 2; return true; }
    if (at + 3 > cff.size()) return false;
    const u32 off_size = cff[at + 2];
    if (off_size < 1 || off_size > 4) return false;
    const size_t offsets = at + 3, data = offsets + (size_t)(count + 1) * off_size - 1;
    if (data >= cff.size()) return false;
    bounds.resize(count + 1);
    for (u32 i = 0; i <= count; ++i) {
        u32 v = 0;
        for (u32 k = 0; k < off_size; ++k) v = v << 8 | cff[offsets + (size_t)i * off_size + k];
        bounds[i] = data + v;
        if (bounds[i] > cff.size()) return false;
    }
    end = bounds[count];
    return true;
}

// The glyph of a code point (cmap format 4, Windows Unicode).
int GlyphOf(const u8* cmap, size_t size, u32 code) {
    if (size < 4) return -1;
    const u32 tables = Be16(cmap + 2);
    for (u32 i = 0; i < tables && 4 + 8 * (i + 1) <= size; ++i) {
        const u8* rec = cmap + 4 + 8 * i;
        if (Be16(rec) != 3 || (Be16(rec + 2) != 1 && Be16(rec + 2) != 10)) continue;
        const u32 at = Be32(rec + 4);
        if (at + 14 > size || Be16(cmap + at) != 4) continue;
        const u8* t = cmap + at;
        const u32 segs = Be16(t + 6) / 2;
        if (at + 16 + 8ull * segs > size) continue;
        const u8 *ends = t + 14, *starts = ends + 2 * segs + 2, *deltas = starts + 2 * segs, *ranges = deltas + 2 * segs;
        for (u32 s = 0; s < segs; ++s) {
            if (code > Be16(ends + 2 * s)) continue;
            if (code < Be16(starts + 2 * s)) break;
            const u32 range = Be16(ranges + 2 * s);
            if (!range) return (int)((code + Be16(deltas + 2 * s)) & 0xffff);
            const u8* g = ranges + 2 * s + range + 2 * (code - Be16(starts + 2 * s));
            if (g + 2 > cmap + size) return -1;
            const u32 id = Be16(g);
            return id ? (int)((id + Be16(deltas + 2 * s)) & 0xffff) : 0;
        }
    }
    return -1;
}

// Which label each game button shows: for each of A, B, X, Y, the icon: 0-3 the font's own A, B, X, Y,
// 4-7 the PlayStation cross, circle, square, triangle.
std::array<int, 4> Labels(int style) {
    const input::Config& c = input::Current();
    std::array<int, 4> out{0, 1, 2, 3};
    for (int g = 0; g < 4; ++g) {
        const int code = c.pad[g][0] - 1; // an SDL gamepad button
        int label;
        switch (code) {
        case SDL_GAMEPAD_BUTTON_SOUTH: label = style == 1 ? 1 : style == 2 ? 0 : 4; break; // B / A / cross
        case SDL_GAMEPAD_BUTTON_EAST: label = style == 1 ? 0 : style == 2 ? 1 : 5; break;  // A / B / circle
        case SDL_GAMEPAD_BUTTON_WEST: label = style == 1 ? 3 : style == 2 ? 2 : 6; break;  // Y / X / square
        case SDL_GAMEPAD_BUTTON_NORTH: label = style == 1 ? 2 : style == 2 ? 3 : 7; break; // X / Y / triangle
        default: label = g; // (bound to something else: the game's own icon)
        }
        out[g] = label;
    }
    return out;
}

int AutomaticStyle() {
    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    int style = 0;
    for (int i = 0; ids && i < count && !style; ++i) {
        switch (SDL_GetGamepadTypeForID(ids[i])) {
        case SDL_GAMEPAD_TYPE_PS3: case SDL_GAMEPAD_TYPE_PS4: case SDL_GAMEPAD_TYPE_PS5: style = 3; break;
        case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO: case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
        case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT: case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR: style = 1; break;
        default: style = 2;
        }
    }
    SDL_free(ids);
    return style;
}
}

// The icon font changed for the controller, or false to use the game's file as it is.
bool ButtonIconFont(const std::string& path, Bytes& out) {
    if (path.find("Font2/Scalable/External.") == std::string::npos || path.size() < 9 || path.compare(path.size() - 9, 9, ".bfotf.zs") != 0) return false;
    int style = input::Current().icons; // 0 automatic, 1 Nintendo, 2 Xbox, 3 PlayStation
    if (const char* e = std::getenv("OOT_BUTTON_ICONS")) style = std::atoi(e); // (tests)
    if (style == 0) style = AutomaticStyle();
    if (style == 0) style = 1; // (no controller at the start: the keyboard - the game's own icons)
    const std::array<int, 4> labels = Labels(style);
    if (labels == std::array<int, 4>{0, 1, 2, 3}) { RT_LOG("button icons: the game's own (style %d)", style); return false; }

    auto fail = [&](const char* why) { RT_LOG("button icons: left as they are (%s)", why); return false; };
    Bytes packed, dictionary;
    if (!ReadRomFile(path, packed)) return fail("font not read");
    if (!Dictionary(dictionary)) return fail("no dictionary");
    const u64 size = ZSTD_getFrameContentSize(packed.data(), packed.size());
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN || size < 64 || size > (16u << 20)) return fail("font size");
    Bytes file(size);
    {
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        const size_t got = ZSTD_decompress_usingDict(dctx, file.data(), file.size(), packed.data(), packed.size(), dictionary.data(), dictionary.size());
        ZSTD_freeDCtx(dctx);
        if (ZSTD_isError(got) || got != file.size()) return fail("font not unpacked");
    }
    // the words after the 8-byte header are XORed with a key; the font inside starts with "OTTO"
    u8 key[4];
    for (int i = 0; i < 4; ++i) key[i] = file[8 + i] ^ (u8)"OTTO"[i];
    Bytes otf(file.begin() + 8, file.end());
    for (size_t i = 0; i < otf.size(); ++i) otf[i] ^= key[i % 4];

    const u32 tables = Be16(&otf[4]);
    size_t cff_at = 0, cff_size = 0, cmap_at = 0, cmap_size = 0, hmtx_at = 0, hmtx_size = 0;
    for (u32 i = 0; i < tables && 12 + 16 * (i + 1) <= otf.size(); ++i) {
        const u8* rec = &otf[12 + 16 * i];
        const size_t at = Be32(rec + 8), len = Be32(rec + 12);
        if (at + len > otf.size()) return fail("table outside the font");
        if (!std::memcmp(rec, "CFF ", 4)) { cff_at = at; cff_size = len; }
        if (!std::memcmp(rec, "cmap", 4)) { cmap_at = at; cmap_size = len; }
        if (!std::memcmp(rec, "hmtx", 4)) { hmtx_at = at; hmtx_size = len; }
    }
    if (!cff_size || !cmap_size || !hmtx_size) return fail("tables missing");
    Bytes cff(otf.begin() + cff_at, otf.begin() + cff_at + cff_size);

    // the Top DICT: where the CharStrings INDEX and the Private DICT are
    std::vector<size_t> bounds;
    size_t end = 0;
    if (cff.size() < 4 || !ReadIndex(cff, cff[2], bounds, end)) return fail("CFF names");
    std::vector<size_t> top;
    if (!ReadIndex(cff, end, top, end) || top.size() != 2) return fail("CFF top");
    size_t charstrings_operand = 0, private_operand = 0;
    int charstrings = -1, private_offset = -1;
    {
        std::vector<std::pair<size_t, int>> operands; // (where, value)
        for (size_t i = top[0]; i < top[1];) {
            const u8 b = cff[i];
            if (b == 28) { operands.push_back({i, (int)(int16_t)Be16(&cff[i + 1])}); i += 3; }
            else if (b == 29) { operands.push_back({i, (int)Be32(&cff[i + 1])}); i += 5; }
            else if (b == 30) { ++i; while (i < top[1] && (cff[i] & 0x0f) != 0x0f && (cff[i] & 0xf0) != 0xf0) ++i; ++i; operands.push_back({0, 0}); }
            else if (b >= 32 && b <= 246) { operands.push_back({i, b - 139}); ++i; }
            else if (b >= 247 && b <= 250) { operands.push_back({i, (b - 247) * 256 + cff[i + 1] + 108}); i += 2; }
            else if (b >= 251 && b <= 254) { operands.push_back({i, -(b - 251) * 256 - cff[i + 1] - 108}); i += 2; }
            else {
                int op = b; if (b == 12) { op = 1200 + cff[i + 1]; ++i; } ++i;
                if (op == 17 && operands.size() == 1) { charstrings_operand = operands[0].first; charstrings = operands[0].second; }
                if (op == 18 && operands.size() == 2) { private_operand = operands[1].first; private_offset = operands[1].second; }
                operands.clear();
            }
        }
    }
    if (charstrings <= 0 || private_offset <= 0) return fail("no CharStrings or Private");
    if (cff[private_operand] != 28 && cff[private_operand] != 29) return fail("Private offset written short");
    std::vector<size_t> glyphs;
    size_t glyphs_end = 0;
    if (!ReadIndex(cff, (size_t)charstrings, glyphs, glyphs_end)) return fail("CharStrings");
    if ((size_t)private_offset < glyphs_end) return fail("Private DICT before the glyphs");

    int ids[4];
    for (int k = 0; k < 4; ++k) {
        ids[k] = GlyphOf(&otf[cmap_at], cmap_size, 0xE040 + k);
        if (ids[k] <= 0 || (size_t)ids[k] + 1 >= glyphs.size()) return fail("face-button glyph missing");
        for (int j = 0; j < k; ++j) if (ids[j] == ids[k]) return fail("face-button glyphs shared");
    }
    std::array<Bytes, 4> own, made;
    for (int k = 0; k < 4; ++k) own[k].assign(cff.begin() + glyphs[ids[k]], cff.begin() + glyphs[ids[k] + 1]);
    Button button;
    {
        const size_t at = hmtx_at + 4ull * ids[0];
        if (at + 4 > hmtx_at + hmtx_size) return fail("hmtx");
        const int lsb = (int16_t)Be16(&otf[at + 2]);
        if (labels[0] >= 4 || labels[1] >= 4 || labels[2] >= 4 || labels[3] >= 4)
            if (!ReadButton(own[0].data(), own[0].size(), lsb, button)) return fail("button circle not found");
    }
    for (int k = 0; k < 4; ++k) made[k] = labels[k] < 4 ? own[labels[k]] : PlayStationGlyph(button, labels[k] - 4);

    // a new CharStrings INDEX in place of the old, with the same offset size; what follows moves by the difference.
    // The font must keep its size (the game reads it into a buffer of the original's size: a larger one crashed it at
    // start), so the table is padded back to its length and anything that would make it longer is refused.
    const u32 count = (u32)glyphs.size() - 1, off_size = cff[(size_t)charstrings + 2];
    Bytes body;
    std::vector<u32> offsets{1};
    for (u32 g = 0; g < count; ++g) {
        int k = -1;
        for (int j = 0; j < 4; ++j) if ((u32)ids[j] == g) k = j;
        if (k >= 0) body.insert(body.end(), made[k].begin(), made[k].end());
        else body.insert(body.end(), cff.begin() + glyphs[g], cff.begin() + glyphs[g + 1]);
        offsets.push_back((u32)body.size() + 1);
    }
    if (off_size < 4 && offsets.back() >= (1u << (8 * off_size))) return fail("glyphs too large");
    Bytes index{(u8)(count >> 8), (u8)count, (u8)off_size};
    for (u32 o : offsets) for (int k = (int)off_size - 1; k >= 0; --k) index.push_back((u8)(o >> (8 * k)));
    index.insert(index.end(), body.begin(), body.end());
    Bytes changed(cff.begin(), cff.begin() + charstrings);
    changed.insert(changed.end(), index.begin(), index.end());
    changed.insert(changed.end(), cff.begin() + glyphs_end, cff.end());
    if (changed.size() > cff.size()) return fail("the drawn icons are larger than the font's own");
    const long new_private = private_offset + ((long)(charstrings + index.size()) - (long)glyphs_end);
    if (cff[private_operand] == 28) { changed[private_operand + 1] = (u8)(new_private >> 8); changed[private_operand + 2] = (u8)new_private; }
    else Put32(&changed[private_operand + 1], (u32)new_private);
    (void)charstrings_operand;
    changed.resize(cff.size(), 0);
    Bytes font(otf);
    std::copy(changed.begin(), changed.end(), font.begin() + cff_at);

    Bytes wrapped(file.begin(), file.begin() + 8);
    if ((Be32(&file[4]) ^ Be32(key)) != otf.size()) return fail("font header");
    for (size_t i = 0; i < font.size(); ++i) wrapped.push_back(font[i] ^ key[i % 4]);

    out.resize(ZSTD_compressBound(wrapped.size()));
    const size_t n = ZSTD_compress(out.data(), out.size(), wrapped.data(), wrapped.size(), 3);
    if (ZSTD_isError(n)) return fail("not packed");
    out.resize(n);
    RT_LOG("button icons: style %d, A B X Y show %d %d %d %d (glyphs %zu of %zu bytes)", style, labels[0], labels[1], labels[2], labels[3], index.size(), glyphs_end - (size_t)charstrings);
    if (const char* dump = std::getenv("OOT_BUTTON_FONT_DUMP")) if (FILE* f = std::fopen(dump, "wb")) { std::fwrite(font.data(), 1, font.size(), f); std::fclose(f); }
    return true;
}
