#include <vector>
#include "input_settings.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>

#include <SDL3/SDL.h>

#include "../rt.h"

namespace input {
namespace {
Config current;
std::mutex mutex;
std::once_flag loaded;

struct Name { const char* shown; const char* saved; };
constexpr Name names[ActionCount] = {
    {"A", "a"}, {"B", "b"}, {"X", "x"}, {"Y", "y"}, {"Left Stick Press", "ls"}, {"Right Stick Press", "rs"},
    {"L", "l"}, {"R", "r"}, {"ZL", "zl"}, {"ZR", "zr"}, {"Plus (Start)", "plus"}, {"Minus (Select)", "minus"},
    {"D-pad Left", "left"}, {"D-pad Up", "up"}, {"D-pad Right", "right"}, {"D-pad Down", "down"},
    {"Move Forward", "move_up"}, {"Move Back", "move_down"}, {"Move Left", "move_left"}, {"Move Right", "move_right"},
    {"Camera Up", "cam_up"}, {"Camera Down", "cam_down"}, {"Camera Left", "cam_left"}, {"Camera Right", "cam_right"},
    {"Walk", "walk"},
};

struct Field { const char* name; int Config::*member; int low, high; };
constexpr Field fields[] = {
    {"mouse_camera", &Config::mouse_camera, 0, 2}, {"mouse_sens", &Config::mouse_sens, 10, 500},
    {"mouse_invert_x", &Config::mouse_invert_x, 0, 1}, {"mouse_invert_y", &Config::mouse_invert_y, 0, 1},
    {"mouse_buttons", &Config::mouse_buttons, 0, 1},
    {"invert_x", &Config::invert_x, 0, 1}, {"invert_y", &Config::invert_y, 0, 1},
    {"deadzone", &Config::deadzone, 0, 40}, {"rumble", &Config::rumble, 0, 150}, {"layout", &Config::layout, 0, 2},
    {"walk_toggle", &Config::walk_toggle, 0, 1}, {"walk_speed", &Config::walk_speed, 15, 90},
    {"background", &Config::background, 0, 1}, {"menu_chord", &Config::menu_chord, 0, 1}, {"no_centering", &Config::no_centering, 0, 1}, {"button_icons", &Config::icons, 0, 3}, {"mouse_menus", &Config::mouse_menus, 0, 1},
    {"gyro", &Config::gyro, 0, 3}, {"gyro_sens", &Config::gyro_sens, 10, 400}, {"gyro_invert_x", &Config::gyro_invert_x, 0, 1}, {"gyro_invert_y", &Config::gyro_invert_y, 0, 1},
};

std::string Path() { return rt::DataDir() + "/input.ini"; }

void Save(const Config& c) {
    std::filesystem::create_directories(std::filesystem::path(Path()).parent_path());
    const std::string temp = Path() + ".new"; // (written beside and moved over the old one, as graphics.ini)
    FILE* f = std::fopen(temp.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "# Controls (the settings window's Controller and Keyboard & Mouse tabs).\n"
                    "# key_<button>: SDL scancodes; 4097..4101 the mouse's buttons, 4112 / 4113 the wheel up / down; 0 none.\n"
                    "# pad_<button>: 1 + an SDL gamepad button, or 256 + axis * 2 (+ 1: the axis's positive direction); 0 none.\n");
    for (const auto& field : fields) std::fprintf(f, "%s=%d\n", field.name, c.*field.member);
    for (int a = 0; a < ActionCount; ++a) {
        std::fprintf(f, "key_%s=%d,%d\n", names[a].saved, c.key[a][0], c.key[a][1]);
        std::fprintf(f, "pad_%s=%d,%d\n", names[a].saved, c.pad[a][0], c.pad[a][1]);
    }
    const bool ok = std::fflush(f) == 0 && !std::ferror(f);
    std::fclose(f);
    std::error_code ec;
    if (ok) std::filesystem::rename(temp, Path(), ec); else std::filesystem::remove(temp, ec);
}

Config Sanitized(Config c) {
    for (const auto& field : fields) c.*field.member = std::clamp(c.*field.member, field.low, field.high);
    for (auto& row : c.key) for (int& code : row) if (code < 0 || code > WheelDown || (code >= SDL_SCANCODE_COUNT && code < MouseLeft)) code = 0;
    for (auto& row : c.pad) for (int& code : row) if (code < 0 || code >= PadAxis + SDL_GAMEPAD_AXIS_COUNT * 2 || (code > SDL_GAMEPAD_BUTTON_COUNT && code < PadAxis)) code = 0;
    return c;
}

void Load() {
    Config c = Defaults();
    if (FILE* f = std::fopen(Path().c_str(), "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            char* eq = std::strchr(line, '=');
            if (!eq || line[0] == '#') continue;
            *eq = 0;
            for (const auto& field : fields) if (!std::strcmp(line, field.name)) c.*field.member = std::atoi(eq + 1);
            const bool key = !std::strncmp(line, "key_", 4), pad = !std::strncmp(line, "pad_", 4);
            if (!key && !pad) continue;
            for (int a = 0; a < ActionCount; ++a) {
                if (std::strcmp(line + 4, names[a].saved)) continue;
                int first = 0, second = 0;
                std::sscanf(eq + 1, "%d,%d", &first, &second);
                int* slot = key ? c.key[a] : c.pad[a];
                slot[0] = first; slot[1] = second;
            }
        }
        std::fclose(f);
    }
    std::lock_guard lock{mutex};
    current = Sanitized(c);
}

// The mouse.
std::mutex mouse_mutex;
float moved_x = 0, moved_y = 0;
std::atomic<unsigned> buttons{0};
std::atomic<int64_t> wheel_up_ns{0}, wheel_down_ns{0};
constexpr int64_t WheelHoldNs = 90'000'000; // (a notch is a short press: long enough for a 20 fps game to see it)
} // namespace

int64_t NowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

const char* ActionName(int action) { return action >= 0 && action < ActionCount ? names[action].shown : ""; }
const char* ActionKey(int action) { return action >= 0 && action < ActionCount ? names[action].saved : ""; }

std::string KeyName(int code) {
    switch (code) {
    case 0: return "-";
    case MouseLeft: return "Mouse Left";
    case MouseMiddle: return "Mouse Middle";
    case MouseRight: return "Mouse Right";
    case MouseX1: return "Mouse 4";
    case MouseX2: return "Mouse 5";
    case WheelUp: return "Wheel Up";
    case WheelDown: return "Wheel Down";
    }
    const char* name = SDL_GetScancodeName((SDL_Scancode)code);
    return name && *name ? name : "Key " + std::to_string(code);
}

std::string PadName(int code) {
    if (code <= 0) return "-";
    if (code >= PadAxis) {
        static const char* const axes[] = {"Left Stick Left", "Left Stick Right", "Left Stick Up", "Left Stick Down", "Right Stick Left", "Right Stick Right",
                                           "Right Stick Up", "Right Stick Down", "Left Trigger", "Left Trigger", "Right Trigger", "Right Trigger"};
        const int at = code - PadAxis;
        return at < 12 ? axes[at] : "Axis";
    }
    static const char* const buttons_shown[] = {"Face Bottom", "Face Right", "Face Left", "Face Top", "Select / Back", "Guide",
                                                "Start", "Left Stick Press", "Right Stick Press", "Left Shoulder", "Right Shoulder", "D-pad Up", "D-pad Down", "D-pad Left",
                                                "D-pad Right", "Misc", "Right Paddle 1", "Left Paddle 1", "Right Paddle 2", "Left Paddle 2", "Touchpad"};
    const int at = code - 1;
    return at < (int)(sizeof buttons_shown / sizeof *buttons_shown) ? buttons_shown[at] : "Button " + std::to_string(at);
}

void ApplyLayout(Config& c, int layout) {
    const auto button = [](SDL_GamepadButton b) { return 1 + (int)b; };
    if (layout == 1) { // by letter, as an Xbox pad is labelled
        c.pad[A][0] = button(SDL_GAMEPAD_BUTTON_SOUTH); c.pad[B][0] = button(SDL_GAMEPAD_BUTTON_EAST);
        c.pad[X][0] = button(SDL_GAMEPAD_BUTTON_WEST); c.pad[Y][0] = button(SDL_GAMEPAD_BUTTON_NORTH);
    } else if (layout == 0) { // by position, as the console's pad
        c.pad[A][0] = button(SDL_GAMEPAD_BUTTON_EAST); c.pad[B][0] = button(SDL_GAMEPAD_BUTTON_SOUTH);
        c.pad[X][0] = button(SDL_GAMEPAD_BUTTON_NORTH); c.pad[Y][0] = button(SDL_GAMEPAD_BUTTON_WEST);
    }
    if (layout != 2) for (int a : {A, B, X, Y}) c.pad[a][1] = 0;
    c.layout = layout;
}

Config Defaults() {
    Config c;
    const auto button = [](SDL_GamepadButton b) { return 1 + (int)b; };
    ApplyLayout(c, 0);
    c.pad[StickL][0] = button(SDL_GAMEPAD_BUTTON_LEFT_STICK); c.pad[StickR][0] = button(SDL_GAMEPAD_BUTTON_RIGHT_STICK);
    c.pad[L][0] = button(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER); c.pad[R][0] = button(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    c.pad[ZL][0] = PadAxis + SDL_GAMEPAD_AXIS_LEFT_TRIGGER * 2 + 1; c.pad[ZR][0] = PadAxis + SDL_GAMEPAD_AXIS_RIGHT_TRIGGER * 2 + 1;
    c.pad[Plus][0] = button(SDL_GAMEPAD_BUTTON_START); c.pad[Minus][0] = button(SDL_GAMEPAD_BUTTON_BACK);
    c.pad[Left][0] = button(SDL_GAMEPAD_BUTTON_DPAD_LEFT); c.pad[Up][0] = button(SDL_GAMEPAD_BUTTON_DPAD_UP);
    c.pad[Right][0] = button(SDL_GAMEPAD_BUTTON_DPAD_RIGHT); c.pad[Down][0] = button(SDL_GAMEPAD_BUTTON_DPAD_DOWN);
    // Keyboard and mouse: WASD and the mouse, with this runtime's earlier keys (J K U I, the arrows) kept beside them
    // where they are free.
    const int keys[ActionCount][2] = {
        {SDL_SCANCODE_E, SDL_SCANCODE_J}, {SDL_SCANCODE_LSHIFT, SDL_SCANCODE_K}, {SDL_SCANCODE_SPACE, SDL_SCANCODE_U}, {MouseLeft, SDL_SCANCODE_I},
        {SDL_SCANCODE_LCTRL, 0}, {SDL_SCANCODE_C, MouseMiddle}, {SDL_SCANCODE_Q, 0}, {SDL_SCANCODE_R, 0},
        {MouseRight, SDL_SCANCODE_Z}, {SDL_SCANCODE_F, SDL_SCANCODE_X}, {SDL_SCANCODE_RETURN, SDL_SCANCODE_TAB}, {SDL_SCANCODE_M, SDL_SCANCODE_BACKSPACE},
        {SDL_SCANCODE_1, 0}, {SDL_SCANCODE_2, WheelUp}, {SDL_SCANCODE_3, 0}, {SDL_SCANCODE_4, WheelDown},
        {SDL_SCANCODE_W, 0}, {SDL_SCANCODE_S, 0}, {SDL_SCANCODE_A, 0}, {SDL_SCANCODE_D, 0},
        {SDL_SCANCODE_UP, 0}, {SDL_SCANCODE_DOWN, 0}, {SDL_SCANCODE_LEFT, 0}, {SDL_SCANCODE_RIGHT, 0},
        {SDL_SCANCODE_CAPSLOCK, 0},
    };
    std::memcpy(c.key, keys, sizeof keys);
    return c;
}

const Config& Current() { std::call_once(loaded, Load); return current; }

void Set(const Config& c) {
    std::call_once(loaded, Load);
    std::lock_guard lock{mutex};
    current = Sanitized(c);
    Save(current);
}

void MouseMoved(float dx, float dy) { std::lock_guard lock{mouse_mutex}; moved_x += dx; moved_y += dy; }
void GiveBack(float dx, float dy) { MouseMoved(dx, dy); }
bool TakeMotion(float& dx, float& dy) {
    std::lock_guard lock{mouse_mutex};
    dx = moved_x; dy = moved_y; moved_x = moved_y = 0;
    if (!captured.load(std::memory_order_relaxed)) { dx = dy = 0; return false; }
    return true;
}
namespace { float gyro_yaw = 0, gyro_pitch = 0; }
void GyroTurned(float yaw, float pitch) { std::lock_guard lock{mouse_mutex}; gyro_yaw += yaw; gyro_pitch += pitch; }
bool TakeGyro(float& yaw, float& pitch) {
    std::lock_guard lock{mouse_mutex};
    yaw = gyro_yaw; pitch = gyro_pitch; gyro_yaw = gyro_pitch = 0;
    return yaw != 0 || pitch != 0;
}
namespace {
Pointer pointer; unsigned pointer_presses = 0; float pointer_wheel = 0;
struct TapItem { int action; int64_t from, until; };
std::vector<TapItem> taps;
}
void PointerAt(float wx, float wy, bool inside) {
    const float x = (float)picture_rect[0].load(), y = (float)picture_rect[1].load(), w = (float)picture_rect[2].load(), h = (float)picture_rect[3].load();
    std::lock_guard lock{mouse_mutex};
    if (!(w > 0) || !(h > 0)) { pointer.inside = false; return; }
    pointer.x = ((wx - x) / w) * 2.0f - 1.0f; pointer.y = 1.0f - ((wy - y) / h) * 2.0f;
    pointer.inside = inside && pointer.x >= -1.0f && pointer.x <= 1.0f && pointer.y >= -1.0f && pointer.y <= 1.0f;
}
Pointer PointerNow() { std::lock_guard lock{mouse_mutex}; return pointer; }
void PointerPress(int button) { if (button >= 1 && button <= 5) { std::lock_guard lock{mouse_mutex}; pointer_presses |= 1u << button; } }
unsigned TakePointerPresses() { std::lock_guard lock{mouse_mutex}; const unsigned p = pointer_presses; pointer_presses = 0; return p; }
void PointerWheel(float notches) { std::lock_guard lock{mouse_mutex}; pointer_wheel += notches; }
int TakePointerWheel() { std::lock_guard lock{mouse_mutex}; const int n = (int)pointer_wheel; pointer_wheel -= (float)n; return n; }
bool MenuPointerLive() { const int64_t t = menu_live_ns.load(std::memory_order_relaxed); return t && NowNs() - t < 250'000'000; }
void Tap(int action, int delay_ms, int hold_ms) {
    const int64_t now = NowNs();
    std::lock_guard lock{mouse_mutex};
    if (taps.size() < 32) taps.push_back({action, now + delay_ms * 1'000'000ll, now + (delay_ms + hold_ms) * 1'000'000ll});
}
uint64_t TappedButtons() {
    const int64_t now = NowNs(); uint64_t held = 0;
    std::lock_guard lock{mouse_mutex};
    for (auto it = taps.begin(); it != taps.end();) {
        if (now >= it->until) { it = taps.erase(it); continue; }
        if (now >= it->from) held |= 1ull << it->action;
        ++it;
    }
    return held;
}
void MouseWheel(float dy) { if (dy > 0) wheel_up_ns = NowNs(); else if (dy < 0) wheel_down_ns = NowNs(); }
bool WheelPressed(bool up) { return NowNs() - (up ? wheel_up_ns : wheel_down_ns).load(std::memory_order_relaxed) < WheelHoldNs; }
void MouseButton(int button, bool down) {
    if (button < 1 || button > 5) return;
    if (down) buttons.fetch_or(1u << button); else buttons.fetch_and(~(1u << button));
}
unsigned MouseButtons() { return captured.load(std::memory_order_relaxed) ? buttons.load(std::memory_order_relaxed) : 0u; }

} // namespace input
