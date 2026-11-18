#include <map>
#include "nvn/overlay.h"
// One local controller backed by SDL gamepads and a keyboard. No Horizon HID IPC.
// Motion, Joy-Con mouse and vibration devices are not advertised by this backend.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <SDL3/SDL.h>
#include "../native.h"
#include "graphics_settings.h"
#include "input_settings.h"


using namespace rt;
namespace {
std::mutex input_mutex;
u32 supported_styles = 1, joy_hold = 0;
std::array<bool, 8> supported_ids{true, true, true, true, true, true, true, true};
std::atomic<u64> sampling_number{0};
std::vector<SDL_Gamepad*> gamepads; // (all connected: player 1)

void Initialize() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD | SDL_INIT_EVENTS)) Fatal("SDL input: %s", SDL_GetError());
        RT_LOG("input: player 1 uses SDL gamepads, the keyboard and the mouse (bindings: the settings window, input.ini)");
    });
}

// Only the first player is connected. Prefer a full controller, then a Joy-Con pair.
u32 Style(u32 id) {
    if (id != 0 || !supported_ids[0]) return 0;
    for (u32 style : {1u, 4u, 8u, 16u})
        if (supported_styles & style) return style;
    return 0;
}

struct NpadState {
    u64 sample, buttons;
    s32 lx, ly, rx, ry;
    u32 attributes, reserved;
};
static_assert(sizeof(NpadState) == 0x28);

// The mouse as the camera. Direct (mouse_camera 2): while the game's own camera follows the player or is the player's
// (engine_hooks.cpp's camera hook ran lately and none of the buttons the game takes the right stick for is held), the
// motion is left for that hook, which turns the camera's angles by it. Otherwise, and with mouse_camera 1, the motion
// is the right stick: every movement adds to a deflection that is shown to the game and dies away within a tenth of a
// second, so the turn follows the hand at any mouse report rate and any poll rate, and aiming, menus and item wheels
// take it as they take the stick.
std::atomic<float> gyro_rate[3]{}; std::atomic<s64> gyro_rate_ns{0}; // (radians a second about SDL's x, y, z; when read)
void MouseCamera(const input::Config& cfg, NpadState& state) {
    static double look_x = 0, look_y = 0;
    static s64 last_ns = 0;
    static const char* test = std::getenv("OOT_MOUSE_TEST"); // ("dx,dy" counts a second, for tests without a mouse)
    static const char* gyro_test = std::getenv("OOT_GYRO_TEST"); // ("yaw,pitch" degrees a second, for tests without a controller)
    const s64 now = input::NowNs();
    const double dt = last_ns ? std::min(0.1, (now - last_ns) / 1e9) : 0.0;
    last_ns = now;
    if (test) {
        float tx = 0, ty = 0; std::sscanf(test, "%f,%f", &tx, &ty);
        input::captured = true; input::MouseMoved(tx * (float)dt, ty * (float)dt);
    }
    if (gyro_test) { float gy = 0, gp = 0; std::sscanf(gyro_test, "%f,%f", &gy, &gp); input::GyroTurned(gy * (float)dt, gp * (float)dt); gyro_rate[1] = gy / 57.29578f; gyro_rate[0] = gp / 57.29578f; gyro_rate_ns = now; }
    input::stick_turning = state.rx != 0 || state.ry != 0;
    input::pad_now[0] = state.lx; input::pad_now[1] = state.ly; input::pad_now[2] = state.rx; input::pad_now[3] = state.ry; input::pad_buttons = state.buttons;
    if (input::swallow_sticks.load(std::memory_order_relaxed)) { state.lx = state.ly = state.rx = state.ry = 0; state.buttons &= ~(1ULL << input::Up | 1ULL << input::Down | 1ULL << input::B); }
    input::camera_free = !(state.buttons & (1ULL << input::ZR | 1ULL << input::L | 1ULL << input::R));
    const bool aiming = state.buttons & (1ULL << input::ZR | 1ULL << input::L | 1ULL << input::R | 1ULL << input::ZL);
    const bool camera_live = now - input::camera_live_ns.load(std::memory_order_relaxed) < 150'000'000;
    const bool mouse_on = cfg.mouse_camera && input::captured.load(std::memory_order_relaxed);
    // (the motion sensor: always, or only while a button that takes the camera is held - aiming, targeting)
    const bool gyro_on = cfg.gyro == 2 || (cfg.gyro == 1 && aiming);
    float gyaw = 0, gpitch = 0;
    const bool gyro_direct = cfg.gyro == 2 && input::camera_free.load() && camera_live; // (left for the camera hook)
    if (!gyro_direct) { input::TakeGyro(gyaw, gpitch); if (!gyro_on) gyaw = gpitch = 0; }
    if (!mouse_on && !gyro_on) { look_x = look_y = 0; return; }
    const bool direct = cfg.mouse_camera == 2 && input::camera_free.load() && camera_live;
    static bool said = false;
    if (direct && mouse_on && !said) { said = true; RT_LOG("input: the mouse turns the camera's own angles"); }
    float dx = 0, dy = 0;
    if (mouse_on && !direct) input::TakeMotion(dx, dy);
    const double k = cfg.mouse_sens / 100.0 * 0.03;
    look_x += (cfg.mouse_invert_x ? -dx : dx) * k;
    look_y += (cfg.mouse_invert_y ? dy : -dy) * k;
    // (a degree of the controller's turn as ten counts of a mouse at the gyro's own sensitivity)
    const double kg = cfg.gyro_sens / 100.0 * 0.3;
    look_x += (cfg.gyro_invert_x ? gyaw : -gyaw) * kg;
    look_y += (cfg.gyro_invert_y ? -gpitch : gpitch) * kg;
    const bool moving = dx || dy || gyaw || gpitch;
    double length = std::sqrt(look_x * look_x + look_y * look_y);
    if (length > 1.5) { look_x *= 1.5 / length; look_y *= 1.5 / length; length = 1.5; }
    if (length > 0.01) {
        // (shown: at most the stick's edge, and at least past the game's own dead zone while the hand moves)
        const double shown = std::clamp(length, moving ? 0.22 : 0.0, 1.0) / length;
        const s32 mx = (s32)std::lround(look_x * shown * 32767), my = (s32)std::lround(look_y * shown * 32767);
        if (std::abs(mx) > std::abs(state.rx)) state.rx = mx;
        if (std::abs(my) > std::abs(state.ry)) state.ry = my;
    }
    const double keep = std::exp(-dt / 0.030); // (the deflection's life: gone within ~0.1 s of the hand stopping)
    look_x *= keep; look_y *= keep;
    if (length < 0.02) look_x = look_y = 0;
}

// The controller's gyroscope (SDL's sensor data: radians a second about the pad's x, pointing right; y, up; z, towards
// the player), as degrees of yaw and pitch turned since the last poll. Turning the pad to the left is a positive yaw.
void ReadGyro(SDL_Gamepad* gamepad) {
    static std::map<SDL_Gamepad*, s64> last;
    if (!SDL_GamepadHasSensor(gamepad, SDL_SENSOR_GYRO)) return;
    if (!SDL_GamepadSensorEnabled(gamepad, SDL_SENSOR_GYRO)) {
        if (!SDL_SetGamepadSensorEnabled(gamepad, SDL_SENSOR_GYRO, true)) return;
        RT_LOG("input gamepad: %s has a gyroscope", SDL_GetGamepadName(gamepad)); input::gyro_seen = true;
    }
    float rate[3]{};
    if (!SDL_GetGamepadSensorData(gamepad, SDL_SENSOR_GYRO, rate, 3)) return;
    const s64 now = input::NowNs(); s64& before = last[gamepad];
    const double dt = before ? std::min(0.05, (now - before) / 1e9) : 0.0; before = now;
    const double to_degrees = 180.0 / 3.14159265358979;
    for (int i = 0; i < 3; ++i) gyro_rate[i].store(rate[i], std::memory_order_relaxed); gyro_rate_ns = now; // (the game's own sensor reads: below)
    double yaw = rate[1] * to_degrees, pitch = rate[0] * to_degrees; // (degrees a second)
    if (std::fabs(yaw) < 0.6) yaw = 0; if (std::fabs(pitch) < 0.6) pitch = 0; // (a resting pad's noise)
    if (yaw != 0 || pitch != 0) input::GyroTurned((float)(yaw * dt), (float)(pitch * dt));
}

NpadState ReadPad(u32 id, u32 requested_style) {
    Initialize();
    NpadState state{};
    state.sample = ++sampling_number;
    if (Style(id) != requested_style) return state;
    state.attributes = requested_style == 4 ? 0x3f : 3; // connected, wired; both halves for a pair
    if (SDL_IsMainThread()) {
        SDL_PumpEvents();
        if (!rt::g_window_pump) { rt::g_window_thread = std::this_thread::get_id(); rt::g_window_pump = [] { SDL_PumpEvents(); }; }
    }
    SDL_UpdateGamepads();
    // Every connected gamepad drives player 1: buttons combine, each stick axis follows whichever pad
    // pushes it furthest.
    {
        int count = 0;
        SDL_JoystickID* ids = SDL_GetGamepads(&count);
        for (auto it = gamepads.begin(); it != gamepads.end();) {
            if (SDL_GamepadConnected(*it)) { ++it; continue; }
            SDL_CloseGamepad(*it); it = gamepads.erase(it);
        }
        for (int i = 0; i < count; ++i) {
            bool open = false;
            for (SDL_Gamepad* g : gamepads) open |= SDL_GetGamepadID(g) == ids[i];
            if (open) continue;
            if (SDL_Gamepad* g = SDL_OpenGamepad(ids[i])) { gamepads.push_back(g); RT_LOG("input gamepad: %s", SDL_GetGamepadName(g)); }
        }
        SDL_free(ids);
    }
    const auto set = [&](int bit, bool down) { if (down) state.buttons |= 1ULL << bit; };
    const input::Config& cfg = input::Current();
    if (cfg.gyro) for (SDL_Gamepad* g : gamepads) ReadGyro(g);
    const bool focused = gfx::window_focused.load(std::memory_order_relaxed);
    static const bool always = std::getenv("OOT_INPUT_BACKGROUND") || std::getenv("OOT_WINDOW_HIDDEN"); // (hidden test sessions have no focus)
    const bool pads_read = focused || cfg.background || always;
    const bool* key = SDL_GetKeyboardState(nullptr);
    const unsigned mouse_buttons = cfg.mouse_buttons ? input::MouseButtons() : 0u;
    // One keyboard-and-mouse input, one gamepad input (any connected pad).
    const auto key_down = [&](int code) {
        if (code <= 0 || !focused) return false;
        if (code < SDL_SCANCODE_COUNT) return key[code] && !(code == SDL_SCANCODE_RETURN && (key[SDL_SCANCODE_LALT] || key[SDL_SCANCODE_RALT])); // (Alt+Enter is fullscreen)
        if (code >= input::MouseLeft && code <= input::MouseX2) return (mouse_buttons >> (code - 0x1000) & 1) != 0;
        if (code == input::WheelUp || code == input::WheelDown) return cfg.mouse_buttons && input::captured.load(std::memory_order_relaxed) && input::WheelPressed(code == input::WheelUp);
        return false;
    };
    const auto pad_down = [&](int code) {
        if (code <= 0 || !pads_read) return false;
        for (SDL_Gamepad* gamepad : gamepads) {
            if (code < input::PadAxis) { if (SDL_GetGamepadButton(gamepad, (SDL_GamepadButton)(code - 1))) return true; continue; }
            const int at = code - input::PadAxis;
            const int value = SDL_GetGamepadAxis(gamepad, (SDL_GamepadAxis)(at / 2));
            if (at & 1 ? value > 16384 : value < -16384) return true;
        }
        return false;
    };
    const auto down = [&](int action) { return key_down(cfg.key[action][0]) || key_down(cfg.key[action][1]) || pad_down(cfg.pad[action][0]) || pad_down(cfg.pad[action][1]); };
    for (int bit = 0; bit < input::ButtonCount; ++bit) set(bit, down(bit));
    // Select and Start together (or the guide button) show or hide the settings window: a pad has no Insert key.
    if (pads_read) {
        bool chord = false;
        for (SDL_Gamepad* gamepad : gamepads)
            chord |= SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_GUIDE) ||
                     (cfg.menu_chord && SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_START) && SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_BACK));
        static bool was = false;
        if (chord && !was) nvn::overlay::Toggle();
        was = chord;
    }
    // The sticks: each as a pair, nothing inside the dead zone and the full way from its edge; every pad's, the
    // furthest one counting.
    const auto stronger = [](s32 a, s32 b) { return std::abs(b) > std::abs(a) ? b : a; };
    if (pads_read) for (SDL_Gamepad* gamepad : gamepads) {
        const auto stick = [&](SDL_GamepadAxis ax, SDL_GamepadAxis ay, s32& out_x, s32& out_y) {
            const double x = SDL_GetGamepadAxis(gamepad, ax) / 32767.0, y = -SDL_GetGamepadAxis(gamepad, ay) / 32767.0;
            const double length = std::sqrt(x * x + y * y), zone = cfg.deadzone / 100.0;
            if (length <= zone || length <= 0) return;
            const double scale = std::min(1.0, (length - zone) / (1.0 - zone)) / length;
            out_x = stronger(out_x, (s32)std::lround(std::clamp(x * scale, -1.0, 1.0) * 32767));
            out_y = stronger(out_y, (s32)std::lround(std::clamp(y * scale, -1.0, 1.0) * 32767));
        };
        stick(SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, state.lx, state.ly);
        stick(SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY, state.rx, state.ry);
    }
    static const bool trace=std::getenv("OOT_INPUT_TRACE")!=nullptr;
    static NpadState previous{};
    if(trace&&(state.buttons!=previous.buttons||state.lx!=previous.lx||state.ly!=previous.ly||
               state.rx!=previous.rx||state.ry!=previous.ry)){
        RT_LOG("physical gamepad input: buttons %#llx left (%d,%d) right (%d,%d)",
            (unsigned long long)state.buttons,state.lx,state.ly,state.rx,state.ry);
        previous=state;
    }
    // The movement and camera inputs: the stick at its edge, a diagonal at its length; while walking, the movement
    // at Walk Speed's share of the way.
    {
        static bool walking = false, walk_was = false;
        const bool walk_key = down(input::Walk);
        if (cfg.walk_toggle) { if (walk_key && !walk_was) walking = !walking; } else walking = walk_key;
        walk_was = walk_key;
        const auto digital = [&](int up, int down_, int left, int right, double reach, s32& out_x, s32& out_y) {
            const int x = (int)down(right) - (int)down(left), y = (int)down(up) - (int)down(down_);
            if (!x && !y) return;
            const double length = x && y ? 0.70710678 : 1.0;
            out_x = (s32)std::lround(x * length * reach * 32767); out_y = (s32)std::lround(y * length * reach * 32767);
        };
        digital(input::MoveUp, input::MoveDown, input::MoveLeft, input::MoveRight, walking ? cfg.walk_speed / 100.0 : 1.0, state.lx, state.ly);
        digital(input::CamUp, input::CamDown, input::CamLeft, input::CamRight, 1.0, state.rx, state.ry);
    }
    if (cfg.invert_x) state.rx = -state.rx;
    if (cfg.invert_y) state.ry = -state.ry;
    MouseCamera(cfg, state);
    state.buttons |= input::TappedButtons(); // (clicks in the game's menus: engine_hooks.cpp)
    nvn::overlay::FilterInput(state.buttons, state.lx, state.ly, state.rx, state.ry, key); // (the settings window takes input while open)
    set(16, state.lx < -16384); set(17, state.ly > 16384);
    set(18, state.lx > 16384); set(19, state.ly < -16384);
    set(20, state.rx < -16384); set(21, state.ry > 16384);
    set(22, state.rx > 16384); set(23, state.ry < -16384);
    return state;
}

// Rumble. The game sends each side of the controller a value {low band's amplitude, its frequency in Hz, high band's
// amplitude, its frequency} for the console's linear actuators; a PC pad has a heavy and a light motor. The low band goes
// to the heavy motor and the high band to the light one, each side's stronger value counting. The console's amplitudes
// are a few hundredths for most effects (a linear actuator at 0.05 is clearly felt, a motor at 5% is not), so they are
// raised by a root curve before the Rumble Strength setting scales them.
void Rumble(u32 side, const float* value) {
    static std::mutex rumble_mutex;
    static float asked[2][2]{}; // [side][0 low, 1 high]
    static u16 sent[2]{};
    static s64 sent_ns = 0, logged = 0;
    const int strength = input::Current().rumble;
    std::lock_guard lock{rumble_mutex};
    const auto curve = [&](float a) { return !(a > 0.0f) ? 0.0f : std::min(1.0f, std::pow(std::min(a, 1.0f), 0.45f) * strength / 100.0f); };
    asked[side & 1][0] = curve(value[0]); asked[side & 1][1] = curve(value[2]);
    if ((value[0] > 0.01f || value[2] > 0.01f) && logged++ < 4) RT_LOG("input: rumble: low %.2f at %.0f Hz, high %.2f at %.0f Hz", value[0], value[1], value[2], value[3]);
    const u16 motors[2] = {(u16)(std::max(asked[0][0], asked[1][0]) * 65535.0f), (u16)(std::max(asked[0][1], asked[1][1]) * 65535.0f)};
    const s64 now = input::NowNs();
    const bool silent = !motors[0] && !motors[1];
    if (motors[0] == sent[0] && motors[1] == sent[1] && (silent || now - sent_ns < 200'000'000)) return; // (a held value is sent again before its time runs out)
    sent[0] = motors[0]; sent[1] = motors[1]; sent_ns = now;
    std::lock_guard pads{input_mutex};
    for (SDL_Gamepad* gamepad : gamepads) SDL_RumbleGamepad(gamepad, motors[0], motors[1], silent ? 0 : 500);
}

void GetStates(Cpu* c, u32 style) {
    if ((s32)c->x[1] <= 0) return Ret(c, 0);
    std::lock_guard lock{input_mutex};
    const NpadState state = ReadPad(*Ptr<u32>(c, 2), style);
    std::memcpy(Ptr(c, 0), &state, sizeof state);
    Ret(c, 1); // one current sample; never invent historical states
}
}

NATIVE("_ZN2nn3hid14InitializeNpadEv") { Initialize(); }
NATIVE("_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") {
    std::lock_guard lock{input_mutex}; supported_styles = (u32)c->x[0];
}
NATIVE("_ZN2nn3hid24GetSupportedNpadStyleSetEv") {
    std::lock_guard lock{input_mutex}; Ret(c, supported_styles);
}
NATIVE("_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm") {
    std::lock_guard lock{input_mutex};
    supported_ids.fill(false);
    for (u64 i = 0; i < c->x[1]; ++i) {
        const u32 id = Ptr<u32>(c, 0)[i];
        if (id < supported_ids.size()) supported_ids[id] = true;
    }
}
NATIVE("_ZN2nn3hid15GetNpadStyleSetERKj") {
    std::lock_guard lock{input_mutex}; Ret(c, Style(*Ptr<u32>(c, 0)));
}
NATIVE("_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE") {
    std::lock_guard lock{input_mutex}; joy_hold = c->x[0];
}
NATIVE("_ZN2nn3hid18GetNpadJoyHoldTypeEv") {
    std::lock_guard lock{input_mutex}; Ret(c, joy_hold);
}
NATIVE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj") { GetStates(c, 1); }
NATIVE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadHandheldStateEiRKj") { GetStates(c, 2); }
NATIVE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyDualStateEiRKj") { GetStates(c, 4); }
NATIVE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyLeftStateEiRKj") { GetStates(c, 8); }
NATIVE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadJoyRightStateEiRKj") { GetStates(c, 16); }

// Raw keyboard: USB usage bits match SDL scancodes for the standard 256-key bitmap.
NATIVE("_ZN2nn3hid18InitializeKeyboardEv") { Initialize(); }
NATIVE("_ZN2nn3hid16GetKeyboardStateEPNS0_13KeyboardStateE") {
    Initialize();
    if (SDL_IsMainThread()) SDL_PumpEvents();
    u8* out = Ptr<u8>(c, 0);
    std::memset(out, 0, 0x30);
    const u64 sample = ++sampling_number;
    std::memcpy(out, &sample, 8);
    out[12] = 1; // connected
    const bool* keys = SDL_GetKeyboardState(nullptr);
    for (int scan = 4; scan < 224; ++scan)
        if (keys[scan]) out[16 + scan / 8] |= 1u << (scan % 8);
    const auto mods = SDL_GetModState();
    u32 modifier = 0;
    if (mods & SDL_KMOD_CTRL) modifier |= 1u << 0;
    if (mods & SDL_KMOD_SHIFT) modifier |= 1u << 1;
    if (mods & SDL_KMOD_LALT) modifier |= 1u << 2;
    if (mods & SDL_KMOD_RALT) modifier |= 1u << 3;
    if (mods & SDL_KMOD_GUI) modifier |= 1u << 4;
    if (mods & SDL_KMOD_CAPS) modifier |= 1u << 8;
    if (mods & SDL_KMOD_SCROLL) modifier |= 1u << 9;
    if (mods & SDL_KMOD_NUM) modifier |= 1u << 10;
    std::memcpy(out + 8, &modifier, sizeof modifier);
}
NATIVE("_ZN2nn3hid15InitializeMouseEv") { Initialize(); }
NATIVE("_ZN2nn3hid14GetMouseStatesEPNS0_10MouseStateEi") {
    if ((s32)c->x[1] <= 0) return Ret(c, 0);
    Initialize();
    std::lock_guard lock{input_mutex};
    struct MouseState { u64 sample; s32 x, y, dx, dy, wheel_y, wheel_x; u32 buttons, attributes; } state{};
    static_assert(sizeof(state) == 0x28);
    state.sample = ++sampling_number;
    static float x = 0, y = 0;
    static SDL_MouseButtonFlags buttons = 0;
    if (SDL_IsMainThread()) {
        SDL_PumpEvents();
        buttons = SDL_GetMouseState(&x, &y);
    }
    state.x = (s32)x; state.y = (s32)y; state.attributes = 1;
    if (buttons & SDL_BUTTON_LMASK) state.buttons |= 1;
    if (buttons & SDL_BUTTON_RMASK) state.buttons |= 2;
    if (buttons & SDL_BUTTON_MMASK) state.buttons |= 4;
    std::memcpy(Ptr(c, 0), &state, sizeof state);
    Ret(c, 1);
}
// No motion, vibration, or Joy-Con mouse handles are exposed. Callers can detect absence.
NATIVE("_ZN2nn3hid16InitializeNmouseEPvmii") { Ret(c, 0); }
NATIVE("_ZN2nn3hid14FinalizeNmouseEv") {}
NATIVE("_ZN2nn3hid16GetNmouseHandlesEPNS0_12NmouseHandleEij") { Ret(c, 0); }
NATIVE("_ZN2nn3hid22InitializeSensorFusionEPvm") { Ret(c, 0); }
NATIVE("_ZN2nn3hid20FinalizeSensorFusionEv") {}
NATIVE("_ZN2nn3hid22GetSensorFusionHandlesEPNS0_18SensorFusionHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") { Ret(c, 0); }
// The game's own motion aiming (its Options: Motion-Control Aiming) reads the controller's six-axis sensor. One handle
// for whatever style is asked; its states are made from the gamepad's gyroscope while Gyro Camera is at "Game's own
// aiming" (a controller at rest otherwise). OOT_SIXAXIS=0: no handle, as before 2026-11-17.
NATIVE("_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") {
    static const bool on = [] { const char* e = std::getenv("OOT_SIXAXIS"); return !e || std::atoi(e) != 0; }();
    if (!on || (s32)c->x[1] <= 0 || !c->x[0]) return Ret(c, 0);
    *Ptr<u32>(c, 0) = 0x00010000u | (*Ptr<u32>(c, 2) & 0xff) << 8 | 3; // (a device index, the pad's id, a type: opaque to the game)
    static bool said = false; if (!said) { said = true; RT_LOG("input: the game took a six-axis sensor handle"); }
    Ret(c, 1);
}
// A vibration handle: {u8 style type (3: full key), u8 npad id, u8 device (0 left, 1 right), u8 reserved}; one device a side.
NATIVE("_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") {
    static const bool off = std::getenv("OOT_NO_RUMBLE") != nullptr;
    const int count = off ? 0 : std::clamp((int)(s32)c->x[1], 0, 2);
    for (int i = 0; i < count; ++i) Ptr<u32>(c, 0)[i] = 3u | (*Ptr<u32>(c, 2) & 0xff) << 8 | (u32)i << 16;
    Ret(c, count);
}

// ---- device-facing HID calls without a host counterpart (yet) ---------------------------------------
NATIVE("_ZN2nn3hid14DisconnectNpadERKj") {}
NATIVE("_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE") {}
NATIVE("_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE") { Rumble(*Ptr<u32>(c, 0) >> 16 & 1, Ptr<float>(c, 1)); }
NATIVE("_ZN2nn3hid15VibrationWriter5WriteERKNS0_14VibrationValueE") { Rumble(0, Ptr<float>(c, 1)); Rumble(1, Ptr<float>(c, 1)); }
NATIVE("_ZN2nn3hid29IsOunceVibrationDeviceMountedERKNS0_21VibrationDeviceHandleE") { Ret(c, 0); }
// What the SDK's own vibration classes (VibrationTarget, VibrationPlayer, VibrationWriter: recompiled code) call; on the
// console each is a request to the HID service. A device is a linear actuator on its side of the pad, always there and
// permitted. (Routed here by data/<version>/sdk.overrides; the list and the answers are Longshot's, host/nn_hid.c.)
NATIVE("_ZN2nn3hid6detail15PermitVibrationEb") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail20IsVibrationPermittedEPb") { *Ptr<u8>(c, 0) = 1; Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail27BeginPermitVibrationSessionEv") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail25EndPermitVibrationSessionEv") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE") {
    Ptr<u32>(c, 0)[0] = 1; Ptr<u32>(c, 0)[1] = (*Ptr<u32>(c, 1) >> 16 & 1) + 1; Ret(c, 0); // {linear actuator; left 1, right 2}
}
NATIVE("_ZN2nn3hid6detail18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE") { Rumble(*Ptr<u32>(c, 0) >> 16 & 1, Ptr<float>(c, 1)); Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail19SendVibrationValuesEPKNS0_21VibrationDeviceHandleEPKNS0_14VibrationValueEi") {
    for (int i = 0; i < (int)(s32)c->x[2] && i < 8; ++i) Rumble(Ptr<u32>(c, 0)[i] >> 16 & 1, Ptr<float>(c, 1) + 4 * i);
    Ret(c, 0);
}
NATIVE("_ZN2nn3hid6detail21EnqueueVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE") { Rumble(*Ptr<u32>(c, 0) >> 16 & 1, Ptr<float>(c, 1)); Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail28FlushEnqueuedVibrationValuesEv") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE") {
    const float idle[4] = {0.0f, 160.0f, 0.0f, 320.0f}; std::memcpy(Ptr(c, 0), idle, sizeof idle); Ret(c, 0);
}
NATIVE("_ZN2nn3hid6detail25SendVibrationGcErmCommandERKNS0_21VibrationDeviceHandleERKNS0_21VibrationGcErmCommandE") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail30GetActualVibrationGcErmCommandEPNS0_21VibrationGcErmCommandERKNS0_21VibrationDeviceHandleE") { *Ptr<u32>(c, 0) = 0; Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail24IsVibrationDeviceMountedEPbRKNS0_21VibrationDeviceHandleE") { *Ptr<u8>(c, 0) = 1; Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail29IsOunceVibrationDeviceMountedEPbRKNS0_21VibrationDeviceHandleE") { *Ptr<u8>(c, 0) = 0; Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail28SetVibrationNxCompatibleModeEb") { Ret(c, 0); }
NATIVE("_ZN2nn3hid6detail34IsVibrationNxCompatibleModeEnabledEPb") { *Ptr<u8>(c, 0) = 0; Ret(c, 0); }
NATIVE("_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE") {}
NATIVE("_ZN2nn3hid25SetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleERKNS0_22GyroscopeZeroDriftModeE") {}
// nn::hid::SixAxisSensorState: delta time and sample number (s64), acceleration (g), angular velocity (1.0 = 360
// degrees a second), angle (turns, summed), a 3x3 direction matrix, attributes (bit 0: connected). Its axes: x to the
// pad's right, y away from the player, z up out of its face; SDL's: x right, y up, z towards the player.
NATIVE("_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE") {
    if ((s32)c->x[1] <= 0 || !c->x[0]) return Ret(c, 0);
    struct State { s64 delta, sample; float acceleration[3], velocity[3], angle[3], direction[9]; u32 attributes, reserved; };
    static_assert(sizeof(State) == 0x60);
    static std::mutex m; static State st{0, 0, {0, 0, -1}, {}, {}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 1, 0}; static s64 last = 0;
    std::lock_guard lock{m};
    const s64 now = input::NowNs(); const double dt = last ? std::min(0.05, (now - last) / 1e9) : 0.0; last = now;
    double w[3]{};
    if (input::Current().gyro == 3 && now - gyro_rate_ns.load() < 200'000'000) {
        const double k = input::Current().gyro_sens / 100.0;
        w[0] = gyro_rate[0].load() * k; w[1] = -gyro_rate[2].load() * k; w[2] = gyro_rate[1].load() * k; // (radians a second, the console's axes)
    }
    for (int i = 0; i < 3; ++i) { st.velocity[i] = (float)(w[i] / 6.283185307); st.angle[i] += (float)(w[i] / 6.283185307 * dt); }
    {   // (the direction turned by the rotation of this step, in the pad's own frame: R <- R * (I + [w dt]x), made orthonormal)
        const double a = w[0] * dt, b = w[1] * dt, g = w[2] * dt; float* R = st.direction; double N[9];
        const double d[9] = {1, -g, b, g, 1, -a, -b, a, 1};
        for (int r = 0; r < 3; ++r) for (int col = 0; col < 3; ++col) N[r * 3 + col] = R[r * 3] * d[col] + R[r * 3 + 1] * d[3 + col] + R[r * 3 + 2] * d[6 + col];
        const auto norm = [](double* v) { const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); if (l > 1e-9) for (int i = 0; i < 3; ++i) v[i] /= l; };
        double x[3] = {N[0], N[3], N[6]}, y[3] = {N[1], N[4], N[7]}; norm(x);
        const double dot = x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; for (int i = 0; i < 3; ++i) y[i] -= dot * x[i]; norm(y);
        const double z[3] = {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
        for (int i = 0; i < 3; ++i) { R[i * 3] = (float)x[i]; R[i * 3 + 1] = (float)y[i]; R[i * 3 + 2] = (float)z[i]; }
        // (gravity as the pad now sees it: the world's down in its frame)
        st.acceleration[0] = -R[6]; st.acceleration[1] = -R[7]; st.acceleration[2] = -R[8];
    }
    st.delta = (s64)(dt * 1e9); ++st.sample;
    std::memcpy(Ptr(c, 0), &st, sizeof st);
    static bool said = false; if (!said) { said = true; RT_LOG("input: the game reads the six-axis sensor"); }
    Ret(c, 1);
}
NATIVE("_ZN2nn3hid13GetJcmsHandleERKjRKNS0_8JcmsTypeE") { Ret(c, 0); }
NATIVE("_ZN2nn3hid9StartJcmsERKNS0_10JcmsHandleE") {}
NATIVE("_ZN2nn3hid8StopJcmsERKNS0_10JcmsHandleE") {}
NATIVE("_ZN2nn3hid12FinalizeJcmsEv") {}
NATIVE("_ZN2nn3hid16GetNmouseTypeSetEPNS_4util10BitFlagSetILi32ENS0_10NmouseTypeEEENS0_12NmouseHandleE") { *Ptr<u32>(c, 0) = 0; }
// The controller-support applet: one player on Npad No1 (the SDL controller).
// ControllerSupportResultInfo {s32 player count; u32 selected npad id}.
NATIVE("_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE") {
    if (c->x[0]) {
        Ptr<s32>(c, 0)[0] = 1;
        Ptr<u32>(c, 0)[1] = 0;
    }
    Ret(c, 0);
}
