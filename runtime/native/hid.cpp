#include "nvn/overlay.h"
// One local controller backed by SDL gamepads and a keyboard. No Horizon HID IPC.
// Motion, Joy-Con mouse and vibration devices are not advertised by this backend.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <SDL3/SDL.h>
#include "../native.h"


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
        RT_LOG("input: player 1 uses an SDL gamepad or keyboard (WASD, arrows, J/K/U/I, Enter)");
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
    const auto stronger = [](s32 a, s32 b) { return std::abs(b) > std::abs(a) ? b : a; };
    for (SDL_Gamepad* gamepad : gamepads) {
        // SDL buttons describe positions. Nintendo A is east, B south, X north, Y west.
        const SDL_GamepadButton buttons[] = {
            SDL_GAMEPAD_BUTTON_EAST, SDL_GAMEPAD_BUTTON_SOUTH, SDL_GAMEPAD_BUTTON_NORTH, SDL_GAMEPAD_BUTTON_WEST,
            SDL_GAMEPAD_BUTTON_LEFT_STICK, SDL_GAMEPAD_BUTTON_RIGHT_STICK,
            SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
        };
        for (int i = 0; i < 8; ++i) set(i, SDL_GetGamepadButton(gamepad, buttons[i]));
        set(8, SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 16384);
        set(9, SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16384);
        set(10, SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_START));
        set(11, SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_BACK));
        set(12, SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_DPAD_LEFT));
        set(13, SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_DPAD_UP));
        set(14, SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_DPAD_RIGHT));
        set(15, SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_DPAD_DOWN));
        const auto axis = [&](SDL_GamepadAxis a) -> s32 {
            const s32 value = SDL_GetGamepadAxis(gamepad, a);
            return value > -4096 && value < 4096 ? 0 : value;
        };
        state.lx = stronger(state.lx, axis(SDL_GAMEPAD_AXIS_LEFTX));
        state.ly = stronger(state.ly, std::min(32767, -axis(SDL_GAMEPAD_AXIS_LEFTY)));
        state.rx = stronger(state.rx, axis(SDL_GAMEPAD_AXIS_RIGHTX));
        state.ry = stronger(state.ry, std::min(32767, -axis(SDL_GAMEPAD_AXIS_RIGHTY)));
    }
    static const bool trace=std::getenv("OOT_INPUT_TRACE")!=nullptr;
    static NpadState previous{};
    if(trace&&(state.buttons!=previous.buttons||state.lx!=previous.lx||state.ly!=previous.ly||
               state.rx!=previous.rx||state.ry!=previous.ry)){
        RT_LOG("physical gamepad input: buttons %#llx left (%d,%d) right (%d,%d)",
            (unsigned long long)state.buttons,state.lx,state.ly,state.rx,state.ry);
        previous=state;
    }
    const bool* key = SDL_GetKeyboardState(nullptr);
    for (const auto [bit, scan] : {std::pair{0, SDL_SCANCODE_J}, {1, SDL_SCANCODE_K},
             {2, SDL_SCANCODE_U}, {3, SDL_SCANCODE_I}, {6, SDL_SCANCODE_Q}, {7, SDL_SCANCODE_E},
             {8, SDL_SCANCODE_LSHIFT}, {9, SDL_SCANCODE_SPACE}, {10, SDL_SCANCODE_RETURN},
             {11, SDL_SCANCODE_BACKSPACE}, {12, SDL_SCANCODE_LEFT}, {13, SDL_SCANCODE_UP},
             {14, SDL_SCANCODE_RIGHT}, {15, SDL_SCANCODE_DOWN}}) set(bit, key[scan]);
    if (key[SDL_SCANCODE_A] || key[SDL_SCANCODE_D]) state.lx = (key[SDL_SCANCODE_D] - key[SDL_SCANCODE_A]) * 32767;
    if (key[SDL_SCANCODE_W] || key[SDL_SCANCODE_S]) state.ly = (key[SDL_SCANCODE_W] - key[SDL_SCANCODE_S]) * 32767;
    nvn::overlay::FilterInput(state.buttons, state.lx, state.ly, state.rx, state.ry, key); // (the Graphics screen takes input while open)
    set(16, state.lx < -16384); set(17, state.ly > 16384);
    set(18, state.lx > 16384); set(19, state.ly < -16384);
    set(20, state.rx < -16384); set(21, state.ry > 16384);
    set(22, state.rx > 16384); set(23, state.ry < -16384);
    return state;
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
NATIVE("_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") { Ret(c, 0); }
NATIVE("_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE") { Ret(c, 0); }

// ---- device-facing HID calls without a host counterpart (yet) ---------------------------------------
NATIVE("_ZN2nn3hid14DisconnectNpadERKj") {}
NATIVE("_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE") {}
NATIVE("_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE") {}
NATIVE("_ZN2nn3hid15VibrationWriter5WriteERKNS0_14VibrationValueE") {}
NATIVE("_ZN2nn3hid29IsOunceVibrationDeviceMountedERKNS0_21VibrationDeviceHandleE") { Ret(c, 0); }
NATIVE("_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE") {}
NATIVE("_ZN2nn3hid25SetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleERKNS0_22GyroscopeZeroDriftModeE") {}
NATIVE("_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE") { Ret(c, 0); }
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
