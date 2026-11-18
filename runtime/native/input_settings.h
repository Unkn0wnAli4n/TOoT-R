// PC controls: which keys, mouse buttons and gamepad inputs press each of the console's buttons, and how the mouse and
// the sticks behave (the Controller and Keyboard & Mouse tabs of the settings window, nvn/overlay.cpp).
// Stored in <data>/input.ini. Read by the HID thread (hid.cpp) on every poll; changed by the settings window.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

namespace input {

// The console's buttons in the order of their bits in the pad state, then what only a keyboard needs.
enum Action {
    A, B, X, Y, StickL, StickR, L, R, ZL, ZR, Plus, Minus, Left, Up, Right, Down,
    MoveUp, MoveDown, MoveLeft, MoveRight, CamUp, CamDown, CamLeft, CamRight, Walk,
    ActionCount, ButtonCount = MoveUp,
};
const char* ActionName(int action);  // as shown
const char* ActionKey(int action);   // as saved

// A keyboard-and-mouse input: 0 none, an SDL scancode (below 0x1000), or one of these.
enum : int { MouseLeft = 0x1001, MouseMiddle, MouseRight, MouseX1, MouseX2, WheelUp = 0x1010, WheelDown };
// A gamepad input: 0 none, 1 + an SDL gamepad button, or PadAxis + axis * 2 + (1: its positive direction).
enum : int { PadAxis = 0x100 };
std::string KeyName(int code);
std::string PadName(int code);

struct Config {
    int key[ActionCount][2]{};
    int pad[ActionCount][2]{};
    int mouse_camera = 2;    // the mouse turns the camera: 0 no, 1 as the right stick, 2 the camera's own angles (the stick's way where the game takes the stick)
    int mouse_sens = 100;    // percent (10..500)
    int mouse_invert_x = 0, mouse_invert_y = 0;
    int mouse_buttons = 1;   // mouse buttons and the wheel press what they are bound to while the mouse is captured
    int invert_x = 0, invert_y = 0; // the camera stick (and the camera keys)
    int deadzone = 15;       // both sticks, percent of the way (0..40)
    int rumble = 100;        // percent (0: off, ..150)
    int layout = 0;          // face buttons: 0 by position as Nintendo's (A right), 1 by letter as an Xbox pad's (A bottom), 2 custom
    int walk_toggle = 0;     // the Walk key: 0 while held, 1 from one press to the next
    int walk_speed = 45;     // percent of the stick's way while walking (15..90)
    int background = 0;      // 1: the controller is read while another window has the focus
    int menu_chord = 1;      // 1: Select and Start together open the settings window
    int gyro = 0;            // the controller's motion sensor turns the camera: 0 no, 1 while aiming or targeting (a camera button held), 2 always, 3 the game's own motion aiming (its six-axis sensor)
    int gyro_sens = 100;     // percent (10..400): 100 turns the camera as far as the controller
    int gyro_invert_x = 0, gyro_invert_y = 0;
    int mouse_menus = 1;     // 1: over the game's own menus the mouse is a pointer (hover, click, right-click back, wheel)
    int no_centering = 0;    // 1: the camera does not turn back behind Link by itself as he walks
};

const Config& Current();
void Set(const Config& c); // applies and saves
Config Defaults();
void ApplyLayout(Config& c, int layout); // the face buttons' gamepad inputs of a layout

// ---- the mouse (events arrive on the window's thread; the HID thread and the camera hook take them) ----
void MouseMoved(float dx, float dy);
void MouseWheel(float dy);
void MouseButton(int button, bool down);    // SDL's button numbers
bool TakeMotion(float& dx, float& dy);      // what moved since the last take; false when the mouse is not captured
void GiveBack(float dx, float dy);
void GyroTurned(float yaw_degrees, float pitch_degrees); // the controller turned this far (HID thread)
bool TakeGyro(float& yaw_degrees, float& pitch_degrees);  // what it turned since the last take
bool WheelPressed(bool up);                 // a notch turned within the last moments
unsigned MouseButtons();                    // bit n: SDL button n is down (captured only)
inline std::atomic<bool> captured{false};   // the window holds the mouse (present.cpp)
inline std::atomic<bool> release_asked{false}; // Escape let the mouse go; a click in the window takes it again
// The camera hook's side of the direct mouse camera (engine_hooks.cpp): whether the game's own camera took the mouse lately.
inline std::atomic<int64_t> camera_live_ns{0};
inline std::atomic<bool> camera_free{true};  // HID: none of the buttons the game takes the right stick for is held
inline std::atomic<bool> gyro_seen{false};      // a connected controller has a gyroscope
inline std::atomic<int> pad_now[4]{};          // HID: the sticks as the game gets them (left x, y, right x, y; -32767..32767)
inline std::atomic<bool> swallow_sticks{false};  // the free camera has the sticks: the game gets them at rest
inline std::atomic<uint64_t> pad_buttons{0};   // HID: the buttons held (bit = Action)
inline std::atomic<bool> stick_turning{false}; // HID: the stick or the camera keys turn the camera now
int64_t NowNs();

// ---- the mouse as a pointer in the game's own menus (engine_hooks.cpp's menu hooks; the window's thread feeds it) ----
inline std::atomic<int64_t> menu_live_ns{0};   // when a menu screen last took the pointer: the mouse is free meanwhile
inline std::atomic<int> picture_rect[4]{};     // the game's picture in the window, pixels: x, y, width, height (presenter)
struct Pointer { float x = 0, y = 0; bool inside = false; }; // across and up the picture, -1..1 from its middle
void PointerAt(float window_x, float window_y, bool inside);   // window pixels
Pointer PointerNow();
void PointerPress(int sdl_button);
unsigned TakePointerPresses();                 // bit n: SDL button n was pressed since the last take
void PointerWheel(float notches);
int TakePointerWheel();                        // whole notches since the last take (up positive)
bool MenuPointerLive();                        // a menu had the pointer within the last quarter second
// Presses made for the player (a click in a menu is the game's A): the button `action` held for hold_ms, from delay_ms on.
void Tap(int action, int delay_ms, int hold_ms);
uint64_t TappedButtons();                      // the buttons such presses hold now (HID thread)

} // namespace input
