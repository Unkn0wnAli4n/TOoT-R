// The settings window (graphics, game, controller, keyboard and mouse): Dear ImGui (third_party/imgui) drawn over the
// presented frame, in the game's own font when it can be read. F1 or Insert shows or hides it; while shown it takes the mouse,
// keyboard and gamepad (the game sees no input). Nothing is set up until it is first shown.
#include "overlay.h"
#include "present.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <zstd.h>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

#include "../graphics_settings.h"
#include "../input_settings.h"

bool ReadRomFile(const std::string& path, std::vector<unsigned char>& out); // fs.cpp

namespace nvn::overlay {
namespace {

std::atomic<bool> shown{std::getenv("OOT_SHOW_GRAPHICS") != nullptr}; // (OOT_SHOW_GRAPHICS=1: shown from the start)
bool restart_notice = false;
std::mutex events_mutex;
std::deque<SDL_Event> events; // (polled on the game's thread, handed to ImGui on the presenting one)

// ---- the game's font (RomFS: zstd with the game's dictionary, then Nintendo's XOR wrapping) ----
std::vector<u8> Zstd(const std::vector<u8>& in, const std::vector<u8>* dict) {
    const unsigned long long size = ZSTD_getFrameContentSize(in.data(), in.size());
    if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN || size > (256u << 20)) return {};
    std::vector<u8> out(size);
    ZSTD_DCtx* ctx = ZSTD_createDCtx();
    const size_t got = dict ? ZSTD_decompress_usingDict(ctx, out.data(), out.size(), in.data(), in.size(), dict->data(), dict->size())
                            : ZSTD_decompressDCtx(ctx, out.data(), out.size(), in.data(), in.size());
    ZSTD_freeDCtx(ctx);
    if (ZSTD_isError(got)) return {};
    out.resize(got);
    return out;
}
template <typename T> T Read(const std::vector<u8>& d, size_t at) { T v{}; if (at + sizeof v <= d.size()) std::memcpy(&v, d.data() + at, sizeof v); return v; }
bool SarcFile(const std::vector<u8>& sarc, const char* name, std::vector<u8>& out) {
    if (sarc.size() < 0x20 || std::memcmp(sarc.data(), "SARC", 4)) return false;
    const u32 data = Read<u32>(sarc, 0xc);
    const u16 header = Read<u16>(sarc, 4);
    const size_t sfat = header;
    const u16 count = Read<u16>(sarc, sfat + 6);
    const size_t nodes = sfat + 0xc, sfnt = nodes + (size_t)count * 16;
    const size_t names = sfnt + Read<u16>(sarc, sfnt + 4);
    for (u32 i = 0; i < count; ++i) {
        const u32 attributes = Read<u32>(sarc, nodes + i * 16 + 4);
        const size_t at = names + (attributes & 0xffffff) * 4;
        if (at >= sarc.size() || std::strcmp(reinterpret_cast<const char*>(sarc.data() + at), name)) continue;
        const u32 begin = Read<u32>(sarc, nodes + i * 16 + 8), end = Read<u32>(sarc, nodes + i * 16 + 12);
        if (data + end > sarc.size() || begin > end) return false;
        out.assign(sarc.begin() + data + begin, sarc.begin() + data + end);
        return true;
    }
    return false;
}
std::vector<u8> GameFont(const char* file) {
    std::vector<u8> pack, dict, packed, font;
    if (!ReadRomFile("/Pack/ZsDic.pack.zs", pack)) return {};
    if (!SarcFile(Zstd(pack, nullptr), "zs.zsdic", dict)) return {};
    if (!ReadRomFile(std::string("/Font2/Scalable/") + file, packed)) return {};
    const std::vector<u8> wrapped = Zstd(packed, &dict);
    if (wrapped.size() < 16) return {};
    // bfotf: [magic ^ key][size ^ key][data ^ key, big-endian words]; the data starts with "OTTO".
    const auto be = [&](size_t at) { return (u32)wrapped[at] << 24 | (u32)wrapped[at + 1] << 16 | (u32)wrapped[at + 2] << 8 | wrapped[at + 3]; };
    const u32 key = be(8) ^ 0x4F54544Fu;
    const u32 size = be(4) ^ key;
    if (size > wrapped.size() - 8) return {};
    font.resize((size + 3) & ~3u);
    for (size_t i = 0; i + 3 < font.size() && 8 + i + 3 < wrapped.size(); i += 4) {
        const u32 w = be(8 + i) ^ key;
        font[i] = w >> 24; font[i + 1] = w >> 16; font[i + 2] = w >> 8; font[i + 3] = w;
    }
    font.resize(size);
    return font;
}

// ---- the settings rows ------------------------------------------------------------------------
// Each row is one member of gfx::Settings and the values it can take; `quality`: changing it makes the preset Custom.
struct Choice { const char* name; int value; };
struct Row { const char* section; const char* label; int gfx::Settings::*member; std::vector<Choice> choices; const char* help; bool quality; bool restart; };
std::vector<Choice> Percents(std::initializer_list<int> values) { static std::deque<std::string> names; std::vector<Choice> c;
    for (int v : values) { names.push_back(std::to_string(v) + "%"); c.push_back({names.back().c_str(), v}); } return c; }
const std::vector<Row>& Rows() {
    using S = gfx::Settings;
    static const std::vector<Row> rows = {
        {"", "Preset", &S::preset, {{"Original", gfx::PresetOriginal}, {"Steam Deck", gfx::PresetDeck}, {"Low", gfx::PresetLow}, {"Medium", gfx::PresetMedium}, {"High", gfx::PresetHigh}, {"Ultra", gfx::PresetUltra}, {"Custom", gfx::PresetCustom}},
         "Original is the game as it ships. Steam Deck and Low use the handheld profile, a fixed lower scene resolution, small shadow maps and a fast anti-aliasing filter.", false, false},

        {"Display", "Window Size", &S::resolution, {{"Default", 0}, {"1280 x 720", 1}, {"1280 x 800", 6}, {"1600 x 900", 2}, {"1920 x 1080", 3}, {"1920 x 1200", 7}, {"2560 x 1440", 4}, {"2560 x 1600", 8}, {"3440 x 1440", 9}, {"3840 x 2160", 5}},
         "The size of the game's window on the desktop.", false, false},
        {"Display", "Display Mode", &S::display_mode, {{"Windowed", 0}, {"Fullscreen", 1}}, "Fullscreen fills the display the window is on.", false, false},
        {"Display", "Aspect Ratio", &S::aspect, {{"16:9 (bars)", 0}, {"16:9 stretched", 1}, {"Display's shape", 2}, {"4:3", 6}, {"16:10", 3}, {"21:9", 4}, {"32:9", 5}},
         "16:9 is the game's own picture, shown whole with bars or stretched. The other choices make the game draw a picture of that shape: more of the world at the sides on a wide screen, no bars on a 16:10 one such as the Steam Deck's (and fewer pixels to draw). The HUD keeps its shape. These take effect after a restart.", false, true},
        {"Display", "HUD Placement", &S::hud_edges, {{"Centred (16:9)", 0}, {"Screen edges", 1}},
         "On a picture wider than 16:9: whether hearts, buttons and the minimap sit at the screen's edges or stay where a 16:9 screen has them. Menus stay in the middle.", false, false},
        {"Display", "Window Scaling Filter", &S::upscaling, {{"Linear", 0}, {"Catmull-Rom", 1}, {"Lanczos", 2}, {"FSR 1", 3}}, "Only matters when the window is not the picture's own size (2560 x 1440): the filter that stretches or shrinks the finished picture to fit it. Not the Upscaler, which builds the picture from the 3D scene.", false, false},
        {"Display", "VSync", &S::vsync, {{"Off", 0}, {"On", 1}, {"Adaptive (VRR)", 2}}, "On waits for the display: no tearing, a little more input delay. Adaptive is for a variable refresh rate display (G-Sync, FreeSync) with VRR switched on in the system: every frame is shown whole the moment it is ready, and the frame rate is held just under the display's top rate so it never has to wait.", false, false},
        {"Display", "Frame Generation", &S::frame_gen, {{"Off", 0}, {"2x", 2}, {"3x", 3}, {"4x", 4}, {"6x", 6}, {"Dynamic", 7}},
         "Shows frames made between the game's own: 2x shows two for each frame the game draws, and so on; Dynamic makes as many as reach your display's refresh rate. NVIDIA DLSS Frame Generation makes them where it runs (an RTX 40 or newer card with DLSS as the Upscaler), AMD FSR 3.1's everywhere else. They are made from the scene's depth and motion, so it works in play and pauses in menus and movies. Never more frames than your display shows. Motion looks smoother at the price of one frame of extra delay. Best with the game at a steady 30 or more.", false, false},
        {"Display", "HDR", &S::hdr, {{"Off", 0}, {"On", 1}},
         "Shows the picture in high dynamic range on a display and system with HDR switched on: the game's picture at the Paper White brightness, its highlights raised towards the display's peak. The game itself draws a standard picture; this widens it. Without an HDR display nothing changes.", false, false},
        {"Display", "HDR Peak Brightness", &S::hdr_peak, {{"400 nits", 400}, {"600 nits", 600}, {"800 nits", 800}, {"1000 nits", 1000}, {"1500 nits", 1500}, {"2000 nits", 2000}}, "The brightest your display can show. Too high a value clips the brightest highlights.", false, false},
        {"Display", "HDR Paper White", &S::hdr_paper, {{"100 nits", 100}, {"150 nits", 150}, {"200 nits", 200}, {"250 nits", 250}, {"300 nits", 300}}, "How bright the game's ordinary white - text, the HUD, a sunlit wall - is shown.", false, false},
        {"Display", "HDR Highlights", &S::hdr_boost, {{"Off", 0}, {"25%", 25}, {"50%", 50}, {"75%", 75}, {"100%", 100}}, "How strongly the brightest parts of the picture are raised above paper white. Off shows the game's picture unchanged, only in the HDR signal.", false, false},
        {"Display", "Frame Rate Cap", &S::frame_cap, {{"15", 15}, {"20", 20}, {"30", 30}, {"40", 40}, {"45", 45}, {"60", 60}, {"90", 90}, {"120", 120}, {"144", 144}, {"165", 165}, {"240", 240}, {"Uncapped", 0}},
         "The most frames shown per second. The game keeps its normal speed at every cap. 40 suits a Steam Deck set to 40 Hz or 80 Hz.", true, false},
        {"Display", "FPS Counter", &S::show_fps, {{"Off", 0}, {"On", 1}}, "Shows the frame rate and frame time in the top left corner.", false, false},

        {"Resolution and anti-aliasing", "Quality Profile", &S::handheld, {{"Console", 0}, {"Handheld", 1}}, "Handheld is the game's own lower-detail profile, as the console uses undocked. Takes effect after a restart.", true, true},
        {"Resolution and anti-aliasing", "Upscaler", &S::upscaler, {{"Game's own (neural)", 0}, {"NVIDIA DLSS", 1}, {"AMD FSR 3.1", 3}, {"AMD FSR 4", 4}, {"None (plain stretch)", 6}, {"None + FXAA", 7}},
         "What makes the full picture from the 3D scene, and smooths its edges. Game's own is the game's neural network. NVIDIA DLSS needs an RTX card and an installation with NVIDIA DLSS added; AMD FSR 3.1 runs on any GPU; AMD FSR 4 needs AMD's FSR 4 library for Vulkan on this PC. None leaves the network out for the fastest picture, with jagged edges, or with the fast FXAA filter. Where the chosen one cannot run, the game's own is used. Upscaler Quality's Native step turns DLSS into DLAA.", false, false},
        {"Resolution and anti-aliasing", "Upscaler Quality", &S::upscale_quality, {{"Native (100%)", 1}, {"Quality (67%)", 2}, {"Balanced (58%)", 3}, {"Performance (50%)", 4}, {"Ultra Performance (35%)", 5}, {"Custom (Render Resolution)", 0}},
         "With DLSS or FSR 3.1: how large the 3D scene is drawn before the upscaler fills the picture from it. Quality looks closest to native; Performance and Ultra Performance are much lighter on the GPU. Native is DLAA with DLSS (or FSR's native anti-aliasing): the sharpest. Custom uses the Render Resolution row.", false, false},
        {"Resolution and anti-aliasing", "DLSS Model", &S::dlss_model, {{"Transformer (K)", 0}, {"Transformer 2 (M)", 1}, {"Transformer 2 (L)", 2}, {"NVIDIA default", 3}},
         "The network DLSS and DLAA run. K is DLSS 4's; M and L are DLSS 4.5's second-generation transformer (L the heavier one).", false, false},
        {"Resolution and anti-aliasing", "Render Resolution", &S::render_scale, {{"Game's dynamic", 0}, {"1440p (native)", 100}, {"1080p", 75}, {"900p", 62}, {"810p", 56}, {"720p", 50}, {"630p", 44}, {"540p", 38}},
         "With the game's own upscaler or None (or Upscaler Quality at Custom): the 3D scene's resolution before it is upscaled to the 1440p picture; the HUD and text stay sharp. The game's own setting moves between these by the GPU's load, up to 1080p. A fixed lower value is the largest single saving on a slow GPU; 1440p is sharper than the game ever draws.", true, false},
        {"Resolution and anti-aliasing", "Sharpness", &S::sharpness, {{"Off", 0}, {"25%", 25}, {"50%", 50}, {"75%", 75}, {"100%", 100}, {"125%", 125}, {"150%", 150}, {"200%", 200}}, "Sharpens the finished picture. DLSS, DLAA and FSR give a softer picture than the game's own upscaler: 75% to 125% suits them. Above 100% edges may show light halos.", false, false},

        {"Detail", "Shadow Quality", &S::shadow_quality, {{"Game's (1600)", 0}, {"Lowest (512)", 512}, {"Low (768)", 768}, {"Medium (1024)", 1024}, {"High (2200)", 2200}, {"Highest (3072)", 3072}},
         "The size of the sun's shadow maps. Smaller maps are cheaper to draw and softer; larger ones give sharper shadows. Applies at once.", true, false},
        {"Detail", "Foliage Shadows", &S::foliage_shadows, {{"Off", 0}, {"On", 1}}, "Whether grass and other small foliage is drawn into the shadow maps. Off saves a good part of the shadow passes; grass then casts no shadow.", true, false},
        {"Detail", "Light Probe", &S::probe_rate, {{"Every frame", 1}, {"Every 2nd frame", 2}, {"Every 4th frame", 4}},
         "The game redraws a small picture of the surroundings every frame to light the scene by (its ambient light and reflections). Updating it less often leaves out up to a tenth of the frame's draws in busy places in the frames between; the lighting then follows changes a little later. It made no difference on a fast PC: try it where the processor is the limit.", true, false},
        {"Detail", "Texture Filtering", &S::filtering, {{"Original", 0}, {"Bilinear", 1}, {"Trilinear", 2}, {"Anisotropic 2x", 3}, {"Anisotropic 4x", 4}, {"Anisotropic 8x", 5}, {"Anisotropic 16x", 6}}, "Sharper textures at an angle with anisotropic filtering.", true, false},
        {"Detail", "Texture Detail", &S::texture_detail, {{"Original", 0}, {"Low", 1}, {"Medium", 2}, {"High", 3}, {"Ultra", 4}}, "Lower detail uses smaller mipmaps: less memory bandwidth.", true, false},
        {"Detail", "Texture Memory", &S::texture_memory, {{"Original", 0}, {"Low", 1}, {"High Quality", 2}}, "Low compresses more textures (less video memory). Applies to newly loaded areas.", true, false},
        {"Detail", "Occlusion Tests", &S::occlusion, {{"Off", 0}, {"GPU (experimental)", 1}},
         "The game asks the GPU which objects are hidden behind others and skips those. Off treats everything as visible, which is faster on most PCs. GPU counts them as the console does: it can help a slow GPU in crowded places, and can make objects flicker for a frame. Takes effect after a restart.", false, true},

        {"Game", "Field of View", &S::fov, Percents({80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130}), "The gameplay camera's angle. Cutscenes keep their own.", false, false},
        {"Game", "Camera Speed", &S::camera_speed, Percents({50, 75, 100, 125, 150, 200, 250, 300}), "How fast the right stick turns the camera.", false, false},
        {"Game", "Text Speed", &S::text_speed, {{"Normal", 100}, {"Fast", 200}, {"Faster", 400}, {"Instant", 1000}}, "How fast dialogue text appears.", false, false},

        {"Game", "Skip Opening", &S::skip_opening, {{"Off", 0}, {"On", 1}}, "A new game starts with Link awake in his house: the opening's two movies are not played. Everything the opening sets is still set. Later cutscenes are not affected.", false, false},

        {"Cheats", "Infinite Health", &S::cheat_health, {{"Off", 0}, {"On", 1}}, "Link's hearts are filled again every frame. A hit that would take all of them at once still does.", false, false},
        {"Cheats", "Infinite Rupees", &S::cheat_rupees, {{"Off", 0}, {"On", 1}}, "Rupees never go down: spending them gives them back.", false, false},
        {"Cheats", "Infinite Magic", &S::cheat_magic, {{"Off", 0}, {"On", 1}}, "The magic meter stays full.", false, false},
        {"Hacks", "Game Speed", &S::game_speed, Percents({25, 50, 75, 100, 125, 150, 200, 300}), "How fast the game itself runs: slow motion below 100%, fast forward above. Sound keeps its own speed.", false, false},
        {"Audio", "Volume", &S::volume, Percents({0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}), "The game's master volume.", false, false},
        {"Audio", "Mute in Background", &S::mute_background, {{"Off", 0}, {"On", 1}}, "No sound while another window has the focus.", false, false},
    };
    return rows;
}
int ValueIndex(const gfx::Settings& s, int r) {
    const Row& row = Rows()[r]; const int value = s.*row.member;
    int best = 0; long distance = 1L << 40;
    for (int i = 0; i < (int)row.choices.size(); ++i) { const long d = std::labs((long)row.choices[i].value - value); if (d < distance) { distance = d; best = i; } }
    return best;
}
void Change(int r, int index) { // (the row's choice `index`)
    const Row& row = Rows()[r];
    if (index < 0 || index >= (int)row.choices.size()) return;
    gfx::Settings s = gfx::Current();
    const int value = row.choices[index].value;
    if (r == 0) {
        if (value == gfx::PresetCustom) return; // (Custom is what changing a single setting gives)
        const int handheld = s.handheld;
        s = gfx::PresetValues(value);
        if (s.handheld != handheld) restart_notice = true;
    } else {
        s.*row.member = value;
        if (row.quality) s.preset = gfx::PresetCustom;
        if (row.restart && !(row.member == &gfx::Settings::aspect && value < 2 && gfx::Current().aspect < 2)) restart_notice = true;
    }
    gfx::Set(s);
}

// ---- ImGui ---------------------------------------------------------------------------------------
struct State {
    bool context = false, vulkan = false;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::vector<std::pair<VkImage, VkImageView>> views; // (one per swapchain image)
} ui;

void CheckResult(VkResult r) { if (r != VK_SUCCESS) RT_LOG("graphics window: Vulkan error %d", (int)r); }

void CreateContext(VkExtent2D extent) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // (the settings themselves are saved in graphics.ini)
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6; style.FrameRounding = 4; style.GrabRounding = 4; style.WindowBorderSize = 1;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;
    const float scale = std::clamp(extent.height / 1080.0f, 1.0f, 3.0f);
    style.ScaleAllSizes(scale);
    static std::vector<u8> game_font = GameFont("ClarimoUDPE-DemiBold.Nin_Ounce_NVN2.bfotf.zs"); // (ImGui keeps a pointer)
    ImFontConfig config; config.FontDataOwnedByAtlas = false;
    if (game_font.empty() || !io.Fonts->AddFontFromMemoryTTF(game_font.data(), (int)game_font.size(), 21.0f * scale, &config)) {
        ImFontConfig fallback; fallback.SizePixels = 13.0f * scale;
        io.Fonts->AddFontDefault(&fallback);
    }
    ImGui_ImplSDL3_InitForVulkan(Vk().window);
    ui.context = true;
}

void InitVulkan(VkFormat format) {
    auto& vk = Vk();
    if (ui.vulkan) {
        if (ui.format == format) return;
        VK_CHECK(vkDeviceWaitIdle(vk.device)); // (the queue lock is held by the caller)
        ImGui_ImplVulkan_Shutdown();
        ui.vulkan = false;
    }
    static VkFormat color; color = format;
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = vk.instance; info.PhysicalDevice = vk.physical; info.Device = vk.device;
    info.QueueFamily = vk.queue_family; info.Queue = vk.queue;
    info.DescriptorPoolSize = 16;
    info.MinImageCount = 2; info.ImageCount = 8;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &color;
    info.CheckVkResultFn = CheckResult;
    ImGui_ImplVulkan_Init(&info);
    ui.vulkan = true; ui.format = format;
    RT_LOG("settings window: ready (F1 or Insert; Select + Start on a controller)");
}

// ---- the controls tabs ------------------------------------------------------------------------------
// A binding being chosen: the next key, mouse button or wheel notch (kind 1) or gamepad input (kind 2) becomes
// action's slot. Set by the window (the presenting thread), taken by Event (the thread that polls).
std::atomic<int> capture_kind{0}, capture_action{0}, capture_slot{0};
std::atomic<int64_t> nav_quiet_until{0}; // (the pad's press that was just bound must not also press the focused item)

void Bind(int kind, int action, int slot, int code) {
    input::Config c = input::Current();
    auto& table = kind == 1 ? c.key : c.pad;
    if (code) for (auto& row : table) for (int& other : row) if (other == code) other = 0; // (an input presses one thing)
    table[action][slot] = code;
    if (kind == 2 && action <= input::Y) c.layout = 2;
    input::Set(c);
}

struct InputRow { const char* label; int input::Config::*member; std::vector<Choice> choices; const char* help; };
const std::vector<InputRow>& PadRows() {
    using C = input::Config;
    static const std::vector<InputRow> rows = {
        {"Button Layout", &C::layout, {{"Nintendo (A on the right)", 0}, {"Xbox (A at the bottom)", 1}, {"Custom", 2}},
         "Where the game's A, B, X and Y are on your controller. Nintendo puts them where the console's pad has them; Xbox follows the letters printed on an Xbox pad. Binding a face button yourself makes it Custom."},
        {"Stick Dead Zone", &C::deadzone, Percents({0, 5, 10, 15, 20, 25, 30, 40}), "How far a stick moves before it counts. Raise it if the camera or Link drifts with the stick at rest."},
        {"Invert Camera X", &C::invert_x, {{"Off", 0}, {"On", 1}}, "Turns the camera stick's left and right around (the camera keys too)."},
        {"Invert Camera Y", &C::invert_y, {{"Off", 0}, {"On", 1}}, "Turns the camera stick's up and down around (the camera keys too)."},
        {"Auto-centre Camera", &C::no_centering, {{"On (the game's)", 0}, {"Off", 1}}, "Off: the camera no longer swings back behind Link by itself while he walks; it stays where you turned it. Z-targeting still centres it."},
        {"Gyro Camera", &C::gyro, {{"Off", 0}, {"While aiming", 1}, {"Always", 2}, {"Game's own aiming", 3}},
         "Turning the controller turns the camera, on controllers with a motion sensor (Switch, DualShock 4, DualSense, Steam Deck). While aiming: only with L, R, ZL or ZR held - the bow, the slingshot, first-person view, targeting. Game's own aiming gives the sensor to the game itself, for its Motion-Control Aiming option."},
        {"Gyro Sensitivity", &C::gyro_sens, Percents({10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400}), "100% turns the camera as far as you turn the controller."},
        {"Invert Gyro X", &C::gyro_invert_x, {{"Off", 0}, {"On", 1}}, "Turns the motion sensor's left and right around."},
        {"Invert Gyro Y", &C::gyro_invert_y, {{"Off", 0}, {"On", 1}}, "Turns the motion sensor's up and down around."},
        {"Rumble Strength", &C::rumble, {{"Off", 0}, {"25%", 25}, {"50%", 50}, {"75%", 75}, {"100%", 100}, {"125%", 125}, {"150%", 150}}, "How strongly the controller's motors play the game's rumble."},
        {"Controller in Background", &C::background, {{"Off", 0}, {"On", 1}}, "On: the controller still plays the game while another window has the focus."},
        {"Select + Start: Settings", &C::menu_chord, {{"Off", 0}, {"On", 1}}, "Pressing Select and Start together shows or hides this window. The guide button and F1 or Insert always do."},
    };
    return rows;
}
const std::vector<InputRow>& KeyRows() {
    using C = input::Config;
    static const std::vector<InputRow> rows = {
        {"Mouse Camera", &C::mouse_camera, {{"Off", 0}, {"As the right stick", 1}, {"Direct", 2}},
         "Direct turns the camera by exactly how far the mouse moves, with no top speed and no smoothing; where the game takes the stick for itself (aiming, menus, item wheels) the mouse is the right stick. As the right stick always is. The mouse is the game's while you play: Escape lets it go, a click takes it back."},
        {"Mouse Sensitivity", &C::mouse_sens, Percents({10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500}), "How far the camera turns for a movement of the mouse."},
        {"Invert Mouse X", &C::mouse_invert_x, {{"Off", 0}, {"On", 1}}, "Turns the mouse's left and right around."},
        {"Invert Mouse Y", &C::mouse_invert_y, {{"Off", 0}, {"On", 1}}, "Turns the mouse's up and down around."},
        {"Mouse in Menus", &C::mouse_menus, {{"Off", 0}, {"On", 1}},
         "Over the game's own menus (the title, the pause menu, options, dialogs) the mouse becomes a pointer: move it over a button to choose it, click to confirm, right-click to go back, turn the wheel to move up and down a list. In play it goes back to the camera."},
        {"Mouse Buttons", &C::mouse_buttons, {{"Off", 0}, {"On", 1}}, "Whether the mouse's buttons and wheel press what they are bound to below."},
        {"Walk Mode", &C::walk_toggle, {{"Hold", 0}, {"Toggle", 1}}, "The Walk key slows the movement keys while held, or from one press to the next."},
        {"Walk Speed", &C::walk_speed, Percents({15, 25, 35, 45, 55, 65, 75, 90}), "How far the movement keys push the stick while walking."},
    };
    return rows;
}

template <typename Changed> void ComboRow(int id, const char* label, const std::vector<Choice>& choices, int value, const char* text, const char*& help, Changed&& changed) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableSetColumnIndex(1);
    ImGui::PushID(id);
    ImGui::SetNextItemWidth(-FLT_MIN);
    int at = 0; long distance = 1L << 40;
    for (int i = 0; i < (int)choices.size(); ++i) { const long d = std::labs((long)choices[i].value - value); if (d < distance) { distance = d; at = i; } }
    if (ImGui::BeginCombo("##value", choices[at].name)) {
        for (int v = 0; v < (int)choices.size(); ++v) {
            if (ImGui::Selectable(choices[v].name, v == at) && v != at) changed(choices[v].value);
            if (v == at) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) help = text;
    ImGui::PopID();
}

bool BeginRows(const char* id) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return false;
    ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, 0.45f);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.55f);
    return true;
}

void InputRows(const std::vector<InputRow>& rows, const char* id, const char*& help) {
    if (!BeginRows(id)) return;
    const input::Config current = input::Current();
    for (int r = 0; r < (int)rows.size(); ++r)
        ComboRow(r, rows[r].label, rows[r].choices, current.*rows[r].member, rows[r].help, help, [&](int value) {
            input::Config c = input::Current();
            if (rows[r].member == &input::Config::layout) { if (value != 2) input::ApplyLayout(c, value); }
            else c.*rows[r].member = value;
            input::Set(c);
        });
    ImGui::EndTable();
}

// The bindings of one device class: a row an action, a button a slot. Click (or A) a slot, then press the new input.
void Bindings(int kind, int first, int last, const char*& help) {
    const input::Config current = input::Current();
    if (!ImGui::BeginTable(kind == 1 ? "keys" : "pads", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) return;
    ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthStretch, 0.34f);
    ImGui::TableSetupColumn("first", ImGuiTableColumnFlags_WidthStretch, 0.33f);
    ImGui::TableSetupColumn("second", ImGuiTableColumnFlags_WidthStretch, 0.33f);
    for (int action = first; action < last; ++action) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(input::ActionName(action));
        for (int slot = 0; slot < 2; ++slot) {
            ImGui::TableSetColumnIndex(1 + slot);
            ImGui::PushID(kind * 1000 + action * 2 + slot);
            const bool waiting = capture_kind == kind && capture_action == action && capture_slot == slot;
            const int code = kind == 1 ? current.key[action][slot] : current.pad[action][slot];
            const std::string name = waiting ? std::string(kind == 1 ? "Press a key..." : "Press a button...") : kind == 1 ? input::KeyName(code) : input::PadName(code);
            if (waiting) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.42f, 0.12f, 1.0f));
            if (ImGui::Button(name.c_str(), ImVec2(-FLT_MIN, 0)) && !capture_kind) { capture_action = action; capture_slot = slot; capture_kind = kind; }
            if (waiting) ImGui::PopStyleColor();
            if (ImGui::IsItemHovered() || ImGui::IsItemFocused())
                help = kind == 1 ? "Click, then press the key, mouse button or wheel direction for this control. Escape cancels, Delete clears it."
                                 : "Choose, then press the controller's button, trigger or stick direction for this control. Escape cancels, Delete clears it.";
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
}

void ControllerTab(const char*& help) {
    int count = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&count)) {
        if (!count) ImGui::TextDisabled("No controller connected.");
        for (int i = 0; i < count && i < 4; ++i) ImGui::TextDisabled("Connected: %s", SDL_GetGamepadNameForID(ids[i]));
        SDL_free(ids);
    }
    InputRows(PadRows(), "pad settings", help);
    if (ImGui::CollapsingHeader("Controller Bindings", ImGuiTreeNodeFlags_DefaultOpen)) {
        Bindings(2, 0, input::ButtonCount, help);
        if (ImGui::Button("Reset Controller Bindings")) {
            input::Config c = input::Current(); const input::Config d = input::Defaults();
            std::memcpy(c.pad, d.pad, sizeof c.pad); c.layout = 0; input::Set(c);
        }
    }
}

void KeyboardTab(const char*& help) {
    InputRows(KeyRows(), "key settings", help);
    if (ImGui::CollapsingHeader("Keyboard and Mouse Bindings", ImGuiTreeNodeFlags_DefaultOpen)) {
        Bindings(1, 0, input::ActionCount, help);
        if (ImGui::Button("Reset Keyboard and Mouse Bindings")) {
            input::Config c = input::Current(); const input::Config d = input::Defaults();
            std::memcpy(c.key, d.key, sizeof c.key); input::Set(c);
        }
    }
}

// The rows of gfx::Settings whose section passes `wanted`, under their section headers.
template <typename Wanted> void SettingRows(Wanted&& wanted, const char*& help) {
    const gfx::Settings current = gfx::Current();
    const auto& rows = Rows();
    const char* section = nullptr; bool table = false, section_open = true;
    const auto end_table = [&] { if (table) { ImGui::EndTable(); table = false; } };
    for (int r = 0; r < (int)rows.size(); ++r) {
        if (!wanted(rows[r].section)) continue;
        if (!section || std::strcmp(section, rows[r].section)) {
            end_table(); section = rows[r].section;
            section_open = !*section || ImGui::CollapsingHeader(section, ImGuiTreeNodeFlags_DefaultOpen);
        }
        if (!section_open) continue;
        if (!table) {
            if (!BeginRows(section)) continue;
            table = true;
        }
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(rows[r].label);
        ImGui::TableSetColumnIndex(1);
        ImGui::PushID(r);
        ImGui::SetNextItemWidth(-FLT_MIN);
        const int value = ValueIndex(current, r);
        if (ImGui::BeginCombo("##value", rows[r].choices[value].name)) {
            for (int v = 0; v < (int)rows[r].choices.size(); ++v) {
                if (r == 0 && rows[r].choices[v].value == gfx::PresetCustom && value != v) continue; // (Custom is what changing a setting gives)
                if (ImGui::Selectable(rows[r].choices[v].name, v == value) && v != value) Change(r, v);
                if (v == value) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) help = rows[r].help;
        ImGui::PopID();
    }
    end_table();
}

void Window(bool appeared) {
    const float scale = ImGui::GetStyle().FramePadding.y / 3.0f;
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(display.x * 0.5f, display.y * 0.5f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(780 * scale, display.x * 0.96f), std::min(860 * scale, display.y * 0.94f)), ImGuiCond_FirstUseEver);
    if (appeared) ImGui::SetNextWindowFocus(); // (a controller that opened it moves in it at once)
    bool open = true;
    if (!ImGui::Begin("Settings", &open, ImGuiWindowFlags_NoCollapse)) { ImGui::End(); return; }
    const char* help = nullptr;
    const float footer = ImGui::GetTextLineHeightWithSpacing() * 3.3f + ImGui::GetStyle().ItemSpacing.y * 2;
    if (ImGui::BeginTabBar("tabs")) {
        static const char* first = std::getenv("OOT_SETTINGS_TAB"); // (the tab shown at first, by name: for pictures of it)
        const auto page = [&](const char* name, auto&& body) {
            const bool select = appeared && first && !std::strcmp(first, name);
            if (!ImGui::BeginTabItem(name, nullptr, select ? ImGuiTabItemFlags_SetSelected : 0)) return;
            if (ImGui::BeginChild(name, ImVec2(0, -footer), ImGuiChildFlags_NavFlattened)) body();
            ImGui::EndChild();
            ImGui::EndTabItem();
        };
        page("Graphics", [&] { SettingRows([](const char* section) { return std::strcmp(section, "Game") && std::strcmp(section, "Audio") && std::strcmp(section, "Cheats") && std::strcmp(section, "Hacks"); }, help); });
        page("Game", [&] { SettingRows([](const char* section) { return !std::strcmp(section, "Game") || !std::strcmp(section, "Audio"); }, help); });
        page("Cheats", [&] { SettingRows([](const char* section) { return !std::strcmp(section, "Cheats"); }, help); });
        page("Hacks", [&] { SettingRows([](const char* section) { return !std::strcmp(section, "Hacks"); }, help); });
        page("Controller", [&] { ControllerTab(help); });
        page("Keyboard & Mouse", [&] { KeyboardTab(help); });
        ImGui::EndTabBar();
    }
    ImGui::Separator();
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
    if (restart_notice) ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Restart the game to apply this setting.");
    ImGui::TextDisabled("%s", help ? help : "F1 or Insert, or Select and Start together on a controller: show or hide this window. LB and RB change tabs.");
    ImGui::PopTextWrapPos();
    ImGui::End();
    if (!open) { shown = false; capture_kind = 0; }
}

// The FPS counter: a corner of the picture, no input. (ImGui's own average of the last frames.)
void Counter() {
    const float scale = ImGui::GetStyle().FramePadding.y / 3.0f;
    ImGui::SetNextWindowPos(ImVec2(12 * scale, 12 * scale));
    ImGui::SetNextWindowBgAlpha(0.45f);
    if (ImGui::Begin("##fps", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs)) {
        const float fps = ImGui::GetIO().Framerate;
        // (with frame generation: the rate shown on the display first, then the game's own)
        if (const float shown = nvn::ShownFps(); shown > fps * 1.2f) ImGui::Text("%.0f fps  (game %.0f, %.1f ms)", shown, fps, fps > 0 ? 1000.0f / fps : 0.0f);
        else ImGui::Text("%.0f fps  %.1f ms", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    }
    ImGui::End();
}

VkImageView ViewOf(VkImage image, VkFormat format) {
    for (auto& [i, v] : ui.views) if (i == image) return v;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view; VK_CHECK(vkCreateImageView(Vk().device, &vi, nullptr, &view));
    ui.views.emplace_back(image, view);
    if (ui.views.size() > 16) { vkDestroyImageView(Vk().device, ui.views.front().second, nullptr); ui.views.erase(ui.views.begin()); }
    return view;
}

void Draw(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent);

// The screen shown while pipelines of earlier runs are made before the game starts (prewarm.cpp). True: Skip was clicked.
bool ProgressWindow(const Progress& p) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    const float scale = style.FramePadding.y / 3.0f;
    const ImVec2 size = io.DisplaySize;
    ImDrawList* background = ImGui::GetBackgroundDrawList();
    const ImU32 top = IM_COL32(14, 18, 28, 255), bottom = IM_COL32(6, 8, 12, 255);
    background->AddRectFilledMultiColor(ImVec2(0, 0), size, top, top, bottom, bottom);
    ImGui::SetNextWindowPos(ImVec2(size.x * 0.5f, size.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(760.0f * scale, size.x * 0.9f), 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30 * scale, 26 * scale));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10 * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5 * scale);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.075f, 0.09f, 0.125f, 0.97f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.30f, 0.26f, 0.14f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.90f, 0.72f, 0.28f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.14f, 0.16f, 0.21f, 1.0f));
    bool skip = false;
    if (ImGui::Begin("##prewarm", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::SetWindowFontScale(1.55f);
        ImGui::TextColored(ImVec4(0.95f, 0.82f, 0.45f, 1.0f), "Preparing shaders");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Dummy(ImVec2(0, 6 * scale));
        ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
        ImGui::TextDisabled("Building the graphics pipelines used in earlier sessions, so new areas load without stutter.");
        ImGui::PopTextWrapPos();
        ImGui::Dummy(ImVec2(0, 20 * scale));
        const float fraction = p.total ? std::min(1.0f, (float)p.done / (float)p.total) : 0.0f;
        char percent[16]; std::snprintf(percent, sizeof percent, "%d%%", (int)(fraction * 100.0f));
        ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 28 * scale), percent);
        ImGui::Dummy(ImVec2(0, 6 * scale));
        ImGui::Text("%u / %u shaders", p.done, p.total);
        char eta[48];
        if (p.eta < 0) std::snprintf(eta, sizeof eta, "estimating...");
        else if (p.eta < 1.5) std::snprintf(eta, sizeof eta, "almost done");
        else if (p.eta < 90) std::snprintf(eta, sizeof eta, "about %d s left", (int)(p.eta + 0.5));
        else std::snprintf(eta, sizeof eta, "about %d min left", (int)(p.eta / 60 + 0.5));
        ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize(eta).x);
        ImGui::TextUnformatted(eta);
        ImGui::Dummy(ImVec2(0, 18 * scale));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 10 * scale));
        ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
        ImGui::TextDisabled("Press any key or button to skip. What is left is built while you play.");
        ImGui::PopTextWrapPos();
        ImGui::Dummy(ImVec2(0, 8 * scale));
        const char* label = "Skip";
        const float button = ImGui::CalcTextSize(label).x + style.FramePadding.x * 8;
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - button);
        if (ImGui::Button(label, ImVec2(button, 0))) skip = true;
    }
    ImGui::End();
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(3);
    return skip;
}

} // namespace

void Toggle() { shown = !shown; if (!shown) capture_kind = 0; }
bool Shown() { return shown.load(std::memory_order_relaxed); }

void Event(const SDL_Event& e) {
    if (e.type == SDL_EVENT_KEY_DOWN && (e.key.scancode == SDL_SCANCODE_INSERT || e.key.scancode == SDL_SCANCODE_F1) && !e.key.repeat) { Toggle(); return; }
    if (!shown) return;
    if (const int kind = capture_kind.load()) { // (a binding is being chosen: this input is it, and ImGui does not see it)
        int code = -1;
        if (e.type == SDL_EVENT_KEY_DOWN) {
            if (e.key.scancode == SDL_SCANCODE_ESCAPE) { capture_kind = 0; return; }
            if (e.key.scancode == SDL_SCANCODE_DELETE) code = 0;
            else if (kind == 1) code = (int)e.key.scancode;
        } else if (kind == 1 && e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) code = 0x1000 + e.button.button;
        else if (kind == 1 && e.type == SDL_EVENT_MOUSE_WHEEL && e.wheel.y != 0) code = e.wheel.y > 0 ? input::WheelUp : input::WheelDown;
        else if (kind == 2 && e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) code = 1 + e.gbutton.button;
        else if (kind == 2 && e.type == SDL_EVENT_GAMEPAD_AXIS_MOTION && std::abs((int)e.gaxis.value) > 20000) code = input::PadAxis + e.gaxis.axis * 2 + (e.gaxis.value > 0);
        if (code >= 0) {
            Bind(kind, capture_action, capture_slot, code);
            nav_quiet_until = input::NowNs() + 500'000'000;
            capture_kind = 0;
        }
        if (e.type != SDL_EVENT_MOUSE_MOTION && e.type != SDL_EVENT_WINDOW_RESIZED && e.type != SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) return;
    }
    std::lock_guard lock{events_mutex};
    if (events.size() < 256) events.push_back(e);
}

void FilterInput(u64& buttons, s32& lx, s32& ly, s32& rx, s32& ry, const bool*) {
    static bool blocking = false; // (after the window closes, nothing held then reaches the game until released)
    if (shown) { blocking = true; buttons = 0; lx = ly = rx = ry = 0; return; }
    if (blocking) {
        if (!buttons && !lx && !ly && !rx && !ry) blocking = false;
        else { buttons = 0; lx = ly = rx = ry = 0; }
    }
}

void ReleaseSwapchainViews() {
    for (auto& [image, view] : ui.views) vkDestroyImageView(Vk().device, view, nullptr);
    ui.views.clear();
}

void Record(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent, VkRect2D) {
    const bool counter = gfx::Current().show_fps != 0;
    if (!shown) {
        if (ui.context) { std::lock_guard lock{events_mutex}; events.clear(); }
        if (!counter) return;
    }
    if (!ui.context) CreateContext(extent);
    InitVulkan(format);
    {
        std::deque<SDL_Event> pending;
        { std::lock_guard lock{events_mutex}; pending.swap(events); }
        for (const SDL_Event& e : pending) ImGui_ImplSDL3_ProcessEvent(&e);
    }
    {   // (while a binding is chosen, and just after, the pad does not also move in the window)
        ImGuiIO& io = ImGui::GetIO();
        if (capture_kind || input::NowNs() < nav_quiet_until) io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableGamepad;
        else io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    static bool was_shown = false;
    if (shown) Window(!was_shown);
    was_shown = shown;
    if (counter) Counter();
    ImGui::Render();
    Draw(cmd, dst, format, extent);
}

bool RecordProgress(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent, const Progress& progress) {
    if (!ui.context) CreateContext(extent);
    InitVulkan(format);
    {
        std::deque<SDL_Event> pending;
        { std::lock_guard lock{events_mutex}; pending.swap(events); }
        for (const SDL_Event& e : pending) ImGui_ImplSDL3_ProcessEvent(&e);
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    const bool skip = ProgressWindow(progress);
    ImGui::Render();
    Draw(cmd, dst, format, extent);
    return skip;
}

void ProgressEvent(const SDL_Event& e) {
    std::lock_guard lock{events_mutex};
    if (events.size() < 256) events.push_back(e);
}

namespace {
void Draw(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent) {
    ImDrawData* data = ImGui::GetDrawData();
    if (!data || data->CmdListsCount == 0) return;

    VkImageMemoryBarrier to_color{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_color.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_color.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_color.srcQueueFamilyIndex = to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.image = dst; to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_color);
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = ViewOf(dst, format); attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO}; ri.renderArea = {{0, 0}, extent}; ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &attachment;
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(data, cmd);
    vkCmdEndRendering(cmd);
    VkImageMemoryBarrier back = to_color;
    back.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    back.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    back.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; back.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &back);
}
} // namespace

} // namespace nvn::overlay
