// The graphics settings window: Dear ImGui (third_party/imgui) drawn over the presented frame, in the
// game's own font when it can be read. Insert shows or hides it; while shown it takes the mouse,
// keyboard and gamepad (the game sees no input). Nothing is set up until it is first shown.
#include "overlay.h"

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
struct Row { const char* label; std::vector<const char*> values; const char* help; };
const std::vector<Row>& Rows() {
    static const std::vector<Row> rows = {
        {"Preset", {"Original", "Low", "Medium", "High", "Ultra", "Custom"}, "Original is the game as it ships. Low and Medium use the handheld profile."},
        {"Window Size", {"Default", "1280 x 720", "1600 x 900", "1920 x 1080", "2560 x 1440", "3840 x 2160"}, "The size of the game's window on the desktop."},
        {"Display Mode", {"Windowed", "Fullscreen"}, "Fullscreen fills the display the window is on."},
        {"Quality Profile", {"Console", "Handheld"}, "Handheld renders at the game's lower handheld resolution. Takes effect after a restart."},
        {"Frame Rate Cap", {"30", "60"}, "30 halves the work per second; the game keeps its normal speed."},
        {"VSync", {"Off", "On"}, "On waits for the display: no tearing, a little more input delay."},
        {"Texture Filtering", {"Original", "Bilinear", "Trilinear", "Anisotropic 2x", "Anisotropic 4x", "Anisotropic 8x", "Anisotropic 16x"}, "Sharper textures at an angle with anisotropic filtering."},
        {"Texture Detail", {"Original", "Low", "Medium", "High", "Ultra"}, "Lower detail uses smaller mipmaps: less memory bandwidth."},
        {"Texture Memory", {"Original", "Low", "High Quality"}, "Low compresses more textures (less video memory). Applies to newly loaded areas."},
        {"Upscaling", {"Linear", "Catmull-Rom", "Lanczos", "FSR 1"}, "How the game's 1080p picture is scaled to a larger window."},
        {"Sharpness", {"Off", "25%", "50%", "75%", "100%"}, "Sharpens the picture as it goes to the window, without halos."},
    };
    return rows;
}
int* Field(gfx::Settings& s, int r) {
    switch (r) {
    case 0: return &s.preset; case 1: return &s.resolution; case 2: return &s.display_mode; case 3: return &s.handheld;
    case 5: return &s.vsync; case 6: return &s.filtering; case 7: return &s.texture_detail; case 8: return &s.texture_memory;
    case 9: return &s.upscaling;
    default: return nullptr;
    }
}
int ValueIndex(const gfx::Settings& s, int r) {
    if (r == 4) return s.frame_cap <= 30 ? 0 : 1;
    if (r == 10) return s.sharpness / 25;
    gfx::Settings copy = s; return *Field(copy, r);
}
void Change(int r, int delta) { // (delta from the current value)
    gfx::Settings s = gfx::Current();
    const int n = (int)Rows()[r].values.size();
    const int v = ((ValueIndex(s, r) + delta) % n + n) % n;
    if (r == 0) {
        if (v == gfx::PresetCustom) return; // (Custom is what changing a single setting gives)
        s = gfx::PresetValues(v);
    } else {
        if (r == 4) s.frame_cap = v == 0 ? 30 : 60; else if (r == 10) s.sharpness = v * 25; else *Field(s, r) = v;
        s.preset = gfx::PresetCustom;
    }
    if (r == 3 || (r == 0 && s.handheld != gfx::Current().handheld)) restart_notice = true;
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
    RT_LOG("graphics window: ready (Insert)");
}

void Window() {
    const float scale = ImGui::GetStyle().FramePadding.y / 3.0f;
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(620 * scale, 0), ImGuiCond_FirstUseEver);
    bool open = true;
    if (!ImGui::Begin("Graphics", &open, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) { ImGui::End(); return; }
    const gfx::Settings current = gfx::Current();
    const auto& rows = Rows();
    const char* help = nullptr;
    if (ImGui::BeginTable("settings", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, 0.45f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.55f);
        for (int r = 0; r < (int)rows.size(); ++r) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(rows[r].label);
            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(r);
            ImGui::SetNextItemWidth(-FLT_MIN);
            const int value = ValueIndex(current, r);
            const char* preview = value >= 0 && value < (int)rows[r].values.size() ? rows[r].values[value] : "?";
            if (ImGui::BeginCombo("##value", preview)) {
                for (int v = 0; v < (int)rows[r].values.size(); ++v) {
                    if (r == 0 && v == gfx::PresetCustom && value != v) continue; // (Custom is what changing a setting gives)
                    if (ImGui::Selectable(rows[r].values[v], v == value) && v != value) Change(r, v - value);
                    if (v == value) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) help = rows[r].help;
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Separator();
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
    if (restart_notice) ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Restart the game to apply the new quality profile.");
    ImGui::TextDisabled("%s", help ? help : "Insert: show or hide this window.");
    ImGui::PopTextWrapPos();
    ImGui::TextDisabled("%.0f fps", ImGui::GetIO().Framerate);
    ImGui::End();
    if (!open) shown = false;
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

} // namespace

void Event(const SDL_Event& e) {
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_INSERT && !e.key.repeat) { shown = !shown; return; }
    if (!shown) return;
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

void Record(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent, VkRect2D) {
    if (!shown) {
        if (ui.context) { std::lock_guard lock{events_mutex}; events.clear(); }
        return;
    }
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
    Window();
    ImGui::Render();
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

} // namespace nvn::overlay
