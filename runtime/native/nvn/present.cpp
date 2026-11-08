#include <cmath>
#include "upscale.h"
#include "overlay.h"
#include <cstring>
#include "../graphics_settings.h"
extern "C" volatile int oot_fixed_resolution, oot_dynres_runs;
extern "C" volatile unsigned oot_fixed_scale_bits;
extern "C" volatile unsigned long long oot_dynres_context;
#include <atomic>
#include <algorithm>
// Presentation: the window's textures are drawn into the SDL window's swapchain.

#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <string>
#include <vector>

#include "../../kernel.h"
#include "present.h"
#include "commands.h"
#include "texture.h"
#include "batch.h"

namespace nvn {
namespace {

struct Swapchain {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{1920, 1080};
    std::vector<VkImage> images;
    VkSemaphore acquired = VK_NULL_HANDLE;
    std::vector<VkSemaphore> rendered; // reused only after its swapchain image is acquired again
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    u64 frames = 0;
};

Swapchain sc;
int built_vsync = -1; // (the Graphics screen's VSync the swapchain was built with)

// OOT_PRESENT_DUMP=<dir>: every 300th presented image is written to <dir>/frame-<n>.ppm (and the next one
// after <dir>/snap appears).
struct Dump {
    const char* dir = std::getenv("OOT_PRESENT_DUMP");
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    u8* mapped = nullptr;
    VkDeviceSize size = 0;
} dump;

void EnsureDumpBuffer(VkDeviceSize size) {
    auto& vk = Vk();
    if (dump.size >= size) return;
    if (dump.buffer) {
        vkDestroyBuffer(vk.device, dump.buffer, nullptr);
        vkFreeMemory(vk.device, dump.memory, nullptr);
    }
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &dump.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk.device, dump.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &dump.memory));
    VK_CHECK(vkBindBufferMemory(vk.device, dump.buffer, dump.memory, 0));
    VK_CHECK(vkMapMemory(vk.device, dump.memory, 0, size, 0, reinterpret_cast<void**>(&dump.mapped)));
    dump.size = size;
}

void WriteDump(u64 frame) {
    const std::string path = std::string(dump.dir) + "/frame-" + std::to_string(frame) + ".ppm", partial = path + ".part";
    FILE* f = std::fopen(partial.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", sc.extent.width, sc.extent.height);
    std::vector<u8> row(sc.extent.width * 3);
    for (u32 y = 0; y < sc.extent.height; ++y) {
        const u8* in = dump.mapped + (u64)y * sc.extent.width * 4;
        for (u32 x = 0; x < sc.extent.width; ++x) { // B8G8R8A8
            row[x * 3] = in[x * 4 + 2];
            row[x * 3 + 1] = in[x * 4 + 1];
            row[x * 3 + 2] = in[x * 4];
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
    std::rename(partial.c_str(), path.c_str()); // (complete images only)
}

void Build() {
    auto& vk = Vk();
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical, vk.surface, &caps);
    if (caps.currentExtent.width != 0xffffffffu) sc.extent = caps.currentExtent;
    else { // (Wayland: the application picks the size; the window's size in real pixels)
        int w = 0, h = 0; SDL_GetWindowSizeInPixels(Vk().window, &w, &h);
        if (w > 0 && h > 0) sc.extent = {std::clamp((u32)w, caps.minImageExtent.width, caps.maxImageExtent.width),
                                          std::clamp((u32)h, caps.minImageExtent.height, caps.maxImageExtent.height)};
    }
    if (sc.extent.width == 0 || sc.extent.height == 0) return;
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = vk.surface;
    info.minImageCount = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount) info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
    info.imageFormat = sc.format;
    info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    info.imageExtent = sc.extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // MAILBOX never blocks the game on the compositor (frame pacing comes from the emulated
    // vsync); FIFO, always available, is the fallback.
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    {
        u32 count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical, vk.surface, &count, nullptr);
        std::vector<VkPresentModeKHR> modes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical, vk.surface, &count, modes.data());
        // No vsync: IMMEDIATE when available (OOT_VSYNC=1: MAILBOX/FIFO).
        const bool vsync = std::getenv("OOT_VSYNC") != nullptr || gfx::Current().vsync;
        built_vsync = gfx::Current().vsync;
        for (VkPresentModeKHR m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) info.presentMode = m;
        if (!vsync) for (VkPresentModeKHR m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) info.presentMode = m;
    }
    info.clipped = VK_TRUE;
    info.oldSwapchain = sc.swapchain;
    VkSwapchainKHR next;
    VK_CHECK(vkCreateSwapchainKHR(vk.device, &info, nullptr, &next));
    if (sc.swapchain) vkDestroySwapchainKHR(vk.device, sc.swapchain, nullptr);
    sc.swapchain = next;
    u32 count = 0;
    vkGetSwapchainImagesKHR(vk.device, sc.swapchain, &count, nullptr);
    sc.images.resize(count);
    vkGetSwapchainImagesKHR(vk.device, sc.swapchain, &count, sc.images.data());
    for(auto semaphore:sc.rendered)vkDestroySemaphore(vk.device,semaphore,nullptr);
    sc.rendered.resize(count);
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for(auto& semaphore:sc.rendered)VK_CHECK(vkCreateSemaphore(vk.device,&semaphore_info,nullptr,&semaphore));
}

} // namespace

void CreateSwapchain() {
    auto& vk = Vk();
    if (sc.swapchain) return;
    Build();
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(vk.device, &si, nullptr, &sc.acquired);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(vk.device, &fi, nullptr, &sc.fence);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = vk.command_pool;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(vk.device, &ai, &sc.cmd);
}

// Blits the actual game window texture to the desktop swapchain.
VkImage WindowTextureImage(u64 window, s32 index, VkExtent2D* extent, VkCommandBuffer cmd);

void ReportDrawStats(u64 window_texture);
void NextFrame();
void ReportCompileTime();
void ReportPredicateStats();

double frame_worst_ms = 0;
u32 frames_over_25 = 0, frames_over_50 = 0;
// The game's thread: window events and frame pacing (the blit follows in queue order).
std::atomic<u64> graphics_module{0};
namespace {
// Real-time game speed (from Hookshot, host/nn_vi.c realtime_step). The engine advances each frame
// by 60 / <its graphics module's nominal frame rate, an int at +0x2c8>, so below 60 fps the game
// ran slow (frame skip compensated by rendering only every other frame: half the frames shown).
// Each present now writes the rate of the last 5 frame intervals there (15-60), carrying the
// rounding error to the next frame so no time is lost. OOT_REALTIME=0: off (frame skip instead).
bool RealtimeOn() { static const bool on = !std::getenv("OOT_REALTIME") || std::atoi(std::getenv("OOT_REALTIME")) != 0; return on; }
void RealtimeStep() {
    if (!RealtimeOn()) return;
    const u64 module = graphics_module.load(std::memory_order_relaxed);
    if (!module) return;
    int* slot = reinterpret_cast<int*>(module + 0x2c8);
    if (*slot < 15 || *slot > 60) { static bool told = false; if (!told) { told = true; RT_LOG("real-time step: the graphics module's frame rate reads %d; left alone", *slot); } return; }
    static auto last = std::chrono::steady_clock::now();
    static double samples[5], carry = 0; static unsigned count = 0, next = 0; static u64 steps = 0;
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last).count(); last = now;
    if (dt <= 0 || dt > 0.25) { count = next = 0; carry = 0; return; } // (a load: start again)
    samples[next++ % 5] = dt; if (count < 5) ++count;
    double sum = 0; for (unsigned i = 0; i < count; ++i) sum += samples[i];
    double ratio = std::clamp(sum / count * 60.0 + carry, 1.0, 4.0);
    const int fps = std::clamp((int)(60.0 / ratio + 0.5), 15, gfx::Current().frame_cap <= 30 ? 30 : 60);
    carry = std::clamp(ratio - 60.0 / fps, -0.5, 0.5);
    *slot = fps;
    if (steps++ % 3000 == 0) RT_LOG("real-time step: ratio %.3f, game frame rate %d", ratio, fps);
}
}
void PresentPacing() {
    // (game thread time per frame: its own work since the last call, and the waits in here)
    static auto left = std::chrono::steady_clock::now(); static double work_ms = 0, wait_ms = 0; static u32 calls = 0;
    const auto entered = std::chrono::steady_clock::now();
    work_ms += std::chrono::duration<double, std::milli>(entered - left).count();
    struct Leave { std::chrono::steady_clock::time_point entered; ~Leave() {
        left = std::chrono::steady_clock::now(); wait_ms += std::chrono::duration<double, std::milli>(left - entered).count();
        if (++calls % 300 == 0) { RT_LOG("game thread per frame: %.2f ms own work, %.2f ms waiting in present pacing", work_ms / 300, wait_ms / 300); work_ms = wait_ms = 0; } } } leave{entered};
    rt::CoreRelease release; // waits on the GPU
    { static const char* t = std::getenv("OOT_THROTTLE_MS"); if (t) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(t))); } // diagnostic
    nvn::Submit();
    // Frame pacing as on the console: the GPU may still run this frame while the game prepares
    // the next, but no further. The game double-buffers per-frame data on that assumption;
    // running further ahead overwrites buffers the GPU still reads.
    {
        static u64 previous_frame = 0;
        const u64 this_frame = SubmittedBatches();
        static const bool sync_frames = std::getenv("OOT_SYNC_FRAMES") != nullptr; // diagnostic: no overlap
        WaitForBatches(sync_frames ? this_frame : previous_frame);
        previous_frame = this_frame;
    }
    // 60 frames per second at most when frame skip is on (or OOT_FPS_LIMIT): the game advances
    // one fixed step per frame, so faster frames would run it fast.
    {
        static const bool limit = FrameSkipEnabled() || RealtimeOn() || std::getenv("OOT_FPS_LIMIT") != nullptr; // (the step is at least 1/60 s)
        static auto next = std::chrono::steady_clock::now();
        if (limit) {
            const auto now = std::chrono::steady_clock::now();
            if (next > now) std::this_thread::sleep_until(next);
            next = std::max(next, now - std::chrono::milliseconds(50)) + std::chrono::nanoseconds(gfx::Current().frame_cap <= 30 ? 33333333 : 16666667);
        }
    }
    RealtimeStep();
    {   // Render scale: the engine's fixed-resolution path at the chosen scale (after its dynamic path ran
        // a few frames, which fills fields the 3D reads); 0 keeps the game's own dynamic resolution.
        const int scale = gfx::Current().render_scale;
        const float v = scale / 100.0f; unsigned bits; std::memcpy(&bits, &v, 4);
        if (scale) oot_fixed_scale_bits = bits;
        oot_fixed_resolution = scale > 0 && oot_dynres_runs >= 30;
        if (std::getenv("OOT_DYNRES_DUMP") && oot_dynres_context) { static u64 m = 0; if (m++ % 600 == 300) {
            const u32* w = reinterpret_cast<const u32*>(oot_dynres_context); std::string t;
            for (u32 i = 0; i < 0x200 / 4; ++i) { char b[24]; std::snprintf(b, sizeof b, " %03x:%08x", i * 4, w[i]); t += b; }
            RT_LOG("dynres context %#llx:%s", (unsigned long long)oot_dynres_context, t.c_str()); } }
        { static u64 n = 0; if (n++ % 600 == 0) RT_LOG("render scale: %s (%d%%), engine dynamic-resolution frames %d", oot_fixed_resolution ? "fixed" : "game's dynamic", scale, oot_dynres_runs); }
    }
    {   // The Graphics screen's window size and display mode (on the thread that handles window events).
        static u64 applied = 0;
        if (applied != gfx::Generation()) {
            applied = gfx::Generation();
            const auto& g = gfx::Current();
            static const int sizes[][2] = {{0, 0}, {1280, 720}, {1600, 900}, {1920, 1080}, {2560, 1440}, {3840, 2160}};
            SDL_SetWindowFullscreen(Vk().window, g.display_mode == 1);
            // (sizes are real pixels; SDL takes window coordinates, which a scaled desktop divides)
            const float density = std::max(0.25f, SDL_GetWindowPixelDensity(Vk().window));
            if (g.display_mode == 0 && g.resolution > 0) SDL_SetWindowSize(Vk().window, (int)std::lround(sizes[g.resolution][0] / density), (int)std::lround(sizes[g.resolution][1] / density));
        }
    }
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) { SavePipelineCacheBlocking(); std::_Exit(0); }
        nvn::overlay::Event(e);
    }
}
// In the presenting queue's order: after the frame's rendering was recorded.
void PresentWindowTexture(u64 window, s32 index) {
    rt::CoreRelease release; // waits on the swapchain
    if (nvn::TimelineOn()) RT_LOG("T %.2f present", nvn::TimelineMs());
    nvn::Submit(); // the frame executes before the blit (queue order)
    nvn::last_present_ns = nvn::GuestTimestampNs();
    if (CurrentFrameSkipped()) { NextFrame(); return; } // (frame skip: nothing rendered, nothing shown)
    auto& vk = Vk();
    if (!sc.swapchain) return;
    auto _pt=std::chrono::steady_clock::now();
    vkWaitForFences(vk.device, 1, &sc.fence, VK_TRUE, UINT64_MAX);
    {   // Rebuild for a new VSync choice or window size (Wayland reports no out-of-date swapchain on resize).
        int pw = 0, ph = 0; SDL_GetWindowSizeInPixels(vk.window, &pw, &ph);
        const bool resized = pw > 0 && ph > 0 && ((u32)pw != sc.extent.width || (u32)ph != sc.extent.height);
        if (built_vsync != gfx::Current().vsync || resized) { std::lock_guard lock{vk.queue_mutex}; std::lock_guard copy{vk.copy_mutex}; vkDeviceWaitIdle(vk.device); Build(); }
    }
    { const double _ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_pt).count(); if(_ms>30)RT_LOG("present step previous blit fence: %.1f ms",_ms); _pt=std::chrono::steady_clock::now(); }
    u32 image_index;
    // A hidden or occluded window may not return images for a long time: never block the game
    // (and never hold the queue lock) on that; this frame is simply not shown.
    VkResult r = vkAcquireNextImageKHR(vk.device, sc.swapchain, 100'000'000, sc.acquired, VK_NULL_HANDLE, &image_index);
    { const double _ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_pt).count(); if(_ms>30)RT_LOG("present step acquire: %.1f ms",_ms); _pt=std::chrono::steady_clock::now(); }
    if (r == VK_TIMEOUT || r == VK_NOT_READY) return;
    std::lock_guard lock{vk.queue_mutex};
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        std::lock_guard copy{vk.copy_mutex};
        vkDeviceWaitIdle(vk.device);
        Build();
        if (r == VK_ERROR_OUT_OF_DATE_KHR) return;
    }
    vkResetFences(vk.device, 1, &sc.fence);
    vkResetCommandBuffer(sc.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(sc.cmd, &bi);
    const VkImage dst = sc.images[image_index];
    VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_dst.srcAccessMask = 0;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.image = dst;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(sc.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_dst);
    VkExtent2D src_extent{};
    const VkImage src = WindowTextureImage(window, index, &src_extent, sc.cmd);
    if (src) {
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(s32)src_extent.width, (s32)src_extent.height, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        // Fit the picture into the window at its own aspect ratio (black bars elsewhere).
        s32 w = (s32)sc.extent.width, h = (s32)sc.extent.height;
        if ((u64)w * src_extent.height > (u64)h * src_extent.width) w = (s32)((u64)h * src_extent.width / src_extent.height);
        else h = (s32)((u64)w * src_extent.height / src_extent.width);
        const s32 x0 = ((s32)sc.extent.width - w) / 2, y0 = ((s32)sc.extent.height - h) / 2;
        blit.dstOffsets[0] = {x0, y0, 0};
        blit.dstOffsets[1] = {x0 + w, y0 + h, 1};
        if (WindowOriginLowerLeft()) std::swap(blit.dstOffsets[0].y, blit.dstOffsets[1].y); // row 0 is the bottom
        VkClearColorValue black{{0, 0, 0, 1.0f}};
        vkCmdClearColorImage(sc.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &to_dst.subresourceRange);
        VkMemoryBarrier cleared{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        cleared.srcAccessMask = cleared.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(sc.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cleared, 0, nullptr, 0, nullptr);
        if (!nvn::UpscaleToWindow(sc.cmd, src, src_extent, WindowOriginLowerLeft(), dst, VkRect2D{{x0, y0}, {(u32)w, (u32)h}}))
            vkCmdBlitImage(sc.cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &blit, VK_FILTER_LINEAR);
        {   // The Graphics screen over the game's picture (overlay.cpp).
            VkMemoryBarrier blitted{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            blitted.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; blitted.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(sc.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &blitted, 0, nullptr, 0, nullptr);
            nvn::overlay::Record(sc.cmd, dst, sc.format, sc.extent, VkRect2D{{x0, y0}, {(u32)w, (u32)h}});
        }
    } else {
        VkClearColorValue color{{0, 0, 0, 1.0f}};
        vkCmdClearColorImage(sc.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &to_dst.subresourceRange);
    }
    // (also on request: a file named "snap" in the directory takes the next image and is removed)
    const bool dumping = dump.dir && ((sc.frames + 1) % 300 == 0 || std::remove((std::string(dump.dir) + "/snap").c_str()) == 0);
    VkImageLayout last = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    if (dumping) {
        EnsureDumpBuffer((VkDeviceSize)sc.extent.width * sc.extent.height * 4);
        VkImageMemoryBarrier to_src = to_dst;
        to_src.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(sc.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_src);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {sc.extent.width, sc.extent.height, 1};
        vkCmdCopyImageToBuffer(sc.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dump.buffer, 1, &copy);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    VkImageMemoryBarrier to_present = to_dst;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = last;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(sc.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_present);
    vkEndCommandBuffer(sc.cmd);
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &sc.acquired;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &sc.cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &sc.rendered[image_index];
    VK_CHECK(vkQueueSubmit(vk.queue, 1, &submit, sc.fence));
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &sc.rendered[image_index];
    present.swapchainCount = 1;
    present.pSwapchains = &sc.swapchain;
    present.pImageIndices = &image_index;
    r = vkQueuePresentKHR(vk.queue, &present);
    { const double _ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_pt).count(); if(_ms>30)RT_LOG("present step submit+present: %.1f ms",_ms); _pt=std::chrono::steady_clock::now(); }

    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        std::lock_guard copy{vk.copy_mutex};
        vkDeviceWaitIdle(vk.device);
        Build();
    }
    static auto last_report = std::chrono::steady_clock::now();
    static u64 last_flushes = FlushCount();
    ++sc.frames;
    NextFrame();
    {   // frame pacing: worst frame and stutters per report window
        static auto previous = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - previous).count();
        previous = now;
        frame_worst_ms = std::max(frame_worst_ms, ms);
        ReportSlowFrame(ms);
        if (ms > 25) ++frames_over_25;
        if (ms > 50) ++frames_over_50;
    }
    if (dumping) {
        vkWaitForFences(vk.device, 1, &sc.fence, VK_TRUE, UINT64_MAX);
        WriteDump(sc.frames);
    }
    if (sc.frames % 300 == 0) ReportDrawStats((*reinterpret_cast<std::vector<u64>**>(window + 0x78))->at(index));
    if (sc.frames % 300 == 0) {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last_report).count();
        const u64 flushes = FlushCount();
        static u64 last_gpu = GpuBusyNanoseconds(), last_wait = FenceWaitNanoseconds();
        const u64 gpu = GpuBusyNanoseconds(), wait = FenceWaitNanoseconds();
        RT_LOG("presented %llu frames (%.1f fps, %.1f GPU submissions, %.2f ms GPU busy, %.2f ms waiting per frame; worst %.0f ms, %u over 25 ms, %u over 50 ms)", (unsigned long long)sc.frames,
               300.0 / seconds, (flushes - last_flushes) / 300.0, (gpu - last_gpu) / 300e6, (wait - last_wait) / 300e6, frame_worst_ms, frames_over_25, frames_over_50);
        frame_worst_ms = 0; frames_over_25 = frames_over_50 = 0;
        last_gpu = gpu; last_wait = wait;
        last_report = now;
        last_flushes = flushes;
        SavePipelineCache(); // persists pipelines built since the last save
        ReportCompileTime();
        ReportPredicateStats();
        ReportBatchStats();
        ReportUniformOverlays();
        { static u64 last_wb = WritebackNanoseconds(); const u64 wb = WritebackNanoseconds(); RT_LOG("GPU write-back copies: %.2f ms per frame", (wb - last_wb) / 300e6); last_wb = wb; }
        RT_LOG("texture uploads: %llu (%.1f MiB)", (unsigned long long)upload_stats.count, upload_stats.bytes / 1048576.0);
        ReportImageCache();
        upload_stats = {};
    }
}

} // namespace nvn
