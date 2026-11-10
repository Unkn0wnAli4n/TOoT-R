// The graphics settings window (Dear ImGui), drawn over the presented frame. Insert shows or hides it.
#pragma once
#include <SDL3/SDL.h>

#include "vk.h"

namespace nvn::overlay {

// The window's SDL events (the thread that polls them): Insert toggles the window; while it is
// shown, mouse and keyboard events go to it.
void Event(const SDL_Event& event);

// HID thread: while the window is shown it takes the player's input (the game sees nothing pressed).
void FilterInput(u64& buttons, s32& lx, s32& ly, s32& rx, s32& ry, const bool* keys);

// The pre-game progress screen (prewarm.cpp): what to show, drawn over the whole window (TRANSFER_DST_OPTIMAL on entry and exit);
// true when its Skip button was clicked. Events (ProgressEvent) go to ImGui on the next call.
struct Progress { u32 done = 0, total = 0; double eta = -1; };
bool RecordProgress(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent, const Progress& progress);
void ProgressEvent(const SDL_Event& event);

// Present: after the game's frame was blitted into dst (TRANSFER_DST_OPTIMAL on entry and exit).
void Record(VkCommandBuffer cmd, VkImage dst, VkFormat format, VkExtent2D extent, VkRect2D game);

// Called with presentation idle, before replacing its swapchain images.
void ReleaseSwapchainViews();

} // namespace nvn::overlay
