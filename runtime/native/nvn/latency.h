// Low latency (the Display tab's Low Latency row): NVIDIA Reflex through VK_NV_low_latency2, AMD Anti-Lag 2 through
// VK_AMD_anti_lag, whichever the GPU has. The driver delays the start of a frame's work so that it reaches the display
// with the least waiting in a queue: less time from a button to the picture, most while the GPU is the limit.
#pragma once
#include <vector>
#include <vulkan/vulkan.h>

namespace nvn::latency {
void DeviceExtensions(VkPhysicalDevice physical, std::vector<const char*>& extensions, const std::vector<VkExtensionProperties>& available, VkPhysicalDeviceFeatures2& features); // (before the device is made)
bool Available();
const void* SwapchainInfo(const void* next); // the swapchain's create info chain with latency mode, when the row is on
void SwapchainBuilt(VkSwapchainKHR swapchain);
bool SwapchainStale();                       // the row went on or off since the swapchain was built
void FrameSubmitting();                      // the game's frame is about to be submitted (simulation over)
void Submitted();
const void* PresentInfo(const void* next);   // the present's chain with this frame's id
void Presenting(bool begin);
void FrameDone();                            // after the present: waits until the driver wants the next frame begun
} // namespace nvn::latency
