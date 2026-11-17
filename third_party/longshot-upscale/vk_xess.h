// (longshot) Intel XeSS Super Resolution, one of the temporal upscaler's engines (vk_temporal.h; vk_xess.cpp), through XeSS's
// Vulkan API. Only Intel's MIT API headers are built in (third_party/xess); Intel's libxess.dll is the player's (put there by
// hand, or downloaded on their PC by a build with --fetch-xess-runtime: scripts/xess_sdk.py), loaded at run time from
// RX_VK_XESS_DIR, the executable's folder or the game folder, never shipped with Longshot. Windows only: Intel makes
// no XeSS library for Linux, where this says so and the game's own upscaler runs. XeSS Frame Generation and XeLL (Intel's
// latency reduction) are D3D12 only, so they are not here.
#pragma once

#include <vector>

#include "vk_internal.h"

namespace mkwvk {

struct TemporalEngine;

/// Whether this build can run XeSS at all (Windows: Intel's headers are built in; elsewhere false)
bool xess_built();
/// Why Intel XeSS cannot be chosen here (the Upscaler row says it), XESS_OK when it can or is not known yet.
/// NOT_BUILT: not Windows (no XeSS library exists there); ENV_OFF: RX_VK_XESS=0; NO_LIBRARY: no libxess.dll where it is looked
/// for, or one without XeSS's Vulkan entry points; NO_VULKAN: the Vulkan instance or device lacks what XeSS needs (an extension
/// or a feature it asked for: Vulkan 1.1, shaderStorageImageWriteWithoutFormat, VK_EXT_mutable_descriptor_type);
/// UNSUPPORTED: Intel's library made no context for this GPU or driver
enum XessWhy { XESS_OK, XESS_NOT_BUILT, XESS_ENV_OFF, XESS_NO_LIBRARY, XESS_NO_VULKAN, XESS_UNSUPPORTED, XESS_WHY_COUNT };
XessWhy xess_why();
/// Intel's library is there and loaded (before the device is made: the Quality preset's Auto (this PC) and the device's
/// extensions ask)
bool xess_library_loaded();
/// Makes a context once to ask Intel's library whether XeSS runs on this device, so xess_why knows before the first upscale
/// (the menu, when it opens). Nothing without the library, or before the device is made
void xess_probe();
/// The instance's extensions XeSS asks for (xessVKGetRequiredInstanceExtensions), added to `want` when the loader has them
/// (all: what it has); api: the instance's Vulkan version. Loads Intel's library first (once). One the loader lacks: XeSS
/// is out for the run (NO_VULKAN), the instance made without it
void xess_instance_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all, uint32_t api);
/// The device's extensions XeSS asks for (xessVKGetRequiredDeviceExtensions), added to `want` when the device has them (all:
/// what it has); one it lacks: XeSS out for the run
void xess_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all);
/// The device's features XeSS asks for (xessVKGetRequiredDeviceFeatures), set in the chain at *first (the
/// VkPhysicalDeviceFeatures2 the device is made with: its own structures are written to, the ones it lacks added by the
/// library), right before vkCreateDevice. Nothing when the extensions were not all there. A failure leaves XeSS out for the
/// run (NO_VULKAN) and the chain as it was, ending at *tail
void xess_device_features(void** first, void** tail);
/// XeSS as the temporal upscaler's engine (Intel XeSS and its Native AA on the Upscaler row); nullptr where it is not built
const TemporalEngine* xess_engine();

}  // namespace mkwvk
