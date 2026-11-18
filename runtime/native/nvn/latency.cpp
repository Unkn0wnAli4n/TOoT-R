#include "latency.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "../../rt.h"
#include "../graphics_settings.h"
#include "vk.h"

namespace nvn::latency {
namespace {
struct State {
    bool reflex = false, antilag = false;
    VkPhysicalDevicePresentIdFeaturesKHR present_id_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
    VkPhysicalDeviceAntiLagFeaturesAMD antilag_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD};
    VkSwapchainLatencyCreateInfoNV swapchain_info{VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV};
    VkPresentIdKHR present_id{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
    PFN_vkSetLatencySleepModeNV set_mode = nullptr; PFN_vkLatencySleepNV sleep = nullptr; PFN_vkSetLatencyMarkerNV marker = nullptr;
    PFN_vkAntiLagUpdateAMD antilag_update = nullptr;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    bool swapchain_latency = false; // (the swapchain was made with latency mode)
    int mode = -1;                  // (the sleep mode set on it)
    VkSemaphore wake = VK_NULL_HANDLE; u64 wake_value = 0;
    u64 frame = 1;                  // (the frame being made: its present id)
    bool marked = false;            // (this frame's start markers were given)
    int antilag_mode = 0;
} s;

// The row: 0 off, 1 on, 2 on + boost (Reflex's boost: the GPU kept at full clock). OOT_LOW_LATENCY=<n> for a run. Not
// with frame generation: its frames are shown by another thread at its own pace.
int Wanted() {
    static const char* env = std::getenv("OOT_LOW_LATENCY");
    if (gfx::Current().frame_gen > 1) return 0;
    return std::clamp(env && *env ? std::atoi(env) : gfx::Current().low_latency, 0, 2);
}
bool ReflexOn() { return s.reflex && s.swapchain && s.swapchain_latency && Wanted() > 0; }
void Mark(VkLatencyMarkerNV m) {
    VkSetLatencyMarkerInfoNV i{VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV};
    i.presentID = s.frame; i.marker = m;
    s.marker(Vk().device, s.swapchain, &i);
}
void AntiLag(VkAntiLagStageAMD stage) {
    const int want = s.antilag ? (Wanted() > 0 ? 1 : 0) : 0;
    if (!want && !s.antilag_mode) return;
    VkAntiLagPresentationInfoAMD p{VK_STRUCTURE_TYPE_ANTI_LAG_PRESENTATION_INFO_AMD};
    p.stage = stage; p.frameIndex = s.frame;
    VkAntiLagDataAMD d{VK_STRUCTURE_TYPE_ANTI_LAG_DATA_AMD};
    d.mode = want ? VK_ANTI_LAG_MODE_ON_AMD : VK_ANTI_LAG_MODE_OFF_AMD; d.maxFPS = 0; d.pPresentationInfo = want ? &p : nullptr;
    s.antilag_update(Vk().device, &d);
    if (want != s.antilag_mode) { s.antilag_mode = want; RT_LOG("low latency: AMD Anti-Lag 2 %s", want ? "on" : "off"); }
}
} // namespace

void DeviceExtensions(VkPhysicalDevice physical, std::vector<const char*>& extensions, const std::vector<VkExtensionProperties>& available, VkPhysicalDeviceFeatures2& features) {
    static const char* env = std::getenv("OOT_LOW_LATENCY");
    if (env && !std::strcmp(env, "-")) return; // (the extensions not even enabled)
    const auto has = [&](const char* name) { for (const auto& e : available) if (!std::strcmp(e.extensionName, name)) return true; return false; };
    const auto add = [&](const char* name) { for (const char* e : extensions) if (!std::strcmp(e, name)) return; extensions.push_back(name); };
    if (has(VK_NV_LOW_LATENCY_2_EXTENSION_NAME) && has(VK_KHR_PRESENT_ID_EXTENSION_NAME)) {
        VkPhysicalDevicePresentIdFeaturesKHR have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
        vkGetPhysicalDeviceFeatures2(physical, &query);
        if (have.presentId) {
            add(VK_NV_LOW_LATENCY_2_EXTENSION_NAME); add(VK_KHR_PRESENT_ID_EXTENSION_NAME);
            // (a present id feature already in the chain - frame generation's - serves both)
            bool chained = false;
            for (VkBaseOutStructure* n = reinterpret_cast<VkBaseOutStructure*>(features.pNext); n; n = n->pNext)
                if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR) { reinterpret_cast<VkPhysicalDevicePresentIdFeaturesKHR*>(n)->presentId = VK_TRUE; chained = true; }
            if (!chained) { s.present_id_feature.presentId = VK_TRUE; s.present_id_feature.pNext = features.pNext; features.pNext = &s.present_id_feature; }
            s.reflex = true;
        }
    } else if (has(VK_AMD_ANTI_LAG_EXTENSION_NAME)) {
        VkPhysicalDeviceAntiLagFeaturesAMD have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
        vkGetPhysicalDeviceFeatures2(physical, &query);
        if (have.antiLag) {
            add(VK_AMD_ANTI_LAG_EXTENSION_NAME);
            s.antilag_feature.antiLag = VK_TRUE; s.antilag_feature.pNext = features.pNext; features.pNext = &s.antilag_feature;
            s.antilag = true;
        }
    }
    RT_LOG("low latency: %s", s.reflex ? "NVIDIA Reflex can run on this GPU" : s.antilag ? "AMD Anti-Lag 2 can run on this GPU" : "this GPU's driver has neither NVIDIA Reflex nor AMD Anti-Lag 2");
}
bool Available() { return s.reflex || s.antilag; }

const void* SwapchainInfo(const void* next) {
    if (!s.reflex || Wanted() == 0) return next;
    s.swapchain_info.latencyModeEnable = VK_TRUE; s.swapchain_info.pNext = next;
    return &s.swapchain_info;
}
void SwapchainBuilt(VkSwapchainKHR swapchain) {
    s.swapchain = swapchain; s.swapchain_latency = s.reflex && Wanted() > 0; s.mode = -1; s.marked = false;
    if (!s.reflex) { if (s.antilag && !s.antilag_update) s.antilag_update = reinterpret_cast<PFN_vkAntiLagUpdateAMD>(vkGetDeviceProcAddr(Vk().device, "vkAntiLagUpdateAMD")); if (!s.antilag_update) s.antilag = false; return; }
    if (!s.set_mode) {
        const VkDevice d = Vk().device;
        s.set_mode = reinterpret_cast<PFN_vkSetLatencySleepModeNV>(vkGetDeviceProcAddr(d, "vkSetLatencySleepModeNV"));
        s.sleep = reinterpret_cast<PFN_vkLatencySleepNV>(vkGetDeviceProcAddr(d, "vkLatencySleepNV"));
        s.marker = reinterpret_cast<PFN_vkSetLatencyMarkerNV>(vkGetDeviceProcAddr(d, "vkSetLatencyMarkerNV"));
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO}; type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; info.pNext = &type;
        if (!s.set_mode || !s.sleep || !s.marker || vkCreateSemaphore(d, &info, nullptr, &s.wake) != VK_SUCCESS) { s.reflex = false; RT_LOG("low latency: NVIDIA Reflex's functions are not there: off"); return; }
    }
}
bool SwapchainStale() { return s.reflex && s.swapchain && s.swapchain_latency != (Wanted() > 0); }

void FrameSubmitting() {
    if (s.antilag) return;
    if (!ReflexOn()) return;
    if (const int want = Wanted(); want != s.mode) {
        VkLatencySleepModeInfoNV m{VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV};
        m.lowLatencyMode = VK_TRUE; m.lowLatencyBoost = want == 2; m.minimumIntervalUs = 0;
        const VkResult r = s.set_mode(Vk().device, s.swapchain, &m);
        RT_LOG("low latency: NVIDIA Reflex %s%s", want == 2 ? "on + boost" : "on", r == VK_SUCCESS ? "" : " (the driver refused the sleep mode)");
        s.mode = want;
    }
    if (!s.marked) return; // (the first frame after the row went on has no start: its markers begin with the next)
    Mark(VK_LATENCY_MARKER_SIMULATION_END_NV); Mark(VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
}
void Submitted() { if (ReflexOn() && s.marked) Mark(VK_LATENCY_MARKER_RENDERSUBMIT_END_NV); }
const void* PresentInfo(const void* next) {
    if (!ReflexOn() || !s.marked) return next;
    s.present_id.swapchainCount = 1; s.present_id.pPresentIds = &s.frame; s.present_id.pNext = next;
    return &s.present_id;
}
void Presenting(bool begin) {
    if (s.antilag) { if (begin && s.antilag_update) AntiLag(VK_ANTI_LAG_STAGE_PRESENT_AMD); return; }
    if (ReflexOn() && s.marked) Mark(begin ? VK_LATENCY_MARKER_PRESENT_START_NV : VK_LATENCY_MARKER_PRESENT_END_NV);
}
void FrameDone() {
    if (s.antilag) { if (s.antilag_update) { ++s.frame; AntiLag(VK_ANTI_LAG_STAGE_INPUT_AMD); } return; }
    if (!ReflexOn()) { s.marked = false; return; }
    ++s.frame;
    VkLatencySleepInfoNV info{VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV};
    info.signalSemaphore = s.wake; info.value = ++s.wake_value;
    if (s.sleep(Vk().device, s.swapchain, &info) == VK_SUCCESS) {
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1; wait.pSemaphores = &s.wake; wait.pValues = &s.wake_value;
        vkWaitSemaphores(Vk().device, &wait, 100'000'000ull); // (never longer than a tenth of a second)
    }
    Mark(VK_LATENCY_MARKER_SIMULATION_START_NV); Mark(VK_LATENCY_MARKER_INPUT_SAMPLE_NV);
    s.marked = true;
}
} // namespace nvn::latency
