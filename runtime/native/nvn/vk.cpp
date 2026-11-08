#include "../graphics_settings.h"
#include <chrono>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <string>
#include <cstring>
#include <vector>

#include <SDL3/SDL_vulkan.h>

#include "vk.h"

namespace nvn {
namespace {

void CreateInstance(VulkanContext& vk) {
    Uint32 count = 0;
    const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!sdl_extensions) rt::Fatal("SDL_Vulkan_GetInstanceExtensions: %s", SDL_GetError());
    std::vector<const char*> extensions(sdl_extensions, sdl_extensions + count);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "tootr";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = (u32)extensions.size();
    info.ppEnabledExtensionNames = extensions.data();
    std::vector<const char*> layers;
    if (std::getenv("OOT_VK_VALIDATION")) layers.push_back("VK_LAYER_KHRONOS_validation");
    info.enabledLayerCount = (u32)layers.size();
    info.ppEnabledLayerNames = layers.data();
    VK_CHECK(vkCreateInstance(&info, nullptr, &vk.instance));
}

void PickDevice(VulkanContext& vk) {
    u32 count = 0;
    vkEnumeratePhysicalDevices(vk.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(vk.instance, &count, devices.data());
    int best_score = -1;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score > best_score) {
            best_score = score;
            vk.physical = d;
            vk.properties = p;
        }
    }
    if (!vk.physical) rt::Fatal("no Vulkan device");
    vkGetPhysicalDeviceMemoryProperties(vk.physical, &vk.memory_properties);
    u32 qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> families(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &qcount, families.data());
    for (u32 i = 0; i < qcount; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(vk.physical, i, vk.surface, &present);
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
            vk.queue_family = i;
            vk.queue_count = families[i].queueCount;
            break;
        }
    }
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    host.pNext = &subgroup;
    props2.pNext = &host;
    vkGetPhysicalDeviceProperties2(vk.physical, &props2);
    vk.host_import_alignment = host.minImportedHostPointerAlignment;
    // The shader translator emits subgroup operations in every stage, task and mesh included
    // (shader_compiler.cpp supported_subgroup_stages).
    const VkShaderStageFlags needed = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    if ((subgroup.supportedStages & needed) != needed || subgroup.subgroupSize != 32)
        RT_LOG("warning: subgroup operations in stages %#x (size %u); task/mesh programs may compute wrong warp results",
               subgroup.supportedStages, subgroup.subgroupSize);
    RT_LOG("Vulkan device: %s (host import alignment %#llx)", vk.properties.deviceName,
           (unsigned long long)vk.host_import_alignment);
}

void CreateDevice(VulkanContext& vk) {
    const float priorities[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = vk.queue_family;
    queue.queueCount = vk.queue_count >= 2 ? 2 : 1;
    queue.pQueuePriorities = priorities;
    std::vector<const char*> extensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
        VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME, // NVN's UNSIGNED_BYTE index type
        VK_EXT_MESH_SHADER_EXTENSION_NAME,      // NVN2 task/mesh programs
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,     // NVN unbound image handles read zero
    };
    u32 extension_count=0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(vk.physical,nullptr,&extension_count,nullptr));
    std::vector<VkExtensionProperties> available(extension_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(vk.physical,nullptr,&extension_count,available.data()));
#ifdef __linux__
    {   // (memory pools backed by exported GPU memory mapped in place: memory.cpp, AMD)
        bool fd = false, dma = false;
        for (const auto& e : available) { fd |= !std::strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME); dma |= !std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME); }
        if (fd && dma) { extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME); extensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME); vk.dma_buf = true; }
    }
#endif
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    for(const auto& extension:available)if(!std::strcmp(extension.extensionName,VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)){
        VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};supported.pNext=&atomic_float;
        vkGetPhysicalDeviceFeatures2(vk.physical,&supported);
        vk.buffer_float32_atomic_add=atomic_float.shaderBufferFloat32AtomicAdd;
        break;
    }
    atomic_float={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    atomic_float.shaderBufferFloat32AtomicAdd=vk.buffer_float32_atomic_add;
    if(vk.buffer_float32_atomic_add)extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    RT_LOG("GPU buffer float32 atomic add: %s",vk.buffer_float32_atomic_add?"hardware":"CAS fallback");
    VkPhysicalDeviceRobustness2FeaturesEXT robustness{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    if(vk.buffer_float32_atomic_add)robustness.pNext=&atomic_float;
    robustness.nullDescriptor=VK_TRUE;
    // Bounds-checked buffer access: reads past a bound range return zero. NVIDIA pays heavily for it (Hookshot measured
    // 50%+ lower frame rates there: their issues 394, 418, 419), so it is on elsewhere only. OOT_ROBUST=0/1 forces it.
    vk.robust = std::getenv("OOT_ROBUST") ? std::atoi(std::getenv("OOT_ROBUST")) != 0 : vk.properties.vendorID != 0x10DE;
    RT_LOG("robust buffer access: %s", vk.robust ? "on" : "off");
    robustness.robustBufferAccess2=vk.robust;
    robustness.robustImageAccess2=vk.robust;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shaders{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    mesh_shaders.pNext=&robustness;
    mesh_shaders.taskShader = VK_TRUE;
    mesh_shaders.meshShader = VK_TRUE;
    {   // (draw-log statistics of task/mesh invocations, where the driver has them: RADV does not)
        VkPhysicalDeviceMeshShaderFeaturesEXT supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &supported;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        mesh_shaders.meshShaderQueries = supported.meshShaderQueries;
        vk.mesh_shader_queries = supported.meshShaderQueries;
    }
    VkPhysicalDeviceIndexTypeUint8FeaturesKHR uint8_indices{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR};
    uint8_indices.indexTypeUint8 = VK_TRUE;
    uint8_indices.pNext = &mesh_shaders;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.pNext = &uint8_indices;
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;
    v13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.pNext = &v13;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.hostQueryReset = VK_TRUE; // occlusion query ring reset after each batch (render.cpp)
    v12.timelineSemaphore = VK_TRUE;
    v12.descriptorIndexing = VK_TRUE;
    v12.runtimeDescriptorArray = VK_TRUE;
    v12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    v12.descriptorBindingPartiallyBound = VK_TRUE;
    v12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    v12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    v12.shaderFloat16 = VK_TRUE;
    v12.shaderInt8 = VK_TRUE;
    v12.storageBuffer8BitAccess = VK_TRUE;
    v12.scalarBlockLayout = VK_TRUE;
    v12.samplerMirrorClampToEdge = VK_TRUE;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.pNext = &v12;
    v11.storageBuffer16BitAccess = VK_TRUE;
    v11.shaderDrawParameters = VK_TRUE;
    v11.variablePointersStorageBuffer = VK_TRUE;
    v11.variablePointers = VK_TRUE;
    // GPU-evaluated NVN render predicates (occlusion results never wait on the CPU).
    VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    // Push descriptors: per-draw descriptor sets without allocating and updating one each draw.
    bool push_descriptors=false;
    for(const auto& extension:available)if(!std::strcmp(extension.extensionName,VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)){push_descriptors=true;break;}
    if(push_descriptors&&!std::getenv("OOT_NO_PUSH_DESCRIPTORS"))extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);else push_descriptors=false;
    // Fault details on device loss (VK_EXT_device_fault).
    VkPhysicalDeviceFaultFeaturesEXT fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    for(const auto& extension:available)if(!std::strcmp(extension.extensionName,VK_EXT_DEVICE_FAULT_EXTENSION_NAME)){vk.device_fault=true;break;}
    if(vk.device_fault){extensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);fault.deviceFault=VK_TRUE;fault.pNext=v11.pNext;v11.pNext=&fault;}
    conditional.pNext = &v11;
    for(const auto& extension:available)if(!std::strcmp(extension.extensionName,VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)){
        VkPhysicalDeviceConditionalRenderingFeaturesEXT supported_conditional{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};supported.pNext=&supported_conditional;
        vkGetPhysicalDeviceFeatures2(vk.physical,&supported);
        vk.conditional_rendering=supported_conditional.conditionalRendering;
    }
    if(vk.conditional_rendering){extensions.push_back(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);conditional.conditionalRendering=VK_TRUE;}
    // Counter reports carry GPU timestamps on the steady clock (the game times its GPU work with them).
    for(const auto& extension:available){
        if(!std::strcmp(extension.extensionName,VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)){vk.calibrated_timestamps="vkGetCalibratedTimestampsKHR";extensions.push_back(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);break;}
        if(!std::strcmp(extension.extensionName,VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)&&!vk.calibrated_timestamps)vk.calibrated_timestamps="vkGetCalibratedTimestampsEXT";
    }
    if(vk.calibrated_timestamps&&!std::strcmp(vk.calibrated_timestamps,"vkGetCalibratedTimestampsEXT"))extensions.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = vk.conditional_rendering ? static_cast<void*>(&conditional) : static_cast<void*>(&v11);
    vkGetPhysicalDeviceFeatures(vk.physical, &features.features); // enable everything the core offers
    features.features.robustBufferAccess = vk.robust;

    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.pNext = &features;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = (u32)extensions.size();
    info.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(vk.physical, &info, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, vk.queue_family, 0, &vk.queue);
    if (queue.queueCount >= 2) vkGetDeviceQueue(vk.device, vk.queue_family, 1, &vk.copy_queue);
    else vk.copy_queue = vk.queue;
    if(push_descriptors){
        vk.push_descriptor_set=reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(vk.device,"vkCmdPushDescriptorSetKHR"));
        VkPhysicalDevicePushDescriptorPropertiesKHR pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};props.pNext=&pp;vkGetPhysicalDeviceProperties2(vk.physical,&props);
        vk.max_push_descriptors=pp.maxPushDescriptors;
    }
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = vk.queue_family;
    VK_CHECK(vkCreateCommandPool(vk.device, &pool, nullptr, &vk.command_pool));
}

} // namespace

VulkanContext& Vk() {
    static VulkanContext vk;
    static std::once_flag once;
    std::call_once(once, [] {
        SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1"); // (controllers also work while the window is unfocused)
        if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD | SDL_INIT_AUDIO)) rt::Fatal("SDL_Init: %s", SDL_GetError());
        gfx::Current();
        vk.window = SDL_CreateWindow("TOoT:R", 1920, 1080,
                                     SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | // (real pixels on scaled desktops: no compositor stretch)
                                     
                                     (std::getenv("OOT_WINDOW_HIDDEN") ? SDL_WINDOW_HIDDEN : 0));
        if (!vk.window) rt::Fatal("SDL_CreateWindow: %s", SDL_GetError());
        CreateInstance(vk);
        if (!SDL_Vulkan_CreateSurface(vk.window, vk.instance, nullptr, &vk.surface))
            rt::Fatal("SDL_Vulkan_CreateSurface: %s", SDL_GetError());
        PickDevice(vk);
        CreateDevice(vk);
    });
    return vk;
}

u32 FindMemoryType(u32 type_bits, VkMemoryPropertyFlags flags) {
    const auto& mp = Vk().memory_properties;
    for (u32 i = 0; i < mp.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
    rt::Fatal("no Vulkan memory type for bits %#x flags %#x", type_bits, flags);
}

} // namespace nvn

namespace nvn {
namespace {
VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
size_t pipeline_cache_saved = 0;
std::string PipelineCachePath() {
    if (const char* p = std::getenv("OOT_PIPELINE_CACHE")) return p;
    return rt::DataDir() + "/pipeline_cache.bin";
}
} // namespace

// Mesa drivers (RADV on the Steam Deck) keep every pipeline in their own disk cache, per pipeline and
// loaded on demand: a cache of the application's own would only be a second copy, all of it in memory
// (it grows past 1 GB) and written again and again. OOT_APP_PIPELINE_CACHE=1 keeps it there too.
bool DriverCachesPipelines() {
    static const bool mesa = [] {
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props.pNext = &driver;
        vkGetPhysicalDeviceProperties2(Vk().physical, &props);
        const bool m = driver.driverID == VK_DRIVER_ID_MESA_RADV || driver.driverID == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                       driver.driverID == VK_DRIVER_ID_MESA_TURNIP || driver.driverID == VK_DRIVER_ID_MESA_NVK;
        if (m && !std::getenv("OOT_APP_PIPELINE_CACHE")) RT_LOG("pipeline cache: the driver's own (%s)", driver.driverName);
        return m && !std::getenv("OOT_APP_PIPELINE_CACHE");
    }();
    return mesa;
}
VkPipelineCache PipelineCache() {
    if (pipeline_cache) return pipeline_cache;
    if (DriverCachesPipelines()) return VK_NULL_HANDLE;
    std::vector<char> data;
    if (FILE* f = std::fopen(PipelineCachePath().c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        data.resize((size_t)std::max(0L, std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        std::fclose(f);
    }
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.empty() ? nullptr : data.data();
    // A cache from another driver or device is rejected by the driver; start empty then.
    if (vkCreatePipelineCache(Vk().device, &info, nullptr, &pipeline_cache) != VK_SUCCESS) {
        info.initialDataSize = 0; info.pInitialData = nullptr;
        VK_CHECK(vkCreatePipelineCache(Vk().device, &info, nullptr, &pipeline_cache));
    }
    pipeline_cache_saved = data.size();
    RT_LOG("pipeline cache: %zu bytes loaded", data.size());
    return pipeline_cache;
}

namespace { void SavePipelineCacheNow(); }
// Writes the cache on a background thread, at most once a minute: it is hundreds of MB, and writing
// it on the command thread stalled the frame for up to 0.7 s.
void SavePipelineCache() {
    if (!pipeline_cache) return;
    static std::atomic<bool> saving{false};
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    static const bool off = std::getenv("OOT_NO_CACHE_SAVE") != nullptr; // (diagnostic: only at exit)
    if (off || now - last < std::chrono::seconds(60) || saving.exchange(true)) return;
    last = now;
    std::thread([] { SavePipelineCacheNow(); saving = false; }).detach();
}
// At exit: pipelines built this session must not be compiled again next time.
void SavePipelineCacheBlocking() { if (pipeline_cache) SavePipelineCacheNow(); }
namespace {
void SavePipelineCacheNow() {
    const auto t0 = std::chrono::steady_clock::now();
    struct Report { std::chrono::steady_clock::time_point t0; size_t before; ~Report() { if (pipeline_cache_saved != before)
        RT_LOG("pipeline cache: %zu MiB saved in %.0f ms", pipeline_cache_saved >> 20, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()); } } report{t0, pipeline_cache_saved};
    size_t size = 0;
    if (vkGetPipelineCacheData(Vk().device, pipeline_cache, &size, nullptr) != VK_SUCCESS || size <= pipeline_cache_saved) return;
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(Vk().device, pipeline_cache, &size, data.data()) != VK_SUCCESS) return;
    const std::string path = PipelineCachePath(), temporary = path + ".tmp";
    if (FILE* f = std::fopen(temporary.c_str(), "wb")) {
        const bool ok = std::fwrite(data.data(), 1, size, f) == size;
        std::fclose(f);
        if (ok && std::rename(temporary.c_str(), path.c_str()) == 0) pipeline_cache_saved = size;
    }
}
}
} // namespace nvn

namespace nvn {
std::string DescribeDeviceAddress(u64 address); // memory.cpp
void ReportDeviceLost() {
    static bool reported = false; if (reported) return; reported = true;
    auto& vk = Vk();
    if (!vk.device_fault) return;
    static auto get = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(vkGetDeviceProcAddr(vk.device, "vkGetDeviceFaultInfoEXT"));
    if (!get) return;
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (get(vk.device, &counts, nullptr) != VK_SUCCESS) return;
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
    VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
    info.pAddressInfos = addresses.data(); info.pVendorInfos = vendor.data(); counts.vendorBinarySize = 0;
    get(vk.device, &counts, &info);
    RT_LOG("GPU fault: %s", info.description);
    for (const auto& a : addresses)
        RT_LOG("  fault address type %d at %#llx (precision %#llx): %s", (int)a.addressType, (unsigned long long)a.reportedAddress,
               (unsigned long long)a.addressPrecision, DescribeDeviceAddress(a.reportedAddress).c_str());
    for (const auto& v : vendor) RT_LOG("  vendor: %s code %#llx data %#llx", v.description, (unsigned long long)v.vendorFaultCode, (unsigned long long)v.vendorFaultData);
}
}
