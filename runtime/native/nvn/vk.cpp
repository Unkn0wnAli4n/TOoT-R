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
#include "vk_dlss.h" // (third_party/longshot-upscale)
#include "shader_compiler.h"

namespace nvn {
namespace {

void CreateInstance(VulkanContext& vk) {
    Uint32 count = 0;
    const char* const* sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!sdl_extensions) rt::Fatal("SDL_Vulkan_GetInstanceExtensions: %s", SDL_GetError());
    std::vector<const char*> extensions(sdl_extensions, sdl_extensions + count);
    {   // (HDR surfaces are listed only with this: output.cpp)
        u32 n = 0; vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> have(n); vkEnumerateInstanceExtensionProperties(nullptr, &n, have.data());
        for (const auto& e : have) if (!std::strcmp(e.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) { extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME); break; }
    }
    {   // (NVIDIA DLSS's, in a build with it: the vendored upscalers, longshot_upscale.cpp)
        u32 n = 0; vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> have(n); vkEnumerateInstanceExtensionProperties(nullptr, &n, have.data());
        mkwvk::dlss_instance_extensions(extensions, have);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "tootr";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = (u32)extensions.size();
    info.ppEnabledExtensionNames = extensions.data();
    std::vector<const char*> layers;
    if (std::getenv("OOT_VK_VALIDATION")) layers.push_back("VK_LAYER_KHRONOS_validation");
    const bool report = std::getenv("OOT_ADDRESS_REPORT") != nullptr;
    const bool validation = std::getenv("OOT_VK_VALIDATION") != nullptr;
    if (report || validation) { extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); info.enabledExtensionCount = (u32)extensions.size(); info.ppEnabledExtensionNames = extensions.data(); }
    info.enabledLayerCount = (u32)layers.size();
    info.ppEnabledLayerNames = layers.data();
    VK_CHECK(vkCreateInstance(&info, nullptr, &vk.instance));
    if (report || validation) { // OOT_ADDRESS_REPORT=1: every GPU address binding, to <data>/address_report.txt
        static FILE* out = report ? std::fopen((rt::DataDir() + "/address_report.txt").c_str(), "w") : nullptr;
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        if (report) mi.messageType |= VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
        mi.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* d, void*) -> VkBool32 {
            if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))
                RT_LOG("Vulkan validation: %s", d->pMessage);
            for (auto* n = static_cast<const VkBaseInStructure*>(d->pNext); n; n = n->pNext)
                if (n->sType == VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT && out) {
                    const auto* b = reinterpret_cast<const VkDeviceAddressBindingCallbackDataEXT*>(n);
                    std::fprintf(out, "%s %#llx+%#llx object %d %#llx%s\n", b->bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT ? "bind" : "unbind",
                                 (unsigned long long)b->baseAddress, (unsigned long long)b->size, d->objectCount ? (int)d->pObjects[0].objectType : -1,
                                 d->objectCount ? (unsigned long long)d->pObjects[0].objectHandle : 0ull, (b->flags & VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT) ? " internal" : "");
                    std::fflush(out);
                }
            return VK_FALSE;
        };
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk.instance, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerEXT messenger;
        if (create) create(vk.instance, &mi, nullptr, &messenger);
    }
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
    vk.subgroup_size = subgroup.subgroupSize;
    if ((subgroup.supportedStages & needed) != needed || subgroup.subgroupSize != 32)
        RT_LOG("subgroup operations in stages %#x, size %u (the game's programs are written for 32 lanes)",
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
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,     // NVN unbound image handles read zero
    };
    u32 extension_count=0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(vk.physical,nullptr,&extension_count,nullptr));
    std::vector<VkExtensionProperties> available(extension_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(vk.physical,nullptr,&extension_count,available.data()));
    // NVN2 task/mesh programs: native stages where the driver runs them as the console does (NVIDIA); elsewhere compute
    // programs and a generated vertex stage (render.cpp "mesh fallback"): RADV's task stage runs 64-lane waves, which the
    // game's 32-lane programs do not survive (the Steam Deck's GPU reset when the camera turned).
    for (const auto& e : available) vk.mesh_shader |= !std::strcmp(e.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    if (vk.mesh_shader) {
        VkPhysicalDeviceMeshShaderFeaturesEXT have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        vk.mesh_shader = have.taskShader && have.meshShader;
    }
    {
        const char* e = std::getenv("OOT_MESH_FALLBACK");
        vk.mesh_fallback = e ? std::atoi(e) != 0 : (!vk.mesh_shader || vk.properties.vendorID != 0x10DE);
        if (!vk.mesh_fallback && !vk.mesh_shader) { RT_LOG("OOT_MESH_FALLBACK=0 ignored: this GPU has no mesh shader stages"); vk.mesh_fallback = true; }
        if (vk.mesh_fallback) vk.mesh_shader = false;
        else extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        RT_LOG("task/mesh programs: %s", vk.mesh_fallback ? "compute fallback" : "native mesh shader stages");
    }
    {   // (compute stages held to the console's 32-lane subgroups where the GPU's own are larger)
        VkPhysicalDeviceVulkan13Features have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        VkPhysicalDeviceVulkan13Properties limits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &limits;
        vkGetPhysicalDeviceProperties2(vk.physical, &p2);
        const bool can = have.subgroupSizeControl && (limits.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) && limits.minSubgroupSize <= 32 && limits.maxSubgroupSize >= 32;
        const char* e = std::getenv("OOT_PIN_SUBGROUP");
        vk.pin_subgroup = can && (e ? std::atoi(e) != 0 : vk.subgroup_size != 32);
        vk.full_subgroups = vk.pin_subgroup && have.computeFullSubgroups;
        // (the graphics stages the GPU can hold to 32 lanes too are: RDNA 2's fragment, task and mesh stages. The ones it
        // cannot, its vertex stage, run at the device's size, and are translated for that: shader_compiler_ls.cpp)
        const bool sizes = have.subgroupSizeControl && limits.minSubgroupSize <= 32 && limits.maxSubgroupSize >= 32 && vk.subgroup_size != 32 && !(e && !std::atoi(e));
        // (graphics stages: opt-in, OOT_PIN_GRAPHICS=1. With it on, a Steam Deck hung at the title's first frame in three
        // runs of three, 2026-11-16 runs 10-12; without it run 9 reached gameplay.)
        static const bool graphics = std::getenv("OOT_PIN_GRAPHICS") && std::atoi(std::getenv("OOT_PIN_GRAPHICS")) != 0;
        vk.pin_stages = sizes ? (graphics ? limits.requiredSubgroupSizeStages : limits.requiredSubgroupSizeStages & (u32)VK_SHADER_STAGE_COMPUTE_BIT) : 0;
        if (!graphics) vk.subgroup_max = 32; // (no stage is translated for wider subgroups either)
        if (!vk.pin_subgroup) vk.pin_stages &= ~(u32)VK_SHADER_STAGE_COMPUTE_BIT;
        if (graphics) vk.subgroup_max = std::max(vk.subgroup_size, have.subgroupSizeControl ? limits.maxSubgroupSize : vk.subgroup_size);
        if (vk.subgroup_size != 32) RT_LOG("subgroups: stages %#x are held to 32 lanes, the others run at up to %u", vk.pin_stages, vk.subgroup_max);
        RT_LOG("compute subgroups: device %u lanes (%u-%u), %s", vk.subgroup_size, limits.minSubgroupSize, limits.maxSubgroupSize, vk.pin_subgroup ? "held to 32" : can ? "as the device runs them" : "cannot be set");
    }
    mkwvk::dlss_device_extensions(extensions, available); mkwvk::dlss_fg_device_extensions(extensions, available); // (VK_NVX_binary_import, VK_NVX_image_view_handle, and frame generation's)
    for (const auto& e : available) if (!std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) { extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME); vk.memory_budget = true; break; }
    // The game's neural networks (the movie's and the picture's upscalers) multiply 16x8x16 half-float matrices: on the
    // device's cooperative matrices where it has that shape (NVIDIA), in software otherwise. OOT_NATIVE_MMA=0: software.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    for (const auto& e : available) if (!std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        const char* off = std::getenv("OOT_NATIVE_MMA"); if (off && !std::atoi(off)) break;
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &coop;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(vkGetInstanceProcAddr(vk.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        bool shape = false;
        if (coop.cooperativeMatrix && properties) {
            u32 n = 0; properties(vk.physical, &n, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> list(n, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
            properties(vk.physical, &n, list.data());
            for (const auto& p : list) shape |= p.MSize == 16 && p.NSize == 8 && p.KSize == 16 && p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                p.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.scope == VK_SCOPE_SUBGROUP_KHR;
        }
        if (shape) { extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); vk.cooperative_matrix = true; }
        coop = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; coop.cooperativeMatrix = shape;
        break;
    }
    bool checkpoints=false;
    if(std::getenv("OOT_GPU_CHECKPOINTS"))for(const auto& extension:available)
        if(!std::strcmp(extension.extensionName,VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)){
            extensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);checkpoints=true;break;
        }
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT placed{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    {   // (memory pools in GPU memory mapped at their guest addresses, without exporting it: memory.cpp)
        bool m2 = false, mp = false;
        for (const auto& e : available) { m2 |= !std::strcmp(e.extensionName, VK_KHR_MAP_MEMORY_2_EXTENSION_NAME); mp |= !std::strcmp(e.extensionName, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME); }
        if (m2 && mp && !std::getenv("OOT_NO_PLACED_MAPS")) {
            VkPhysicalDeviceMapMemoryPlacedFeaturesEXT have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
            vkGetPhysicalDeviceFeatures2(vk.physical, &query);
            VkPhysicalDeviceMapMemoryPlacedPropertiesEXT props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &props;
            vkGetPhysicalDeviceProperties2(vk.physical, &p2);
            if (have.memoryMapPlaced && have.memoryUnmapReserve && props.minPlacedMemoryMapAlignment <= 4096) {
                extensions.push_back(VK_KHR_MAP_MEMORY_2_EXTENSION_NAME); extensions.push_back(VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME);
                placed.memoryMapPlaced = VK_TRUE; placed.memoryUnmapReserve = VK_TRUE;
                vk.placed_maps = true;
            }
        }
    }
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
    // Shader local memory starts at zero on NVIDIA only (its driver hands an unwritten read another lane's values); on
    // AMD the zeroing of OoT's large local arrays went through scratch memory and faulted the GPU (Steam Deck: page
    // faults and resets when new scenery came into view). OOT_LOCAL_ZERO=0/1 forces it.
#ifdef _WIN32
    if (!std::getenv("OOT_LOCAL_ZERO") && vk.properties.vendorID != 0x10DE) _putenv_s("OOT_LOCAL_ZERO", "0");
    if (!std::getenv("OOT_LOCAL_CRS") && vk.properties.vendorID != 0x10DE) _putenv_s("OOT_LOCAL_CRS", "0");
    if (!std::getenv("OOT_MESH_NOCULL") && vk.properties.vendorID != 0x10DE) _putenv_s("OOT_MESH_NOCULL", "1");
#else
    if (!std::getenv("OOT_LOCAL_ZERO") && vk.properties.vendorID != 0x10DE) setenv("OOT_LOCAL_ZERO", "0", 0);
    if (!std::getenv("OOT_LOCAL_CRS") && vk.properties.vendorID != 0x10DE) setenv("OOT_LOCAL_CRS", "0", 0); // (shader_compiler.cpp)
    // (the meshlet culling task shader's grid walk ran past the GPU timeout on the Steam Deck: oot_ir_passes.h)
    if (!std::getenv("OOT_MESH_NOCULL") && vk.properties.vendorID != 0x10DE) setenv("OOT_MESH_NOCULL", "1", 0);
#endif
    robustness.robustBufferAccess2=vk.robust;
    robustness.robustImageAccess2=vk.robust;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shaders{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    mesh_shaders.pNext=&robustness;
    mesh_shaders.taskShader = VK_TRUE;
    mesh_shaders.meshShader = VK_TRUE;
    if (vk.mesh_shader) {   // (draw-log statistics of task/mesh invocations, where the driver has them: RADV does not)
        VkPhysicalDeviceMeshShaderFeaturesEXT supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &supported;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        mesh_shaders.meshShaderQueries = supported.meshShaderQueries;
        vk.mesh_shader_queries = supported.meshShaderQueries;
    }
    VkPhysicalDeviceIndexTypeUint8FeaturesKHR uint8_indices{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR};
    uint8_indices.indexTypeUint8 = VK_TRUE;
    uint8_indices.pNext = vk.mesh_shader ? static_cast<void*>(&mesh_shaders) : static_cast<void*>(&robustness);
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.pNext = &uint8_indices;
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;
    v13.maintenance4 = VK_TRUE;
    v13.pipelineCreationCacheControl = VK_TRUE; // (render.cpp: a pipeline from the driver's cache only, or not now)
    v13.subgroupSizeControl = vk.pin_subgroup || vk.pin_stages != 0; v13.computeFullSubgroups = vk.full_subgroups;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.pNext = &v13;
    v12.bufferDeviceAddress = VK_TRUE;
    {   // (the guest address table at the same GPU address every run: memory.cpp)
        VkPhysicalDeviceVulkan12Features have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        vk.address_replay = have.bufferDeviceAddressCaptureReplay && !std::getenv("OOT_NO_ADDRESS_REPLAY");
        v12.bufferDeviceAddressCaptureReplay = vk.address_replay;
        // (what the translated shaders declare: narrow members in uniform blocks, the Vulkan memory model)
        v12.uniformAndStorageBuffer8BitAccess = have.uniformAndStorageBuffer8BitAccess;
        v12.vulkanMemoryModel = have.vulkanMemoryModel; v12.vulkanMemoryModelDeviceScope = have.vulkanMemoryModelDeviceScope;
    }
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
    {
        VkPhysicalDeviceVulkan11Features have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; query.pNext = &have;
        vkGetPhysicalDeviceFeatures2(vk.physical, &query);
        v11.uniformAndStorageBuffer16BitAccess = have.uniformAndStorageBuffer16BitAccess;
    }
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

    VkPhysicalDeviceAddressBindingReportFeaturesEXT address_report{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT};
    if (std::getenv("OOT_ADDRESS_REPORT")) {
        u32 n = 0; vkEnumerateDeviceExtensionProperties(vk.physical, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> all(n); vkEnumerateDeviceExtensionProperties(vk.physical, nullptr, &n, all.data());
        for (const auto& e : all) if (!std::strcmp(e.extensionName, VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
            address_report.reportAddressBinding = VK_TRUE; address_report.pNext = features.pNext; features.pNext = &address_report;
            RT_LOG("GPU address bindings are reported (address_report.txt)");
        }
    }
    if (vk.placed_maps) { placed.pNext = features.pNext; features.pNext = &placed; }
    if (vk.cooperative_matrix) { coop.pNext = features.pNext; features.pNext = &coop; }
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.pNext = &features;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = (u32)extensions.size();
    info.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(vk.physical, &info, nullptr, &vk.device));
    if(checkpoints)vk.checkpoint=reinterpret_cast<PFN_vkCmdSetCheckpointNV>(vkGetDeviceProcAddr(vk.device,"vkCmdSetCheckpointNV"));
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

// One workgroup of a 32-thread compute program over a storage buffer of words (shader_compiler.h ProbeCooperativeMatrices).
VulkanContext* probing = nullptr;
bool RunMatrixProgram(const std::vector<uint32_t>& code, std::vector<uint32_t>& words) {
    auto& vk = *probing; const VkDeviceSize size = words.size() * 4; bool ok = false;
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE; VkDescriptorPool pool = VK_NULL_HANDLE; VkCommandBuffer cmd = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    do {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(vk.device, &bi, nullptr, &buffer) != VK_SUCCESS) break;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk.device, buffer, &req);
        u32 type = ~0u; const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (u32 i = 0; i < vk.memory_properties.memoryTypeCount && type == ~0u; ++i) if ((req.memoryTypeBits >> i & 1) && (vk.memory_properties.memoryTypes[i].propertyFlags & want) == want) type = i;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        if (type == ~0u || vkAllocateMemory(vk.device, &ai, nullptr, &memory) != VK_SUCCESS) break;
        vkBindBufferMemory(vk.device, buffer, memory, 0);
        u32* mapped = nullptr; if (vkMapMemory(vk.device, memory, 0, size, 0, reinterpret_cast<void**>(&mapped)) != VK_SUCCESS) break;
        std::memcpy(mapped, words.data(), size);
        VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = 1; li.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(vk.device, &li, nullptr, &set_layout) != VK_SUCCESS) break;
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount = 1; pl.pSetLayouts = &set_layout;
        if (vkCreatePipelineLayout(vk.device, &pl, nullptr, &layout) != VK_SUCCESS) break;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize = code.size() * 4; mi.pCode = code.data();
        if (vkCreateShaderModule(vk.device, &mi, nullptr, &module) != VK_SUCCESS) break;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; ci.layout = layout;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo lanes{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; lanes.requiredSubgroupSize = 32;
        if (vk.pin_subgroup) ci.stage.pNext = &lanes;
        if (vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) != VK_SUCCESS) break;
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(vk.device, &dp, nullptr, &pool) != VK_SUCCESS) break;
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; da.descriptorPool = pool; da.descriptorSetCount = 1; da.pSetLayouts = &set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE; if (vkAllocateDescriptorSets(vk.device, &da, &set) != VK_SUCCESS) break;
        VkDescriptorBufferInfo info{buffer, 0, size};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet = set; write.descriptorCount = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
        vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ca.commandPool = vk.command_pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vk.device, &ca, &cmd) != VK_SUCCESS) break;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(cmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (vkCreateFence(vk.device, &fi, nullptr, &fence) != VK_SUCCESS) break;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(vk.queue, 1, &si, fence) != VK_SUCCESS || vkWaitForFences(vk.device, 1, &fence, VK_TRUE, 5'000'000'000ull) != VK_SUCCESS) break;
        std::memcpy(words.data(), mapped, size); ok = true;
    } while (false);
    if (fence) vkDestroyFence(vk.device, fence, nullptr);
    if (cmd) vkFreeCommandBuffers(vk.device, vk.command_pool, 1, &cmd);
    if (pool) vkDestroyDescriptorPool(vk.device, pool, nullptr);
    if (pipeline) vkDestroyPipeline(vk.device, pipeline, nullptr);
    if (module) vkDestroyShaderModule(vk.device, module, nullptr);
    if (layout) vkDestroyPipelineLayout(vk.device, layout, nullptr);
    if (set_layout) vkDestroyDescriptorSetLayout(vk.device, set_layout, nullptr);
    if (buffer) vkDestroyBuffer(vk.device, buffer, nullptr);
    if (memory) vkFreeMemory(vk.device, memory, nullptr);
    return ok;
}
void ProbeMatrices(VulkanContext& vk) {
    if (!vk.cooperative_matrix) { RT_LOG("matrix multiplies: in software (no 16x8x16 half-float cooperative matrices)"); return; }
    probing = &vk; std::string report;
    const bool native = nvshader::ProbeCooperativeMatrices(RunMatrixProgram, &report);
    RT_LOG("matrix multiplies: %s (%s)", native ? "cooperative matrices" : "in software", report.c_str());
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
        if (const SDL_DisplayMode* mode = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(vk.window)); mode && mode->h > 0) gfx::display_ratio.store((double)mode->w / mode->h);
        if (const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(vk.window)); mode && mode->refresh_rate > 1) gfx::display_refresh.store(mode->refresh_rate);
        CreateInstance(vk);
        if (!SDL_Vulkan_CreateSurface(vk.window, vk.instance, nullptr, &vk.surface))
            rt::Fatal("SDL_Vulkan_CreateSurface: %s", SDL_GetError());
        PickDevice(vk);
        CreateDevice(vk);
        ProbeMatrices(vk);
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
// The drivers keep compiled pipelines on disk themselves (Mesa's cache, NVIDIA's shader disk cache: both enlarged in
// main.cpp). An application cache of its own grew to 2 GB on NVIDIA, was held in memory whole, and saving it held the
// driver's cache for seconds (the executor's pipeline creations waited: 0.6-1.4 s hitches). OOT_APP_PIPELINE_CACHE=1: ours.
bool DriverCachesPipelines() {
    static const bool driver = [] {
        VkPhysicalDeviceDriverProperties properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props.pNext = &properties;
        vkGetPhysicalDeviceProperties2(Vk().physical, &props);
        const bool d = !std::getenv("OOT_APP_PIPELINE_CACHE");
        if (d) {
            RT_LOG("pipeline cache: the driver's own (%s)", properties.driverName);
            if (std::remove(PipelineCachePath().c_str()) == 0) RT_LOG("pipeline cache: the old cache file was removed");
        }
        return d;
    }();
    return driver;
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
    if(vk.checkpoint){
        auto checkpoints=reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(vkGetDeviceProcAddr(vk.device,"vkGetQueueCheckpointDataNV"));
        const VkQueue queues[]{vk.queue,vk.copy_queue};
        for(u32 q=0;q<(vk.copy_queue!=vk.queue?2u:1u);++q){
            u32 count=0;checkpoints(queues[q],&count,nullptr);
            std::vector<VkCheckpointDataNV> data(count,{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
            checkpoints(queues[q],&count,data.data());
            for(const auto& d:data)RT_LOG("GPU checkpoint queue %u stage %#x marker %#llx",q,(u32)d.stage,(unsigned long long)reinterpret_cast<u64>(d.pCheckpointMarker));
        }
    }
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
namespace nvn {
VkResult TracedCreateBuffer(const VkBufferCreateInfo* info, VkBuffer* buffer, const char* file, int line) {
    const VkResult r = vkCreateBuffer(Vk().device, info, nullptr, buffer);
    static const bool on = std::getenv("OOT_ADDRESS_REPORT") != nullptr;
    if (on && r == VK_SUCCESS && info->size >= (1u << 20)) { const char* f = std::strrchr(file, '/');
        RT_LOG("buffer %#llx: %.1f MiB, made at %s:%d", (unsigned long long)(u64)*buffer, info->size / 1048576.0, f ? f + 1 : file, line); }
    return r;
}
} // namespace nvn
// (engine_hooks.cpp: the display's shape is known once the window exists)
void oot_ensure_window() { nvn::Vk(); }
