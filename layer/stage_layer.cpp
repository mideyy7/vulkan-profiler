// VK_LAYER_stage_profiler: a Vulkan layer that profiles any Vulkan app without
// recompiling it. The loader places it between the app and the driver, so every
// call below passes through here on its way down.
//
// For now it only passes every call through to the next layer.

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

// Every dispatchable handle (instance, physical device, device, queue, command
// buffer) starts with a pointer to the loader's dispatch table. Objects that
// belong to the same device share that pointer, so it identifies the device.
template <typename Handle>
void* dispatchKey(Handle handle) {
    return *reinterpret_cast<void**>(handle);
}

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr = nullptr;
    PFN_vkDestroyInstance nextDestroyInstance = nullptr;
};

// The next layer's (or driver's) functions for one device.
struct DeviceData {
    PFN_vkGetDeviceProcAddr nextGetDeviceProcAddr = nullptr;
    PFN_vkDestroyDevice nextDestroyDevice = nullptr;
};

// Apps may call Vulkan from several threads, so the maps are guarded.
std::mutex gMutex;
std::unordered_map<void*, InstanceData> gInstances;
std::unordered_map<void*, DeviceData*> gDevices;

InstanceData findInstance(void* key) {
    const std::lock_guard<std::mutex> lock(gMutex);
    return gInstances.at(key);
}

DeviceData* findDevice(void* key) {
    const std::lock_guard<std::mutex> lock(gMutex);
    return gDevices.at(key);
}

// ---------------------------------------------------------------------------
// Instance functions
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* createInfo,
                                              const VkAllocationCallbacks* allocator,
                                              VkInstance* instance) {
    // The loader puts a chain-link struct in pNext telling each layer where the
    // next layer down is.
    auto* link = static_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(createInfo->pNext));
    while (link != nullptr && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                                link->function == VK_LAYER_LINK_INFO)) {
        link = static_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(link->pNext));
    }
    if (link == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr =
        link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    // Advance the chain so the next layer finds its own link.
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    auto nextCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
        nextGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    const VkResult result = nextCreateInstance(createInfo, allocator, instance);
    if (result != VK_SUCCESS) {
        return result;
    }

    InstanceData data;
    data.instance = *instance;
    data.nextGetInstanceProcAddr = nextGetInstanceProcAddr;
    data.nextDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        nextGetInstanceProcAddr(*instance, "vkDestroyInstance"));
    const std::lock_guard<std::mutex> lock(gMutex);
    gInstances[dispatchKey(*instance)] = data;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance,
                                           const VkAllocationCallbacks* allocator) {
    if (instance == VK_NULL_HANDLE) {
        return;
    }
    InstanceData data;
    {
        const std::lock_guard<std::mutex> lock(gMutex);
        auto it = gInstances.find(dispatchKey(instance));
        if (it == gInstances.end()) {
            return;
        }
        data = it->second;
        gInstances.erase(it);
    }
    data.nextDestroyInstance(instance, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physicalDevice,
                                            const VkDeviceCreateInfo* createInfo,
                                            const VkAllocationCallbacks* allocator,
                                            VkDevice* device) {
    auto* link = static_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(createInfo->pNext));
    while (link != nullptr && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                                link->function == VK_LAYER_LINK_INFO)) {
        link = static_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(link->pNext));
    }
    if (link == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr =
        link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const PFN_vkGetDeviceProcAddr nextGetDeviceProcAddr =
        link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    // A physical device shares its instance's dispatch key.
    const InstanceData instance = findInstance(dispatchKey(physicalDevice));
    auto nextCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
        nextGetInstanceProcAddr(instance.instance, "vkCreateDevice"));
    const VkResult result = nextCreateDevice(physicalDevice, createInfo, allocator, device);
    if (result != VK_SUCCESS) {
        return result;
    }

    auto* data = new DeviceData;
    auto next = [&](const char* name) { return nextGetDeviceProcAddr(*device, name); };
    data->nextGetDeviceProcAddr = nextGetDeviceProcAddr;
    data->nextDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(next("vkDestroyDevice"));

    const std::lock_guard<std::mutex> lock(gMutex);
    gDevices[dispatchKey(*device)] = data;
    return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Device functions
// ---------------------------------------------------------------------------

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks* allocator) {
    if (device == VK_NULL_HANDLE) {
        return;
    }
    DeviceData* data = nullptr;
    {
        const std::lock_guard<std::mutex> lock(gMutex);
        auto it = gDevices.find(dispatchKey(device));
        if (it == gDevices.end()) {
            return;
        }
        data = it->second;
        gDevices.erase(it);
    }
    data->nextDestroyDevice(device, allocator);
    delete data;
}

// ---------------------------------------------------------------------------
// Function lookup
// ---------------------------------------------------------------------------

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name);

// Our replacement for a device-level function name, or nullptr.
PFN_vkVoidFunction interceptedDeviceFunction(const char* name) {
    struct Entry {
        const char* name;
        PFN_vkVoidFunction function;
    };
    static const Entry entries[] = {
        {"vkGetDeviceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(GetDeviceProcAddr)},
        {"vkDestroyDevice", reinterpret_cast<PFN_vkVoidFunction>(DestroyDevice)},
    };
    for (const Entry& entry : entries) {
        if (std::strcmp(name, entry.name) == 0) {
            return entry.function;
        }
    }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name) {
    if (PFN_vkVoidFunction ours = interceptedDeviceFunction(name)) {
        return ours;
    }
    return findDevice(dispatchKey(device))->nextGetDeviceProcAddr(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance,
                                                              const char* name) {
    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(GetInstanceProcAddr);
    }
    if (std::strcmp(name, "vkCreateInstance") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(CreateInstance);
    }
    if (std::strcmp(name, "vkDestroyInstance") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(DestroyInstance);
    }
    if (std::strcmp(name, "vkCreateDevice") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(CreateDevice);
    }
    if (PFN_vkVoidFunction ours = interceptedDeviceFunction(name)) {
        return ours;
    }
    if (instance == VK_NULL_HANDLE) {
        return nullptr;
    }
    return findInstance(dispatchKey(instance)).nextGetInstanceProcAddr(instance, name);
}

}  // namespace

// The only exported symbol. The loader calls it to agree an interface version
// and get the layer's two lookup functions.
extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* version) {
    if (version == nullptr || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (version->loaderLayerInterfaceVersion > 2) {
        version->loaderLayerInterfaceVersion = 2;
    }
    version->pfnGetInstanceProcAddr = GetInstanceProcAddr;
    version->pfnGetDeviceProcAddr = GetDeviceProcAddr;
    version->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
