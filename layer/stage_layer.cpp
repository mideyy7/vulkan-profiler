// VK_LAYER_stage_profiler: a Vulkan layer that profiles any Vulkan app without
// recompiling it. The loader places it between the app and the driver, so every
// call below passes through here on its way down.
//
// It counts dispatch, barrier and copy commands as they are recorded, and times
// how long the CPU spends inside vkQueueSubmit and the wait functions. It does
// not add GPU timestamps to other apps' command buffers: that would change the
// work being measured (see docs/layer.md).
//
// Limitation: newer variants (vkQueueSubmit2, vkCmdPipelineBarrier2,
// vkCmdCopyBuffer2, vkCmdDispatchIndirect, vkCmdDispatchBase) pass through
// without being counted.

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

using Clock = std::chrono::steady_clock;

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

struct TimedCalls {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> totalNs{0};

    void add(Clock::time_point start) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start);
        count += 1;
        totalNs += static_cast<uint64_t>(elapsed.count());
    }
};

// The next layer's (or driver's) functions for one device, and its counters.
struct DeviceData {
    PFN_vkGetDeviceProcAddr nextGetDeviceProcAddr = nullptr;
    PFN_vkDestroyDevice nextDestroyDevice = nullptr;
    PFN_vkQueueSubmit nextQueueSubmit = nullptr;
    PFN_vkWaitForFences nextWaitForFences = nullptr;
    PFN_vkQueueWaitIdle nextQueueWaitIdle = nullptr;
    PFN_vkDeviceWaitIdle nextDeviceWaitIdle = nullptr;
    PFN_vkCmdDispatch nextCmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier nextCmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer nextCmdCopyBuffer = nullptr;

    TimedCalls queueSubmit;
    TimedCalls waitForFences;
    TimedCalls queueWaitIdle;
    TimedCalls deviceWaitIdle;
    std::atomic<uint64_t> submittedCommandBuffers{0};
    std::atomic<uint64_t> dispatches{0};
    std::atomic<uint64_t> barriers{0};
    std::atomic<uint64_t> copies{0};
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

void printTimed(const char* name, const TimedCalls& calls) {
    const uint64_t count = calls.count.load();
    const double totalUs = static_cast<double>(calls.totalNs.load()) / 1e3;
    std::fprintf(stderr, "[stage_profiler] %-18s calls=%llu total_us=%.1f mean_us=%.3f\n", name,
                 static_cast<unsigned long long>(count), totalUs,
                 count > 0 ? totalUs / static_cast<double>(count) : 0.0);
}

void printReport(const DeviceData& d) {
    std::fprintf(stderr, "[stage_profiler] report for this device\n");
    printTimed("vkQueueSubmit", d.queueSubmit);
    printTimed("vkWaitForFences", d.waitForFences);
    printTimed("vkQueueWaitIdle", d.queueWaitIdle);
    printTimed("vkDeviceWaitIdle", d.deviceWaitIdle);
    std::fprintf(stderr, "[stage_profiler] command buffers submitted=%llu\n",
                 static_cast<unsigned long long>(d.submittedCommandBuffers.load()));
    std::fprintf(stderr, "[stage_profiler] recorded commands: dispatch=%llu barrier=%llu copy=%llu\n",
                 static_cast<unsigned long long>(d.dispatches.load()),
                 static_cast<unsigned long long>(d.barriers.load()),
                 static_cast<unsigned long long>(d.copies.load()));
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
    data->nextQueueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(next("vkQueueSubmit"));
    data->nextWaitForFences = reinterpret_cast<PFN_vkWaitForFences>(next("vkWaitForFences"));
    data->nextQueueWaitIdle = reinterpret_cast<PFN_vkQueueWaitIdle>(next("vkQueueWaitIdle"));
    data->nextDeviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(next("vkDeviceWaitIdle"));
    data->nextCmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(next("vkCmdDispatch"));
    data->nextCmdPipelineBarrier =
        reinterpret_cast<PFN_vkCmdPipelineBarrier>(next("vkCmdPipelineBarrier"));
    data->nextCmdCopyBuffer = reinterpret_cast<PFN_vkCmdCopyBuffer>(next("vkCmdCopyBuffer"));

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
    printReport(*data);
    data->nextDestroyDevice(device, allocator);
    delete data;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t submitCount,
                                           const VkSubmitInfo* submits, VkFence fence) {
    DeviceData* d = findDevice(dispatchKey(queue));
    for (uint32_t s = 0; s < submitCount; ++s) {
        d->submittedCommandBuffers += submits[s].commandBufferCount;
    }
    const Clock::time_point start = Clock::now();
    const VkResult result = d->nextQueueSubmit(queue, submitCount, submits, fence);
    d->queueSubmit.add(start);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice device, uint32_t fenceCount,
                                             const VkFence* fences, VkBool32 waitAll,
                                             uint64_t timeout) {
    DeviceData* d = findDevice(dispatchKey(device));
    const Clock::time_point start = Clock::now();
    const VkResult result = d->nextWaitForFences(device, fenceCount, fences, waitAll, timeout);
    d->waitForFences.add(start);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queue) {
    DeviceData* d = findDevice(dispatchKey(queue));
    const Clock::time_point start = Clock::now();
    const VkResult result = d->nextQueueWaitIdle(queue);
    d->queueWaitIdle.add(start);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device) {
    DeviceData* d = findDevice(dispatchKey(device));
    const Clock::time_point start = Clock::now();
    const VkResult result = d->nextDeviceWaitIdle(device);
    d->deviceWaitIdle.add(start);
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    DeviceData* d = findDevice(dispatchKey(cmd));
    d->dispatches += 1;
    d->nextCmdDispatch(cmd, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(
    VkCommandBuffer cmd, VkPipelineStageFlags srcStages, VkPipelineStageFlags dstStages,
    VkDependencyFlags flags, uint32_t memoryCount, const VkMemoryBarrier* memory,
    uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, uint32_t imageCount,
    const VkImageMemoryBarrier* images) {
    DeviceData* d = findDevice(dispatchKey(cmd));
    d->barriers += 1;
    d->nextCmdPipelineBarrier(cmd, srcStages, dstStages, flags, memoryCount, memory, bufferCount,
                              buffers, imageCount, images);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst,
                                         uint32_t regionCount, const VkBufferCopy* regions) {
    DeviceData* d = findDevice(dispatchKey(cmd));
    d->copies += 1;
    d->nextCmdCopyBuffer(cmd, src, dst, regionCount, regions);
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
        {"vkQueueSubmit", reinterpret_cast<PFN_vkVoidFunction>(QueueSubmit)},
        {"vkWaitForFences", reinterpret_cast<PFN_vkVoidFunction>(WaitForFences)},
        {"vkQueueWaitIdle", reinterpret_cast<PFN_vkVoidFunction>(QueueWaitIdle)},
        {"vkDeviceWaitIdle", reinterpret_cast<PFN_vkVoidFunction>(DeviceWaitIdle)},
        {"vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(CmdDispatch)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(CmdPipelineBarrier)},
        {"vkCmdCopyBuffer", reinterpret_cast<PFN_vkVoidFunction>(CmdCopyBuffer)},
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
