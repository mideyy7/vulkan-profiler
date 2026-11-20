#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// Throws std::runtime_error naming `action` if `result` is not VK_SUCCESS.
void check(VkResult result, const char* action);

// Number of validation warnings and errors reported since the program started.
// It lives outside VulkanContext so it can still be read after the context is
// destroyed, which is when leaked objects are reported.
uint32_t validationMessageCount();

// Number of messages whose ID starts with "SYNC-HAZARD", for the validation canary test.
uint32_t syncHazardMessageCount();

// Every fence wait in the app goes through waitForFence, which times it. The
// totals let Phase 7 compare the app's own measurement with the layer's.
struct FenceWaitStats {
    uint64_t count = 0;
    double totalNs = 0.0;
};
void waitForFence(VkDevice device, VkFence fence);
FenceWaitStats fenceWaitStats();

// Recording wrappers that count commands, so the layer's counts can be checked.
struct RecordedCommandCounts {
    uint64_t dispatches = 0;
    uint64_t barriers = 0;
    uint64_t copies = 0;
};
void cmdDispatch(VkCommandBuffer cmd, uint32_t groupCountX);
void cmdPipelineBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStages,
                        VkPipelineStageFlags dstStages, uint32_t memoryBarrierCount,
                        const VkMemoryBarrier* memoryBarriers, uint32_t bufferBarrierCount,
                        const VkBufferMemoryBarrier* bufferBarriers);
void cmdCopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, VkDeviceSize size);
RecordedCommandCounts recordedCommandCounts();

class VulkanContext {
public:
    // Validation is on by default. Benchmarks turn it off because the layer's
    // checks add CPU time to every call being measured.
    explicit VulkanContext(bool enableValidation = true);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamilyIndex() const { return queueFamilyIndex_; }
    uint32_t timestampValidBits() const { return timestampValidBits_; }
    bool validationEnabled() const { return validationEnabled_; }

    float timestampPeriod() const;
    std::string deviceName() const;
    std::string driverDescription() const;

    void printDeviceInfo() const;

private:
    void createInstance();
    void createDebugMessenger();
    void pickPhysicalDevice();
    void createDevice();
    void destroy();

    bool validationEnabled_ = true;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex_ = 0;
    uint32_t timestampValidBits_ = 0;
};
