#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

// Throws std::runtime_error naming `action` if `result` is not VK_SUCCESS.
void check(VkResult result, const char* action);

// Number of validation warnings and errors reported since the program started.
// It lives outside VulkanContext so it can still be read after the context is
// destroyed, which is when leaked objects are reported.
uint32_t validationMessageCount();

// Number of messages whose ID starts with "SYNC-HAZARD", for the validation canary test.
uint32_t syncHazardMessageCount();

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamilyIndex() const { return queueFamilyIndex_; }

    void printDeviceInfo() const;

private:
    void createInstance();
    void createDebugMessenger();
    void pickPhysicalDevice();
    void createDevice();
    void destroy();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex_ = 0;
    uint32_t timestampValidBits_ = 0;
};
