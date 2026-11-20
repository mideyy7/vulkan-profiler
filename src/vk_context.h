#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

// Number of validation warnings and errors reported since the program started.
// It lives outside VulkanContext so it can still be read after the context is
// destroyed, which is when leaked objects are reported.
uint32_t validationMessageCount();

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
