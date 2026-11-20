#include "vk_context.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr const char* kValidationFeaturesExtension = "VK_EXT_validation_features";
constexpr const char* kPortabilitySubsetExtension = "VK_KHR_portability_subset";

std::atomic<uint32_t> gValidationMessages{0};

std::vector<VkExtensionProperties> instanceExtensions(const char* layerName) {
    uint32_t count = 0;
    check(vkEnumerateInstanceExtensionProperties(layerName, &count, nullptr),
          "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(count);
    check(vkEnumerateInstanceExtensionProperties(layerName, &count, extensions.data()),
          "vkEnumerateInstanceExtensionProperties");
    return extensions;
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    check(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr),
          "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(count);
    check(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data()),
          "vkEnumerateDeviceExtensionProperties");
    return extensions;
}

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const VkExtensionProperties& extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

bool hasValidationLayer() {
    uint32_t count = 0;
    check(vkEnumerateInstanceLayerProperties(&count, nullptr), "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> layers(count);
    check(vkEnumerateInstanceLayerProperties(&count, layers.data()),
          "vkEnumerateInstanceLayerProperties");
    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, kValidationLayer) == 0) {
            return true;
        }
    }
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL onDebugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void*) {
    ++gValidationMessages;
    const char* label =
        severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning";
    std::fprintf(stderr, "[vulkan %s] %s\n", label, data->pMessage);
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo() {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    info.pfnUserCallback = onDebugMessage;
    return info;
}

// A usable family can run compute work and record timestamps.
std::optional<uint32_t> findComputeQueueFamily(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    for (uint32_t index = 0; index < count; ++index) {
        const bool hasCompute = (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (hasCompute && families[index].timestampValidBits > 0) {
            return index;
        }
    }
    return std::nullopt;
}

int deviceTypeScore(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return 4;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return 3;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return 1;
        default:
            return 0;
    }
}

const char* deviceTypeName(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "CPU (software renderer)";
        default:
            return "other";
    }
}

}  // namespace

void check(VkResult result, const char* action) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(action) + " failed with VkResult " +
                                 std::to_string(result));
    }
}

uint32_t validationMessageCount() {
    return gValidationMessages.load();
}

VulkanContext::VulkanContext() {
    try {
        createInstance();
        createDebugMessenger();
        pickPhysicalDevice();
        createDevice();
    } catch (...) {
        // A destructor does not run when its constructor throws.
        destroy();
        throw;
    }
}

VulkanContext::~VulkanContext() {
    destroy();
}

void VulkanContext::createInstance() {
    if (!hasValidationLayer()) {
        throw std::runtime_error(std::string(kValidationLayer) + " is not installed");
    }

    // Without this setting, synchronization validation ignores shader accesses.
    // The final 0 keeps any value already set in the environment.
    setenv("VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "true", 0);

    const std::vector<VkExtensionProperties> available = instanceExtensions(nullptr);
    const std::vector<VkExtensionProperties> fromLayer = instanceExtensions(kValidationLayer);
    if (!hasExtension(fromLayer, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) ||
        !hasExtension(fromLayer, kValidationFeaturesExtension)) {
        throw std::runtime_error("the validation layer lacks a required extension");
    }

    std::vector<const char*> extensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                                           kValidationFeaturesExtension};

    VkInstanceCreateFlags flags = 0;
    if (hasExtension(available, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "gpu_profiler";
    application.apiVersion = VK_API_VERSION_1_2;

    const VkValidationFeatureEnableEXT enabledFeatures[] = {
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
    VkValidationFeaturesEXT validationFeatures{};
    validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = enabledFeatures;

    // Chained here so messages from vkCreateInstance and vkDestroyInstance are caught too.
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo = messengerCreateInfo();
    messengerInfo.pNext = &validationFeatures;

    const char* layers[] = {kValidationLayer};

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pNext = &messengerInfo;
    createInfo.flags = flags;
    createInfo.pApplicationInfo = &application;
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = layers;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    check(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
}

void VulkanContext::createDebugMessenger() {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create == nullptr) {
        throw std::runtime_error("vkCreateDebugUtilsMessengerEXT is unavailable");
    }
    const VkDebugUtilsMessengerCreateInfoEXT info = messengerCreateInfo();
    check(create(instance_, &info, nullptr, &debugMessenger_), "vkCreateDebugUtilsMessengerEXT");
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physicalDevices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, physicalDevices.data()),
          "vkEnumeratePhysicalDevices");

    int bestScore = -1;
    for (VkPhysicalDevice candidate : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        const std::optional<uint32_t> family = findComputeQueueFamily(candidate);
        const int score = deviceTypeScore(properties.deviceType);
        if (properties.apiVersion >= VK_API_VERSION_1_2 && family.has_value() &&
            score > bestScore) {
            bestScore = score;
            physicalDevice_ = candidate;
            queueFamilyIndex_ = *family;
        }
    }
    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("no Vulkan 1.2 device with compute and timestamps was found");
    }

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &familyCount, families.data());
    timestampValidBits_ = families[queueFamilyIndex_].timestampValidBits;
}

void VulkanContext::createDevice() {
    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamilyIndex_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    // The Vulkan spec requires this extension on any device that offers it (MoltenVK does).
    std::vector<const char*> extensions;
    if (hasExtension(deviceExtensions(physicalDevice_), kPortabilitySubsetExtension)) {
        extensions.push_back(kPortabilitySubsetExtension);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);
}

void VulkanContext::destroy() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (debugMessenger_ != VK_NULL_HANDLE) {
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyMessenger != nullptr) {
            destroyMessenger(instance_, debugMessenger_, nullptr);
        }
        debugMessenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::printDeviceInfo() const {
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &driver;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);

    const VkPhysicalDeviceProperties& properties = properties2.properties;
    const VkPhysicalDeviceLimits& limits = properties.limits;

    std::printf("Device name:                  %s\n", properties.deviceName);
    std::printf("Device type:                  %s\n", deviceTypeName(properties.deviceType));
    std::printf("Vulkan API version:           %u.%u.%u\n",
                VK_API_VERSION_MAJOR(properties.apiVersion),
                VK_API_VERSION_MINOR(properties.apiVersion),
                VK_API_VERSION_PATCH(properties.apiVersion));
    std::printf("Driver:                       %s %s\n", driver.driverName, driver.driverInfo);
    std::printf("Driver version (raw):         %u\n", properties.driverVersion);
    std::printf("Compute queue family:         %u\n", queueFamilyIndex_);
    std::printf("timestampPeriod:              %f ns/tick\n",
                static_cast<double>(limits.timestampPeriod));
    std::printf("timestampComputeAndGraphics:  %s\n",
                limits.timestampComputeAndGraphics == VK_TRUE ? "true" : "false");
    std::printf("timestampValidBits:           %u\n", timestampValidBits_);
    std::printf("Max workgroup size:           %u x %u x %u\n", limits.maxComputeWorkGroupSize[0],
                limits.maxComputeWorkGroupSize[1], limits.maxComputeWorkGroupSize[2]);
    std::printf("Max workgroup invocations:    %u\n", limits.maxComputeWorkGroupInvocations);
    std::printf("Max shared memory:            %u bytes\n", limits.maxComputeSharedMemorySize);
}
