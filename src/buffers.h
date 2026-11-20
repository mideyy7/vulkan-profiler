#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>

class VulkanContext;

// A Vulkan buffer with its own dedicated memory allocation.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // Non-null only for host-visible, persistently mapped buffers.
};

// Find a memory type that satisfies both the buffer's requirements and the
// caller's property flags (e.g. HOST_VISIBLE | HOST_COHERENT).
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties);

// Create a buffer and allocate + bind memory for it.  If the memory is
// host-visible, it is persistently mapped.
Buffer createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memoryProperties);

// Upload `dataSize` bytes from `data` to `dst` through a temporary staging
// buffer.  The copy is recorded into a one-shot command buffer, submitted on
// the context's queue and waited on with a fence.
void uploadToDeviceLocal(const VulkanContext& ctx, Buffer& dst, const void* data,
                         VkDeviceSize dataSize);

// Free the buffer and its memory.  Resets the struct to null handles.
void destroyBuffer(VkDevice device, Buffer& buf);
