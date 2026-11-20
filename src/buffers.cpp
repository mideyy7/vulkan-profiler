#include "buffers.h"

#include "vk_context.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void check(VkResult result, const char* action) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(action) + " failed with VkResult " +
                                 std::to_string(result));
    }
}

}  // namespace

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("no suitable memory type found");
}

Buffer createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memoryProperties) {
    Buffer buf{};
    buf.size = size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx.device(), &bufferInfo, nullptr, &buf.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.device(), buf.buffer, &requirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(ctx.physicalDevice(), requirements.memoryTypeBits, memoryProperties);
    check(vkAllocateMemory(ctx.device(), &allocInfo, nullptr, &buf.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(ctx.device(), buf.buffer, buf.memory, 0), "vkBindBufferMemory");

    // Persistently map host-visible memory so we can read/write it at any time.
    if (memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        check(vkMapMemory(ctx.device(), buf.memory, 0, size, 0, &buf.mapped), "vkMapMemory");
    }

    return buf;
}

void uploadToDeviceLocal(const VulkanContext& ctx, Buffer& dst, const void* data,
                         VkDeviceSize dataSize) {
    // Create a host-visible staging buffer and copy data into it.
    Buffer staging = createBuffer(ctx, dataSize,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, data, static_cast<size_t>(dataSize));

    // Record a command buffer that copies staging → dst.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = ctx.queueFamilyIndex();
    VkCommandPool pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(ctx.device(), &poolInfo, nullptr, &pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(ctx.device(), &allocInfo, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer");

    VkBufferCopy region{};
    region.size = dataSize;
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);

    check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    // Submit and wait.
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(ctx.device(), &fenceInfo, nullptr, &fence), "vkCreateFence");
    check(vkQueueSubmit(ctx.queue(), 1, &submitInfo, fence), "vkQueueSubmit");
    check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    destroyBuffer(ctx.device(), staging);
}

void destroyBuffer(VkDevice device, Buffer& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
    }
    if (buf.memory != VK_NULL_HANDLE) {
        // Unmapping is implicit when the memory is freed.
        vkFreeMemory(device, buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
    buf.mapped = nullptr;
    buf.size = 0;
}
