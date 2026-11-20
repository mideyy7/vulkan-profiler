#include "buffers.h"
#include "vk_context.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<uint32_t> loadShader(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open shader: " + path);
    }
    const size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % sizeof(uint32_t) != 0) {
        throw std::runtime_error("shader file size is not a multiple of 4: " + path);
    }
    std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize));
    return code;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size() * sizeof(uint32_t);
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

// ---------------------------------------------------------------------------
// --info
// ---------------------------------------------------------------------------

int runInfo() {
    {
        VulkanContext context;
        context.printDeviceInfo();
    }
    const uint32_t messages = validationMessageCount();
    std::printf("Validation messages:          %u\n", messages);
    return messages == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Tests from Phase 1
// ---------------------------------------------------------------------------

bool testCleanShutdown() {
    {
        VulkanContext context;
    }
    const uint32_t messages = validationMessageCount();
    std::printf("validation messages: %u (expected 0)\n", messages);
    return messages == 0;
}

bool testValidationLive() {
    VulkanContext context;
    const uint32_t before = validationMessageCount();

    std::printf("The validation error printed below is expected.\n");
    std::fflush(stdout);
    VkFenceCreateInfo invalidInfo{};
    invalidInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    invalidInfo.flags = 0x40000000;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(context.device(), &invalidInfo, nullptr, &fence);
    vkDestroyFence(context.device(), fence, nullptr);

    const uint32_t reported = validationMessageCount() - before;
    std::printf("messages from the invalid call: %u (expected at least 1)\n", reported);
    return reported > 0;
}

bool testValidationLeak() {
    std::printf("The validation errors printed below are expected.\n");
    std::fflush(stdout);
    {
        VulkanContext context;
        VkFenceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence leakedFence = VK_NULL_HANDLE;
        vkCreateFence(context.device(), &info, nullptr, &leakedFence);
    }
    const uint32_t messages = validationMessageCount();
    std::printf("messages from the leak: %u (expected at least 1)\n", messages);
    return messages > 0;
}

// ---------------------------------------------------------------------------
// Vector add test (Phase 2)
// ---------------------------------------------------------------------------

bool testVectorAdd() {
    // Test N values including edge cases: not a multiple of the workgroup size (64).
    const uint32_t testSizes[] = {1, 63, 64, 65, 1000};
    bool allPassed = true;

    for (uint32_t N : testSizes) {
        VulkanContext ctx;
        const uint32_t before = validationMessageCount();

        const VkDeviceSize bufSize = N * sizeof(float);

        // Prepare input data.
        std::vector<float> hostA(N), hostB(N), expected(N);
        for (uint32_t i = 0; i < N; ++i) {
            hostA[i] = static_cast<float>(i) * 1.5f;
            hostB[i] = static_cast<float>(i) * 0.5f + 1.0f;
            expected[i] = hostA[i] + hostB[i];
        }

        // Create device-local buffers for a, b and c.
        Buffer bufA = createBuffer(ctx, bufSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer bufB = createBuffer(ctx, bufSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer bufC = createBuffer(ctx, bufSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Host-visible readback buffer.
        Buffer readback = createBuffer(ctx, bufSize,
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Upload input data.
        uploadToDeviceLocal(ctx, bufA, hostA.data(), bufSize);
        uploadToDeviceLocal(ctx, bufB, hostB.data(), bufSize);

        // Load the shader and create the pipeline.
        const std::vector<uint32_t> code = loadShader("shaders/vector_add.comp.spv");
        VkShaderModule shaderModule = createShaderModule(ctx.device(), code);

        // Descriptor set layout: 3 storage buffers at bindings 0, 1, 2.
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int b = 0; b < 3; ++b) {
            bindings[b].binding = static_cast<uint32_t>(b);
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
        setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayoutInfo.bindingCount = 3;
        setLayoutInfo.pBindings = bindings;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        check(vkCreateDescriptorSetLayout(ctx.device(), &setLayoutInfo, nullptr, &setLayout),
              "vkCreateDescriptorSetLayout");

        // Push constant range for the element count.
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &setLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        check(vkCreatePipelineLayout(ctx.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
              "vkCreatePipelineLayout");

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        check(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                       &pipeline),
              "vkCreateComputePipelines");

        // Descriptor pool and set.
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        check(vkCreateDescriptorPool(ctx.device(), &poolInfo, nullptr, &descriptorPool),
              "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo setAllocInfo{};
        setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocInfo.descriptorPool = descriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &setLayout;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(ctx.device(), &setAllocInfo, &descriptorSet),
              "vkAllocateDescriptorSets");

        // Write the descriptor set to point at our three buffers.
        VkDescriptorBufferInfo bufInfos[3]{};
        bufInfos[0] = {bufA.buffer, 0, bufSize};
        bufInfos[1] = {bufB.buffer, 0, bufSize};
        bufInfos[2] = {bufC.buffer, 0, bufSize};
        VkWriteDescriptorSet writes[3]{};
        for (int w = 0; w < 3; ++w) {
            writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[w].dstSet = descriptorSet;
            writes[w].dstBinding = static_cast<uint32_t>(w);
            writes[w].descriptorCount = 1;
            writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[w].pBufferInfo = &bufInfos[w];
        }
        vkUpdateDescriptorSets(ctx.device(), 3, writes, 0, nullptr);

        // Record the command buffer: dispatch + copy to readback.
        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = ctx.queueFamilyIndex();
        VkCommandPool cmdPool = VK_NULL_HANDLE;
        check(vkCreateCommandPool(ctx.device(), &cmdPoolInfo, nullptr, &cmdPool),
              "vkCreateCommandPool");

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = cmdPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx.device(), &cmdAllocInfo, &cmd),
              "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
                           &N);
        vkCmdDispatch(cmd, (N + 63) / 64, 1, 1);

        // Barrier: COMPUTE → TRANSFER so the copy reads the dispatch's writes.
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = bufC.buffer;
        barrier.offset = 0;
        barrier.size = bufSize;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0,
                             nullptr);

        // Copy result to readback buffer.
        VkBufferCopy region{};
        region.size = bufSize;
        vkCmdCopyBuffer(cmd, bufC.buffer, readback.buffer, 1, &region);

        // Barrier: TRANSFER → HOST (C6) so the CPU read sees the copy's writes.
        VkBufferMemoryBarrier hostBarrier{};
        hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.buffer = readback.buffer;
        hostBarrier.offset = 0;
        hostBarrier.size = bufSize;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hostBarrier, 0,
                             nullptr);

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

        // Read back and compare.
        const auto* result = static_cast<const float*>(readback.mapped);
        bool passed = true;
        for (uint32_t i = 0; i < N; ++i) {
            if (result[i] != expected[i]) {
                std::printf("  N=%u: MISMATCH at [%u]: got %f, expected %f\n", N, i, result[i],
                            expected[i]);
                passed = false;
                break;
            }
        }

        // Check for validation messages during this run.
        const uint32_t newMessages = validationMessageCount() - before;
        if (newMessages > 0) {
            std::printf("  N=%u: %u validation messages\n", N, newMessages);
            passed = false;
        }

        std::printf("  N=%u: %s\n", N, passed ? "PASS" : "FAIL");
        if (!passed) allPassed = false;

        // Clean up.
        vkDestroyFence(ctx.device(), fence, nullptr);
        vkDestroyCommandPool(ctx.device(), cmdPool, nullptr);
        vkDestroyPipeline(ctx.device(), pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device(), pipelineLayout, nullptr);
        vkDestroyDescriptorPool(ctx.device(), descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device(), setLayout, nullptr);
        vkDestroyShaderModule(ctx.device(), shaderModule, nullptr);
        destroyBuffer(ctx.device(), readback);
        destroyBuffer(ctx.device(), bufC);
        destroyBuffer(ctx.device(), bufB);
        destroyBuffer(ctx.device(), bufA);
    }

    return allPassed;
}

// ---------------------------------------------------------------------------
// Test dispatch
// ---------------------------------------------------------------------------

int runTest(const std::string& name) {
    bool passed = false;
    if (name == "clean_shutdown") {
        passed = testCleanShutdown();
    } else if (name == "validation_live") {
        passed = testValidationLive();
    } else if (name == "validation_leak") {
        passed = testValidationLeak();
    } else if (name == "vector_add") {
        passed = testVectorAdd();
    } else {
        std::fprintf(stderr, "Unknown test: %s\n", name.c_str());
        return 2;
    }
    std::printf("%s: %s\n", name.c_str(), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

int printUsage() {
    std::fprintf(stderr, "Usage: gpu_profiler --info\n");
    std::fprintf(stderr, "       gpu_profiler --test <name>\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (argc == 2 && mode == "--info") {
            return runInfo();
        }
        if (argc == 3 && mode == "--test") {
            return runTest(argv[2]);
        }
        return printUsage();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
