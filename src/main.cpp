#include "buffers.h"
#include "report.h"
#include "simulation.h"
#include "vk_context.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
        const std::vector<uint32_t> code = loadShader(shaderPath("vector_add.comp.spv"));
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
// Simulation tests (Phases 3 to 6)
// ---------------------------------------------------------------------------

constexpr uint32_t kTestSeeds = 5;

// GPU forces against the double-precision CPU reference (C3, C4).
bool testForce(ForceKernel kernel) {
    const uint32_t sizes[] = {1, 63, 64, 65, 1000};
    constexpr double kLimit = 1e-5;
    VulkanContext ctx;
    bool passed = true;

    for (uint32_t n : sizes) {
        double worst = 0.0;
        for (uint32_t seed = 1; seed <= kTestSeeds; ++seed) {
            const std::vector<Particle> particles = generateParticles(n, seed);
            Simulation sim(ctx, particles, kernel);
            const std::vector<std::array<float, 4>> gpu = sim.computeAndReadForces();
            const std::vector<Vec3d> cpu = cpuForces(particles);

            for (uint32_t i = 0; i < n; ++i) {
                const double dx = gpu[i][0] - cpu[i].x;
                const double dy = gpu[i][1] - cpu[i].y;
                const double dz = gpu[i][2] - cpu[i].z;
                const double errorNorm = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (n == 1) {
                    // A lone particle feels no force, so anything but zero is wrong (C4).
                    if (errorNorm != 0.0) {
                        worst = INFINITY;
                    }
                    continue;
                }
                const double cpuNorm =
                    std::sqrt(cpu[i].x * cpu[i].x + cpu[i].y * cpu[i].y + cpu[i].z * cpu[i].z);
                worst = std::max(worst, errorNorm / cpuNorm);
            }
        }
        const bool ok = worst < kLimit;
        std::printf("  N=%-5u max relative error %.3e over %u seeds: %s\n", n, worst,
                    kTestSeeds, ok ? "PASS" : "FAIL");
        passed = passed && ok;
    }
    return passed;
}

// Total momentum before and after 100 steps (C5).
bool testMomentum() {
    const uint32_t sizes[] = {8, 64, 256};
    constexpr uint32_t kSteps = 100;
    constexpr double kLimit = 1e-5;
    VulkanContext ctx;
    bool passed = true;

    auto momentum = [](const std::vector<Particle>& ps) {
        Vec3d p{0.0, 0.0, 0.0};
        for (const Particle& q : ps) {
            p.x += static_cast<double>(q.mass) * q.vx;
            p.y += static_cast<double>(q.mass) * q.vy;
            p.z += static_cast<double>(q.mass) * q.vz;
        }
        return p;
    };

    for (uint32_t n : sizes) {
        double worst = 0.0;
        for (uint32_t seed = 1; seed <= kTestSeeds; ++seed) {
            const std::vector<Particle> initial = generateParticles(n, seed);
            double scale = 0.0;  // Sum of m|v|, never close to zero (C5).
            for (const Particle& q : initial) {
                scale += static_cast<double>(q.mass) *
                         std::sqrt(static_cast<double>(q.vx) * q.vx +
                                   static_cast<double>(q.vy) * q.vy +
                                   static_cast<double>(q.vz) * q.vz);
            }
            Simulation sim(ctx, initial);
            for (uint32_t s = 0; s < kSteps; ++s) {
                sim.runBatch(1);
            }
            const Vec3d before = momentum(initial);
            const Vec3d after = momentum(sim.readParticles());
            const double dx = after.x - before.x;
            const double dy = after.y - before.y;
            const double dz = after.z - before.z;
            worst = std::max(worst, std::sqrt(dx * dx + dy * dy + dz * dz) / scale);
        }
        const bool ok = worst < kLimit;
        std::printf("  N=%-4u relative momentum drift %.3e after %u steps: %s\n", n, worst,
                    kSteps, ok ? "PASS" : "FAIL");
        passed = passed && ok;
    }
    return passed;
}

// A deliberately missing barrier must be reported (C2).
bool testValidationCanary() {
    VulkanContext ctx;
    Simulation sim(ctx, generateParticles(256, 1));
    std::printf("The SYNC-HAZARD error printed below is expected.\n");
    std::fflush(stdout);
    const uint32_t before = syncHazardMessageCount();
    sim.stepMissingBarrier();
    const uint32_t hazards = syncHazardMessageCount() - before;
    std::printf("SYNC-HAZARD messages: %u (expected at least 1)\n", hazards);
    return hazards > 0;
}

// Timestamps must be in order, each stage short, and the GPU span inside the frame.
bool testTimestamps() {
    VulkanContext ctx;
    bool passed = true;
    for (uint32_t steps : {1u, 8u}) {
        Simulation sim(ctx, generateParticles(1024, 1));
        for (int run = 0; run < 5; ++run) {
            const BatchSample b = sim.runBatch(steps);
            for (size_t q = 0; q + 1 < b.ticks.size(); ++q) {
                if (b.ticks[q + 1] < b.ticks[q]) {
                    std::printf("  batch %u: timestamp %zu goes backwards\n", steps, q + 1);
                    passed = false;
                }
                const double stageNs =
                    static_cast<double>(b.ticks[q + 1] - b.ticks[q]) * b.nsPerTick;
                if (stageNs > 1e9) {
                    std::printf("  batch %u: stage %zu took %.0f ns\n", steps, q, stageNs);
                    passed = false;
                }
            }
            const double gpuNs =
                static_cast<double>(b.ticks.back() - b.ticks.front()) * b.nsPerTick;
            if (gpuNs > b.frameNs) {
                std::printf("  batch %u: GPU span %.0f ns exceeds frame %.0f ns\n", steps, gpuNs,
                            b.frameNs);
                passed = false;
            }
        }
        std::printf("  batch %u: checked 5 submissions\n", steps);
    }
    return passed;
}

// Statistics against hand-computed values.
bool testStats() {
    std::vector<double> hundred;
    for (int v = 100; v >= 1; --v) {
        hundred.push_back(v);
    }
    const Stats a = computeStats(hundred);
    const Stats b = computeStats({3.0, 1.0, 2.0});
    const bool passed = a.mean == 50.5 && a.min == 1 && a.p50 == 50 && a.p99 == 99 &&
                        a.max == 100 && b.mean == 2 && b.p50 == 2 && b.p99 == 3;
    std::printf("  1..100: mean %.1f min %.0f p50 %.0f p99 %.0f max %.0f\n", a.mean, a.min,
                a.p50, a.p99, a.max);
    std::printf("  {3,1,2}: mean %.1f p50 %.0f p99 %.0f\n", b.mean, b.p50, b.p99);
    return passed;
}

// Batched and unbatched runs must give bit-identical particles (C9).
bool testBatchEquivalence() {
    constexpr uint32_t kTotalSteps = 800;  // Divisible by 16, 25 and 32.
    const std::vector<Particle> initial = generateParticles(256, 1);
    VulkanContext ctx;

    Simulation reference(ctx, initial);
    for (uint32_t s = 0; s < kTotalSteps; ++s) {
        reference.runBatch(1);
    }
    const std::vector<Particle> expected = reference.readParticles();

    bool passed = true;
    for (uint32_t K : {16u, 25u, 32u}) {
        Simulation batched(ctx, initial);
        for (uint32_t s = 0; s < kTotalSteps; s += K) {
            batched.runBatch(K);
        }
        const std::vector<Particle> got = batched.readParticles();
        double maxDelta = 0.0;
        const float* x = &expected[0].px;
        const float* y = &got[0].px;
        for (size_t f = 0; f < expected.size() * 8; ++f) {
            maxDelta = std::max(maxDelta, std::fabs(static_cast<double>(x[f]) - y[f]));
        }
        const bool identical =
            std::memcmp(expected.data(), got.data(), expected.size() * sizeof(Particle)) == 0;
        std::printf("  K=%-2u max|delta| = %g, bytes identical: %s\n", K, maxDelta,
                    identical ? "yes" : "no");
        passed = passed && identical && maxDelta == 0.0;
    }
    return passed;
}

// ---------------------------------------------------------------------------
// --report and --sweep
// ---------------------------------------------------------------------------

struct Options {
    uint32_t particles = 2048;
    uint32_t batch = 1;
    uint32_t steps = 200;
    uint32_t warmup = 20;
    uint32_t seed = 1;
    uint32_t maxParticles = 32768;
    ForceKernel kernel = ForceKernel::Naive;
    std::string csv;
    bool validation = true;
};

uint32_t ceilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// Warm-up then measured batches (C8). Returns per-stage rows.
std::vector<StageRow> measure(VulkanContext& ctx, const Options& o, uint32_t particles,
                              RunInfo& info) {
    Simulation sim(ctx, generateParticles(particles, o.seed), o.kernel);
    const uint32_t warmupBatches = ceilDiv(o.warmup, o.batch);
    const uint32_t measuredBatches = ceilDiv(o.steps, o.batch);
    for (uint32_t b = 0; b < warmupBatches; ++b) {
        sim.runBatch(o.batch);
    }
    std::vector<BatchSample> samples;
    for (uint32_t b = 0; b < measuredBatches; ++b) {
        samples.push_back(sim.runBatch(o.batch));
    }

    info.device = ctx.deviceName();
    info.driver = ctx.driverDescription();
    info.particles = particles;
    info.batch = o.batch;
    info.kernel = kernelName(o.kernel);
    info.warmupSteps = warmupBatches * o.batch;
    info.measuredSteps = measuredBatches * o.batch;
    info.validation = ctx.validationEnabled();
    return buildStageRows(samples);
}

int finishRun(bool validation) {
    if (validation && validationMessageCount() > 0) {
        std::printf("FAIL: %u validation messages\n", validationMessageCount());
        return 1;
    }
    return 0;
}

int runReport(const Options& o) {
    {
        VulkanContext ctx(o.validation);
        RunInfo info;
        const std::vector<StageRow> rows = measure(ctx, o, o.particles, info);
        printReport(info, rows);
        if (!o.csv.empty()) {
            writeReportCsv(o.csv, info, rows);
            std::printf("\nWrote %s\n", o.csv.c_str());
        }
    }
    return finishRun(o.validation);
}

int runSweep(const Options& o) {
    {
        VulkanContext ctx(o.validation);
        std::ofstream csv;
        bool wroteHeader = false;
        if (!o.csv.empty()) {
            csv.open(o.csv);
            if (!csv) {
                throw std::runtime_error("cannot write " + o.csv);
            }
        }
        std::printf("%8s %6s %7s %12s %12s %12s %12s\n", "N", "batch", "kernel",
                    "frame_us", "frame_p99", "gpu_us", "overhead_us");
        for (uint32_t n = 256; n <= o.maxParticles; n *= 2) {
            RunInfo info;
            const std::vector<StageRow> rows = measure(ctx, o, n, info);
            const Stats frame = findRow(rows, "frame").stats;
            const Stats gpu = findRow(rows, "gpu span").stats;
            const Stats overhead = findRow(rows, "overhead (frame - gpu)").stats;
            std::printf("%8u %6u %7s %12.2f %12.2f %12.2f %12.2f\n", n, o.batch,
                        info.kernel.c_str(), frame.mean / 1e3, frame.p99 / 1e3, gpu.mean / 1e3,
                        overhead.mean / 1e3);
            std::fflush(stdout);
            if (csv.is_open()) {
                if (!wroteHeader) {
                    info.particles = 0;  // Varies per row.
                    csv << csvMetadata(info);
                    csv << "n,batch,kernel,measured_steps,frame_mean_ns,frame_p99_ns,"
                           "gpu_mean_ns,overhead_mean_ns,force_mean_ns,integrate_mean_ns,"
                           "submit_mean_ns,wait_mean_ns\n";
                    wroteHeader = true;
                }
                csv << n << ',' << o.batch << ',' << kernelName(o.kernel) << ','
                    << info.measuredSteps << ',' << frame.mean << ',' << frame.p99 << ','
                    << gpu.mean << ',' << overhead.mean << ','
                    << findRow(rows, "force").stats.mean << ','
                    << findRow(rows, "integrate").stats.mean << ','
                    << findRow(rows, "cpu submit").stats.mean << ','
                    << findRow(rows, "cpu fence wait").stats.mean << '\n';
            }
        }
        if (csv.is_open()) {
            std::printf("Wrote %s\n", o.csv.c_str());
        }
    }
    return finishRun(o.validation);
}

uint32_t parseCount(const std::string& flag, const char* value) {
    const unsigned long parsed = std::stoul(value);
    if (parsed == 0 || parsed > 1000000) {
        throw std::runtime_error(flag + " must be between 1 and 1000000");
    }
    return static_cast<uint32_t>(parsed);
}

Options parseOptions(int argc, char** argv, int first) {
    Options o;
    for (int a = first; a < argc; ++a) {
        const std::string flag = argv[a];
        if (flag == "--no-validation") {
            o.validation = false;
            continue;
        }
        if (a + 1 >= argc) {
            throw std::runtime_error("missing value after " + flag);
        }
        const char* value = argv[++a];
        if (flag == "--n") {
            o.particles = parseCount(flag, value);
        } else if (flag == "--batch") {
            o.batch = parseCount(flag, value);
        } else if (flag == "--steps") {
            o.steps = parseCount(flag, value);
        } else if (flag == "--warmup") {
            o.warmup = parseCount(flag, value);
        } else if (flag == "--seed") {
            o.seed = parseCount(flag, value);
        } else if (flag == "--max-n") {
            o.maxParticles = parseCount(flag, value);
        } else if (flag == "--csv") {
            o.csv = value;
        } else if (flag == "--kernel") {
            const std::string kernel = value;
            if (kernel == "naive") {
                o.kernel = ForceKernel::Naive;
            } else if (kernel == "tiled") {
                o.kernel = ForceKernel::Tiled;
            } else {
                throw std::runtime_error("--kernel must be naive or tiled");
            }
        } else {
            throw std::runtime_error("unknown option " + flag);
        }
    }
    return o;
}

// ---------------------------------------------------------------------------
// Test dispatch
// ---------------------------------------------------------------------------

int runTest(const std::string& name) {
    // These tests deliberately cause validation messages, so they are not
    // also required to finish with zero messages.
    const bool expectsMessages =
        name == "validation_live" || name == "validation_leak" || name == "validation_canary";

    bool passed = false;
    if (name == "clean_shutdown") {
        passed = testCleanShutdown();
    } else if (name == "validation_live") {
        passed = testValidationLive();
    } else if (name == "validation_leak") {
        passed = testValidationLeak();
    } else if (name == "vector_add") {
        passed = testVectorAdd();
    } else if (name == "force_naive") {
        passed = testForce(ForceKernel::Naive);
    } else if (name == "force_tiled") {
        passed = testForce(ForceKernel::Tiled);
    } else if (name == "momentum") {
        passed = testMomentum();
    } else if (name == "validation_canary") {
        passed = testValidationCanary();
    } else if (name == "timestamps") {
        passed = testTimestamps();
    } else if (name == "stats") {
        passed = testStats();
    } else if (name == "batch_equivalence") {
        passed = testBatchEquivalence();
    } else {
        std::fprintf(stderr, "Unknown test: %s\n", name.c_str());
        return 2;
    }
    if (!expectsMessages && validationMessageCount() > 0) {
        std::printf("%u validation messages\n", validationMessageCount());
        passed = false;
    }
    std::printf("%s: %s\n", name.c_str(), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

int printUsage() {
    std::fprintf(stderr,
                 "Usage: gpu_profiler --info\n"
                 "       gpu_profiler --test <name>\n"
                 "       gpu_profiler --report [options]\n"
                 "       gpu_profiler --sweep [options]\n"
                 "Options: --n N  --batch K  --kernel naive|tiled  --steps S  --warmup W\n"
                 "         --seed S  --max-n N  --csv PATH  --no-validation\n");
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
        if (mode == "--report") {
            return runReport(parseOptions(argc, argv, 2));
        }
        if (mode == "--sweep") {
            return runSweep(parseOptions(argc, argv, 2));
        }
        return printUsage();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
