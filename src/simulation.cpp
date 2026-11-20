#include "simulation.h"

#include "buffers.h"
#include "vk_context.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Shader loading
// ---------------------------------------------------------------------------

std::string shaderPath(const std::string& fileName) {
    // GPU_PROFILER_SHADER_DIR is set by CMakeLists.txt to the build's shader folder.
    return std::string(GPU_PROFILER_SHADER_DIR) + "/" + fileName;
}

std::vector<uint32_t> loadShader(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open shader: " + path);
    }
    const auto fileSize = static_cast<size_t>(file.tellg());
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
// Particle generation (C13)
// ---------------------------------------------------------------------------

std::vector<Particle> generateParticles(uint32_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<Particle> particles(count);

    // Build floats directly from raw mt19937 output (C13) so the result is
    // identical on Mac and Linux regardless of standard-library differences.
    // The top 24 bits fit a float exactly, so t is in [0, 1) and never 1.
    auto nextFloat = [&](float lo, float hi) -> float {
        const uint32_t raw = rng();
        const float t = static_cast<float>(raw >> 8) * (1.0f / 16777216.0f);
        return lo + (hi - lo) * t;
    };

    // Masses scale with 1/N so the total mass, and so the typical
    // acceleration, stays similar at every particle count.
    const float massScale = 1.0f / static_cast<float>(count == 0 ? 1 : count);

    for (auto& p : particles) {
        p.px = nextFloat(-1.0f, 1.0f);
        p.py = nextFloat(-1.0f, 1.0f);
        p.pz = nextFloat(-1.0f, 1.0f);
        p.mass = nextFloat(0.5f, 1.5f) * massScale;
        p.vx = nextFloat(-0.1f, 0.1f);
        p.vy = nextFloat(-0.1f, 0.1f);
        p.vz = nextFloat(-0.1f, 0.1f);
        p.pad = 0.0f;
    }

    return particles;
}

// ---------------------------------------------------------------------------
// CPU force reference (C3) — double precision
// ---------------------------------------------------------------------------

std::vector<Vec3d> cpuForces(const std::vector<Particle>& particles) {
    const size_t N = particles.size();
    std::vector<Vec3d> acc(N, {0.0, 0.0, 0.0});

    const double G = static_cast<double>(kG);
    const double eps2 = static_cast<double>(kSoftening) * static_cast<double>(kSoftening);

    for (size_t i = 0; i < N; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        const double xi = static_cast<double>(particles[i].px);
        const double yi = static_cast<double>(particles[i].py);
        const double zi = static_cast<double>(particles[i].pz);

        for (size_t j = 0; j < N; ++j) {
            const double dx = static_cast<double>(particles[j].px) - xi;
            const double dy = static_cast<double>(particles[j].py) - yi;
            const double dz = static_cast<double>(particles[j].pz) - zi;
            const double dist2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_dist3 = 1.0 / (dist2 * std::sqrt(dist2));
            const double f = G * static_cast<double>(particles[j].mass) * inv_dist3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }

        acc[i] = {ax, ay, az};
    }

    return acc;
}

// ---------------------------------------------------------------------------
// Simulation — construction and destruction
// ---------------------------------------------------------------------------

Simulation::Simulation(VulkanContext& ctx, const std::vector<Particle>& particles)
    : ctx_(ctx),
      particleCount_(static_cast<uint32_t>(particles.size())),
      groups_((particleCount_ + kWorkgroupSize - 1) / kWorkgroupSize) {
    try {
        create(particles);
    } catch (...) {
        // A destructor does not run when its constructor throws.
        destroy();
        throw;
    }
}

Simulation::~Simulation() {
    destroy();
}

void Simulation::create(const std::vector<Particle>& particles) {
    const VkDeviceSize particleSize = particleCount_ * sizeof(Particle);
    const VkDeviceSize accelSize = particleCount_ * 4 * sizeof(float);

    // --- Buffers ---
    particleBuf_ = createBuffer(ctx_, particleSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    accelBuf_ = createBuffer(ctx_, accelSize,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    particleReadback_ = createBuffer(ctx_, particleSize,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    accelReadback_ = createBuffer(ctx_, accelSize,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    uploadToDeviceLocal(ctx_, particleBuf_, particles.data(), particleSize);

    // --- Shader modules ---
    forceModule_ =
        createShaderModule(ctx_.device(), loadShader(shaderPath("force_calc.comp.spv")));
    integrateModule_ =
        createShaderModule(ctx_.device(), loadShader(shaderPath("integrate.comp.spv")));

    // --- Descriptor set layout: 2 storage buffers ---
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (int b = 0; b < 2; ++b) {
        bindings[b].binding = static_cast<uint32_t>(b);
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(ctx_.device(), &layoutInfo, nullptr, &descSetLayout_),
          "vkCreateDescriptorSetLayout");

    // --- Pipeline layout with push constants ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &descSetLayout_;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushRange;
    check(vkCreatePipelineLayout(ctx_.device(), &pipeLayoutInfo, nullptr, &pipelineLayout_),
          "vkCreatePipelineLayout");

    // --- Compute pipelines ---
    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = pipelineLayout_;

    pipeInfo.stage.module = forceModule_;
    check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                   &forcePipeline_),
          "vkCreateComputePipelines (force)");

    pipeInfo.stage.module = integrateModule_;
    check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                   &integratePipeline_),
          "vkCreateComputePipelines (integrate)");

    // --- Descriptor pool and set ---
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(ctx_.device(), &poolInfo, nullptr, &descPool_),
          "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo setAllocInfo{};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = descPool_;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts = &descSetLayout_;
    check(vkAllocateDescriptorSets(ctx_.device(), &setAllocInfo, &descSet_),
          "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo bufInfos[2]{};
    bufInfos[0] = {particleBuf_.buffer, 0, particleSize};
    bufInfos[1] = {accelBuf_.buffer, 0, accelSize};
    VkWriteDescriptorSet writes[2]{};
    for (int w = 0; w < 2; ++w) {
        writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[w].dstSet = descSet_;
        writes[w].dstBinding = static_cast<uint32_t>(w);
        writes[w].descriptorCount = 1;
        writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[w].pBufferInfo = &bufInfos[w];
    }
    vkUpdateDescriptorSets(ctx_.device(), 2, writes, 0, nullptr);

    // --- Command pool, buffer, fence ---
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdPoolInfo.queueFamilyIndex = ctx_.queueFamilyIndex();
    check(vkCreateCommandPool(ctx_.device(), &cmdPoolInfo, nullptr, &cmdPool_),
          "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(ctx_.device(), &cmdAllocInfo, &cmd_),
          "vkAllocateCommandBuffers");

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(vkCreateFence(ctx_.device(), &fenceInfo, nullptr, &fence_), "vkCreateFence");
}

void Simulation::destroy() {
    const VkDevice device = ctx_.device();
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device, fence_, nullptr);
    if (cmdPool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmdPool_, nullptr);
    if (descPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descPool_, nullptr);
    if (integratePipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device, integratePipeline_, nullptr);
    if (forcePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device, forcePipeline_, nullptr);
    if (pipelineLayout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
    if (descSetLayout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, descSetLayout_, nullptr);
    if (integrateModule_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device, integrateModule_, nullptr);
    if (forceModule_ != VK_NULL_HANDLE) vkDestroyShaderModule(device, forceModule_, nullptr);
    fence_ = VK_NULL_HANDLE;
    cmdPool_ = VK_NULL_HANDLE;
    descPool_ = VK_NULL_HANDLE;
    integratePipeline_ = VK_NULL_HANDLE;
    forcePipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    descSetLayout_ = VK_NULL_HANDLE;
    integrateModule_ = VK_NULL_HANDLE;
    forceModule_ = VK_NULL_HANDLE;
    destroyBuffer(device, accelReadback_);
    destroyBuffer(device, particleReadback_);
    destroyBuffer(device, accelBuf_);
    destroyBuffer(device, particleBuf_);
}

// ---------------------------------------------------------------------------
// Simulation — recording helpers
// ---------------------------------------------------------------------------

void Simulation::beginCommands(VkCommandBufferUsageFlags flags) {
    check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = flags;
    check(vkBeginCommandBuffer(cmd_, &beginInfo), "vkBeginCommandBuffer");

    // Push constants and the descriptor set stay bound across pipeline
    // changes because both pipelines share one pipeline layout.
    const PushConstants pc{particleCount_, kG, kSoftening, kDt};
    vkCmdPushConstants(cmd_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1,
                            &descSet_, 0, nullptr);
}

void Simulation::recordForceDispatch() {
    // Force dispatch: reads particles, writes accelerations.
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, forcePipeline_);
    cmdDispatch(cmd_, groups_);
}

void Simulation::recordForceBarrier() {
    // Barrier: accel buffer SHADER_WRITE → SHADER_READ so integrate sees
    // the forces that were just computed.
    VkBufferMemoryBarrier forceToIntegrate{};
    forceToIntegrate.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    forceToIntegrate.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    forceToIntegrate.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    forceToIntegrate.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    forceToIntegrate.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    forceToIntegrate.buffer = accelBuf_.buffer;
    forceToIntegrate.offset = 0;
    forceToIntegrate.size = VK_WHOLE_SIZE;
    cmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, nullptr, 1, &forceToIntegrate);
}

void Simulation::recordIntegrateDispatch() {
    // Integrate dispatch: reads accel, reads and writes particles.
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline_);
    cmdDispatch(cmd_, groups_);
}

void Simulation::recordStepBoundaryBarrier(VkPipelineStageFlags srcStages) {
    // Step-boundary barrier (C7): the next step reads and writes both buffers,
    // so cover both with SHADER_WRITE → SHADER_READ | SHADER_WRITE.
    VkBufferMemoryBarrier stepBarriers[2]{};
    for (int b = 0; b < 2; ++b) {
        stepBarriers[b].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stepBarriers[b].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        stepBarriers[b].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        stepBarriers[b].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stepBarriers[b].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stepBarriers[b].offset = 0;
        stepBarriers[b].size = VK_WHOLE_SIZE;
    }
    stepBarriers[0].buffer = particleBuf_.buffer;
    stepBarriers[1].buffer = accelBuf_.buffer;
    cmdPipelineBarrier(cmd_, srcStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, nullptr, 2,
                       stepBarriers);
}

void Simulation::recordBatch(uint32_t steps) {
    // No ONE_TIME_SUBMIT flag: this command buffer is submitted many times.
    beginCommands(0);

    for (uint32_t k = 0; k < steps; ++k) {
        recordForceDispatch();
        recordForceBarrier();
        recordIntegrateDispatch();
        if (k + 1 < steps) {
            recordStepBoundaryBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
    }

    // Readback: particles SHADER_WRITE → TRANSFER_READ, copy, then TRANSFER → HOST (C6).
    VkBufferMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.buffer = particleBuf_.buffer;
    toTransfer.offset = 0;
    toTransfer.size = VK_WHOLE_SIZE;
    cmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, 1, &toTransfer);

    cmdCopyBuffer(cmd_, particleBuf_.buffer, particleReadback_.buffer, particleBuf_.size);

    VkBufferMemoryBarrier toHost{};
    toHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHost.buffer = particleReadback_.buffer;
    toHost.offset = 0;
    toHost.size = VK_WHOLE_SIZE;
    cmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                       nullptr, 1, &toHost);

    // The next submission's first step must wait for this one's integrate
    // writes and readback copy (C7).
    recordStepBoundaryBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_TRANSFER_BIT);

    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    recordedSteps_ = steps;
}

void Simulation::submitAndWait() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd_;

    check(vkResetFences(ctx_.device(), 1, &fence_), "vkResetFences");
    check(vkQueueSubmit(ctx_.queue(), 1, &submitInfo, fence_), "vkQueueSubmit");
    waitForFence(ctx_.device(), fence_);
}

// ---------------------------------------------------------------------------
// Simulation — public methods
// ---------------------------------------------------------------------------

std::vector<std::array<float, 4>> Simulation::computeAndReadForces() {
    const VkDeviceSize accelSize = particleCount_ * 4 * sizeof(float);

    beginCommands(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    recordForceDispatch();

    // Barrier: COMPUTE SHADER_WRITE → TRANSFER TRANSFER_READ on accel buffer.
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = accelBuf_.buffer;
    barrier.offset = 0;
    barrier.size = accelSize;
    cmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, 1, &barrier);

    cmdCopyBuffer(cmd_, accelBuf_.buffer, accelReadback_.buffer, accelSize);

    // Barrier: TRANSFER → HOST (C6).
    VkBufferMemoryBarrier hostBarrier{};
    hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.buffer = accelReadback_.buffer;
    hostBarrier.offset = 0;
    hostBarrier.size = accelSize;
    cmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                       nullptr, 1, &hostBarrier);

    // Later work must wait for this copy before overwriting the accelerations.
    recordStepBoundaryBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_TRANSFER_BIT);

    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");
    recordedSteps_ = 0;
    submitAndWait();

    std::vector<std::array<float, 4>> result(particleCount_);
    std::memcpy(result.data(), accelReadback_.mapped, static_cast<size_t>(accelSize));
    return result;
}

void Simulation::runBatch(uint32_t steps) {
    if (steps == 0) {
        throw std::runtime_error("runBatch needs at least one step");
    }
    if (recordedSteps_ != steps) {
        recordBatch(steps);
    }
    submitAndWait();
}

std::vector<Particle> Simulation::readParticles() const {
    std::vector<Particle> result(particleCount_);
    std::memcpy(result.data(), particleReadback_.mapped,
                static_cast<size_t>(particleCount_ * sizeof(Particle)));
    return result;
}
