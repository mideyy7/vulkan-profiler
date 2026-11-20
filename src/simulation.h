#pragma once

#include "buffers.h"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct Particle {
    float px, py, pz, mass;
    float vx, vy, vz, pad;
};
static_assert(sizeof(Particle) == 32, "Particle must be 32 bytes for std430 layout");

struct PushConstants {
    uint32_t count;
    float G;
    float softening;
    float dt;
};
static_assert(sizeof(PushConstants) == 16, "PushConstants must be 16 bytes");

// Simulation constants (from implementation.md).
inline constexpr float kG = 1.0f;
inline constexpr float kSoftening = 0.05f;
inline constexpr float kDt = 0.001f;
inline constexpr uint32_t kWorkgroupSize = 64;

// ---------------------------------------------------------------------------
// Shader loading
// ---------------------------------------------------------------------------

// Absolute path of a compiled shader, so the program works from any folder.
std::string shaderPath(const std::string& fileName);
std::vector<uint32_t> loadShader(const std::string& path);
VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code);

// ---------------------------------------------------------------------------
// Particle generation (C13: deterministic from raw mt19937 output)
// ---------------------------------------------------------------------------

std::vector<Particle> generateParticles(uint32_t count, uint32_t seed);

// ---------------------------------------------------------------------------
// CPU force reference in double precision (C3)
// ---------------------------------------------------------------------------

struct Vec3d {
    double x, y, z;
};

std::vector<Vec3d> cpuForces(const std::vector<Particle>& particles);

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

class Simulation {
public:
    Simulation(VulkanContext& ctx, const std::vector<Particle>& particles);
    ~Simulation();

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    // Run the force kernel only, then read back accelerations.
    // Returns one {ax, ay, az, 0} per particle.
    std::vector<std::array<float, 4>> computeAndReadForces();

    // Run `steps` full steps in one command buffer, one submit and one fence
    // wait, then copy the particles back. steps = 1 is the unbatched case.
    void runBatch(uint32_t steps);

    // One step with the force->integrate barrier deliberately left out (C2 canary).
    void stepMissingBarrier();

    // Particle state copied back by the most recent runBatch.
    std::vector<Particle> readParticles() const;

    uint32_t particleCount() const { return particleCount_; }

private:
    void create(const std::vector<Particle>& particles);
    void destroy();
    void beginCommands(VkCommandBufferUsageFlags flags);
    void recordForceDispatch();
    void recordForceBarrier();
    void recordIntegrateDispatch();
    void recordStepBoundaryBarrier(VkPipelineStageFlags srcStages);
    void recordBatch(uint32_t steps);
    void submitAndWait();

    VulkanContext& ctx_;
    uint32_t particleCount_;
    uint32_t groups_;

    Buffer particleBuf_{};
    Buffer accelBuf_{};
    Buffer particleReadback_{};
    Buffer accelReadback_{};

    VkShaderModule forceModule_ = VK_NULL_HANDLE;
    VkShaderModule integrateModule_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline forcePipeline_ = VK_NULL_HANDLE;
    VkPipeline integratePipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    // Steps recorded in cmd_ by recordBatch, or 0 if cmd_ holds something else.
    // A batch is recorded once and then resubmitted.
    uint32_t recordedSteps_ = 0;
};
