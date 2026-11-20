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
