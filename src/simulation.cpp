#include "simulation.h"

#include "buffers.h"
#include "vk_context.h"

#include <chrono>
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
