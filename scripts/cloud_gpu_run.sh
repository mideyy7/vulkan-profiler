#!/usr/bin/env bash
# One-shot run on a rented Ubuntu 24.04 machine with an NVIDIA GPU:
# installs the tools, checks that Vulkan sees the NVIDIA GPU, builds, runs the
# tests and every benchmark, and packs the results into linux-results.tgz.
#
# Usage (from the repository root):  bash scripts/cloud_gpu_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n=== %s\n' "$1"; }

step "1. NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found: this machine has no NVIDIA driver."
    echo "Choose a machine image that comes with the NVIDIA driver installed."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)

# Containers (e.g. Vast.ai) usually run as root without sudo installed.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

step "2. Build tools and Vulkan packages"
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    cmake g++ python3 git libvulkan-dev vulkan-tools vulkan-validationlayers glslc

# In containers the NVIDIA Vulkan library can be present without the small
# JSON file that tells the Vulkan loader where it is. Add it if missing.
if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1 \
    && ldconfig -p | grep -q 'libGLX_nvidia.so.0'; then
    echo "Adding the missing NVIDIA Vulkan driver manifest"
    $SUDO mkdir -p /etc/vulkan/icd.d
    printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}\n' \
        | $SUDO tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null
fi

step "3. Vulkan must see the NVIDIA GPU"
if ! vulkaninfo --summary 2>/dev/null | grep -qiE 'driverName *= *NVIDIA'; then
    echo "Vulkan does not see the NVIDIA GPU. Devices it does see:"
    vulkaninfo --summary 2>&1 | grep -E 'deviceName|driverName' || true
    echo
    echo "Likely fixes:"
    echo " - The driver was installed without its graphics/Vulkan part. Try:"
    echo "     $SUDO apt-get install -y libnvidia-gl-${driver_major}"
    echo " - Inside a container (e.g. Vast.ai), recreate it with the environment"
    echo "   variable NVIDIA_DRIVER_CAPABILITIES=all; the default hides Vulkan."
    exit 1
fi
vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo'

step "4. Build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

step "5. Tests (with validation on)"
tests_ok=yes
ctest --test-dir build --output-on-failure || tests_ok=no
ctest --test-dir build > docs-test-log.txt 2>&1 || true

step "6. Benchmarks (validation off)"
bash scripts/check_linux_gpu.sh || true
scripts/run_benchmarks.sh docs/linux
mv docs-test-log.txt docs/linux/ctest.txt
tar czf linux-results.tgz docs/linux

step "Done"
if [ "$tests_ok" = no ]; then
    echo "WARNING: some tests failed (see docs/linux/ctest.txt). Check them before trusting the numbers."
fi
echo "Results: $(pwd)/linux-results.tgz"
echo "Copy it to your Mac before stopping the machine, e.g.:"
echo "  scp <user>@<machine-address>:$(pwd)/linux-results.tgz ."
