#!/usr/bin/env bash
# Read-only check: can this Linux machine produce the project's real-GPU numbers?
# It installs nothing and changes nothing.

have() { command -v "$1" >/dev/null 2>&1; }

section() { printf '\n--- %s ---\n' "$1"; }

section "Machine"
uname -srm
[ -r /etc/os-release ] && grep '^PRETTY_NAME=' /etc/os-release
uptime

section "GPU hardware"
if have lspci; then
    lspci | grep -Ei 'vga|3d controller|display controller' || echo "none listed by lspci"
else
    echo "lspci not available"
fi
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
fi

section "Render device access"
render_ok=no
for node in /dev/dri/renderD*; do
    [ -e "$node" ] || continue
    if [ -r "$node" ] && [ -w "$node" ]; then
        echo "$node: accessible"
        render_ok=yes
    else
        echo "$node: no permission"
    fi
done
[ -e /dev/nvidia0 ] && [ -r /dev/nvidia0 ] && [ -w /dev/nvidia0 ] && {
    echo "/dev/nvidia0: accessible"
    render_ok=yes
}
[ "$render_ok" = yes ] || echo "no accessible GPU device found"

section "Vulkan drivers installed"
drivers=$(ls /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json 2>/dev/null)
if [ -n "$drivers" ]; then echo "$drivers"; else echo "none found"; fi
hardware_driver=no
if echo "$drivers" | grep -Eiv 'lvp|lavapipe|llvmpipe|^$' | grep -q .; then
    hardware_driver=yes
fi

section "Vulkan devices"
gpu_status=unknown
if have vulkaninfo; then
    summary=$(vulkaninfo --summary 2>/dev/null)
    echo "$summary" | grep -E '^\s*(deviceType|deviceName|driverName|driverInfo)\s*=' \
        || echo "vulkaninfo found no devices"
    gpu_status=$(echo "$summary" | awk -F'= *' '
        /^[[:space:]]*deviceType[[:space:]]*=/ { type = $2 }
        /^[[:space:]]*driverName[[:space:]]*=/ {
            name = tolower($2)
            if (type ~ /DISCRETE_GPU|INTEGRATED_GPU/ &&
                name !~ /llvmpipe|lavapipe|moltenvk/) found = 1
        }
        END { print (found ? "real" : "none") }')
else
    echo "vulkaninfo not installed"
fi

section "Build tools"
tools_ok=yes
for tool in cmake c++ glslc; do
    if have "$tool"; then
        echo "$tool: found"
    else
        echo "$tool: MISSING"
        tools_ok=no
    fi
done

header=no
for dir in /usr/include "${VULKAN_SDK:-/nonexistent}/include"; do
    [ -f "$dir/vulkan/vulkan.h" ] && header=yes
done
echo "vulkan headers: $([ "$header" = yes ] && echo found || echo MISSING)"
[ "$header" = yes ] || tools_ok=no

layer=no
for dir in /usr/share/vulkan/explicit_layer.d /etc/vulkan/explicit_layer.d \
           "${VULKAN_SDK:-/nonexistent}/share/vulkan/explicit_layer.d" \
           "${VULKAN_SDK:-/nonexistent}/etc/vulkan/explicit_layer.d"; do
    ls "$dir"/*khronos_validation*.json >/dev/null 2>&1 && layer=yes
done
echo "validation layer: $([ "$layer" = yes ] && echo found || echo MISSING)"
[ "$layer" = yes ] || tools_ok=no

section "Result"
if [ "$gpu_status" = real ] && [ "$tools_ok" = yes ]; then
    echo "READY"
elif [ "$gpu_status" = real ]; then
    echo "GPU OK, TOOLS MISSING"
elif [ "$gpu_status" = unknown ] && [ "$hardware_driver" = yes ] && [ "$render_ok" = yes ]; then
    echo "CHECK GPU"
else
    echo "NO REAL GPU"
fi
