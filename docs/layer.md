# VK_LAYER_stage_profiler

> Draft. Rewrite in your own words.

## How a layer intercepts calls

An app calls Vulkan through the loader. When a layer is enabled, the loader builds a chain: app → layer → … → driver. At `vkCreateInstance` and `vkCreateDevice`, the loader passes each layer a link struct in `pNext` that holds the next layer's `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr`. The layer looks up the next layer's functions, stores them in a per-instance or per-device dispatch table, and advances the link for the layer below.

After that, the loader asks the layer for function pointers through its `vkGetDeviceProcAddr`. For the functions it cares about, the layer returns its own versions, which record a count or a time and then call the stored next function. For everything else, it returns the next layer's pointer directly.

Each device's data is found from any handle belonging to it (device, queue or command buffer) through the **dispatch key**: the first pointer-sized word of every dispatchable handle points to the loader's dispatch table, which is shared by all objects of the same device.

## What it records

- CPU time inside `vkQueueSubmit`, `vkWaitForFences`, `vkQueueWaitIdle` and `vkDeviceWaitIdle`
- Number of `vkCmdDispatch`, `vkCmdPipelineBarrier` and `vkCmdCopyBuffer` commands recorded
- Number of command buffers submitted

The report prints when the app destroys the device.

## Why it doesn't add GPU timestamps to other apps

Writing timestamps would mean inserting commands into another app's command buffers and creating query pools on its device. That changes the work being measured, can break apps that reuse or submit command buffers in unexpected ways, and needs careful synchronisation to read the results back. The in-app profiler can place timestamps exactly because it owns the command buffers.

## Agreement with the app

`scripts/check_layer.py` runs the app with the layer and compares results (Apple M1):

- Dispatch, barrier and copy counts match exactly.
- 221 fence waits on both sides.
- Mean wait per fence: app 1,524.5 µs, layer 1,524.1 µs (0.03% apart).
- Zero validation messages with the validation layer stacked on top.
- Unmodified `vkcube --c 300` reported 301 `vkQueueSubmit` calls.

## Limitations

Newer function variants pass through without being counted: `vkQueueSubmit2`, `vkCmdPipelineBarrier2`, `vkCmdCopyBuffer2`, `vkCmdDispatchIndirect` and `vkCmdDispatchBase`.
