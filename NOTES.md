# Development Notes

## Thu 20 Nov — First compute on the GPU

Got a shader running on the GPU and reading the results back. Wrote `buffers.h/cpp` with helpers for allocating Vulkan buffers and uploading data through a staging copy. The vector add shader (`c[i] = a[i] + b[i]`) uses `std430` storage buffers and a push constant for the count.

The main new things I learned:
- **Memory types**: Vulkan separates the buffer from its memory. You have to find a memory type that's both compatible with the buffer and has the properties you need (device-local for GPU work, host-visible for CPU access).
- **Staging copies**: Device-local memory is fast for the GPU but the CPU can't map it, so you upload through a temporary host-visible buffer with a `vkCmdCopyBuffer`.
- **Descriptor sets**: This is how you tell the pipeline which buffers to bind to which shader bindings. You create a layout, allocate a set from a pool, then write the buffer info into it.
- **Barriers**: After the dispatch writes to the output buffer, a `COMPUTE→TRANSFER` barrier makes the writes visible to the copy. After the copy, a `TRANSFER→HOST` barrier (C6 from the implementation plan) makes the copy visible to the CPU read.

Tested with N = 1, 63, 64, 65 and 1000 — exact match on all, zero validation messages.
