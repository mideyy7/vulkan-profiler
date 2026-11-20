# Synchronisation

> Draft. Rewrite in your own words.

Vulkan doesn't order GPU work for you. Two commands in the same queue can overlap unless a barrier says otherwise. A barrier has two parts:

- **An execution dependency** (stage masks): work in the source stages must finish before work in the destination stages starts.
- **A memory dependency** (access masks): writes made by the source accesses are made visible to the destination accesses, so the second command doesn't read stale cached data.

## Every barrier in one step

| Where | Buffer | Stages | Access | Why |
| --- | --- | --- | --- | --- |
| Force → integrate | accelerations | compute → compute | shader write → shader read | Integrate reads the forces just written |
| Between steps (batch > 1) | particles and accelerations | compute → compute | shader write → shader read + write | Next force reads new positions; next force overwrites accelerations that integrate read |
| Before readback | particles | compute → transfer | shader write → transfer read | The copy reads the final positions |
| After readback | readback buffer | transfer → host | transfer write → host read | The CPU reads the copy (required by the spec even after a fence wait) |
| End of every submission | particles and accelerations | compute + transfer → compute | shader write → shader read + write | The next submission must not overwrite data this one's copy is still reading |
| After the upload copy | everything | transfer → compute + transfer | transfer write → shader read/write, transfer read | Initial particles are visible to the first dispatch in a later submission |

A fence wait only tells the CPU the GPU has finished. It doesn't make one submission's writes visible to the next submission's shaders. That is why the end-of-submission and upload barriers exist.

## How it is checked

Synchronization validation is enabled in every test run. By default it doesn't track memory accessed by shaders. Without `VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true`, a program with no barriers at all reported zero hazards, so the app sets it before creating the instance. The `validation_canary` test leaves out the force → integrate barrier and passes only if the checker reports a `SYNC-HAZARD`.
