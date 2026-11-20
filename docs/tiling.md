# Tiling the force kernel through shared memory

> Draft with measured data. Rewrite the explanations in your own words.

## The idea

In `force_calc.comp`, each invocation reads all N particles from global memory, so a workgroup of 64 reads every particle 64 times. In `force_calc_tiled.comp`, the 64 invocations of a workgroup each load one particle into `shared vec4 tile[64]`. After a `barrier()`, all 64 read the whole tile from fast on-chip shared memory. A second `barrier()` stops anyone overwriting the tile while others still read it. Global reads per workgroup drop from 64·N to N.

Two details matter for correctness:

- **No early return.** Every invocation must reach every `barrier()`. An invocation past the end of the array still loads a padding value and joins the barriers, and only skips the final write. `glslc` accepts the broken version without warning.
- **The last tile is partial** when N is not a multiple of 64, so the inner loop runs `min(64, N − tileStart)` times. `force_tiled` tests N = 63, 65 and 1000 against the CPU reference.

## Results on Apple M1 (batch 32, median of 3 runs)

| N | naive force µs | tiled force µs | speed-up |
| --- | --- | --- | --- |
| 256 | 47.0 | 52.4 | 0.90× |
| 2,048 | 279.3 | 204.0 | 1.37× |
| 16,384 | 11,834.1 | 10,573.1 | 1.12× |
| 32,768 | 37,248.2 | 31,913.0 | 1.17× |

Frame time at N = 16,384: 12,919.5 µs naive, 11,024.4 µs tiled, so 1.17×. Full table in `mac/headline.md`, chart in `mac/tiling.svg`.

## Where it helps and where it doesn't

At small N there are only a few workgroups, the particle array fits in cache anyway, and the barriers add work, so tiling is no faster or slightly slower. The M1 also has unified memory and large caches, which already soften the cost of repeated global reads. The gain is expected to be larger on a discrete GPU with separate video memory, but that is untested until the Linux run.
