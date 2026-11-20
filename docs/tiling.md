# Tiling the force kernel through shared memory

> Draft with measured data. Rewrite the explanations in your own words.

## The idea

In `force_calc.comp`, each invocation reads all N particles from global memory, so a workgroup of 64 reads every particle 64 times. In `force_calc_tiled.comp`, the 64 invocations of a workgroup each load one particle into `shared vec4 tile[64]`. After a `barrier()`, all 64 read the whole tile from fast on-chip shared memory. A second `barrier()` stops anyone overwriting the tile while others still read it. Global reads per workgroup drop from 64·N to N.

Two details matter for correctness:

- **No early return.** Every invocation must reach every `barrier()`. An invocation past the end of the array still loads a padding value and joins the barriers, and only skips the final write. `glslc` accepts the broken version without warning.
- **The last tile is partial** when N is not a multiple of 64, so the inner loop runs `min(64, N − tileStart)` times. `force_tiled` tests N = 63, 65 and 1000 against the CPU reference.

## Results on the NVIDIA GeForce RTX 4090 (batch 32, median of 3 runs)

| N | naive force µs | tiled force µs | force speed-up | frame speed-up |
| --- | --- | --- | --- | --- |
| 256 | 11.5 | 7.0 | 1.64× | 1.39× |
| 2,048 | 86.7 | 40.2 | 2.15× | 2.02× |
| 16,384 | 715.7 | 313.9 | 2.28× | **2.25×** |
| 32,768 | 1,630.9 | 850.4 | 1.92× | 1.90× |

Frame time at N = 16,384: 725.8 µs naive, 322.9 µs tiled. Full table in `linux/headline.md`, chart in `linux/tiling.svg`.

## Where it helps and where it doesn't

On the RTX 4090, tiling helps at every N from 256 up, peaking around 2.2–2.3× from N = 4,096 to 16,384. At N = 32,768 the gain falls to 1.9×. These measurements don't show why; one possibility is that the naive kernel's reads start to benefit from a different caching pattern at that size.

## Secondary: Apple M1 through MoltenVK

The same kernel gave only 1.17× at N = 16,384 on the M1, and was slightly slower at N = 256. The M1 has unified memory and large caches, which already soften the cost of the naive kernel's repeated global reads, so there is less for tiling to save.
