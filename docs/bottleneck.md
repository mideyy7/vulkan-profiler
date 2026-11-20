# Finding and removing the per-step overhead

> Draft with measured data. Rewrite the explanations in your own words.

All numbers are from the Apple M1 through MoltenVK (`docs/mac/`). Linux GPU numbers will replace them as the primary result.

## What the breakdown showed

With one step per submit (`--batch 1`), every step pays for a `vkQueueSubmit`, the GPU running the work, and a `vkWaitForFences` on the CPU. The report measures `frame` from just before the submit to just after the wait, and `gpu span` from the first to the last timestamp. Their difference, `overhead`, is time when the GPU has finished or not started, but the CPU is still submitting or waiting.

From `headline.md` (naive kernel, median of 3 runs):

| N | frame µs, batch 1 | overhead µs, batch 1 |
| --- | --- | --- |
| 256 | 567.8 | 304.6 |
| 2,048 | 1,218.6 | 379.3 |
| 32,768 | 51,125.9 | 809.9 |

The overhead stays in the hundreds of microseconds while frame time grows about 90×. At small N it is about half the frame, so the app is **overhead-bound**. At large N the force kernel dominates, so the app is **compute-bound** (`sweep.svg`).

## The fix: batching

`--batch K` records K steps into one command buffer and submits it once. The CPU round trip between steps is replaced by a GPU pipeline barrier, which orders step k's writes before step k+1's reads without the CPU taking part. See `synchronisation.md`.

At N = 2,048 (5 runs each, `report_n2048_batch{1,32}_run*.csv`):

| | batch 1, median | batch 32, median |
| --- | --- | --- |
| frame per step | 753.7 µs | 369.8 µs |
| overhead per step | 304.8 µs | 26.4 µs |
| GPU span per step | 448.9 µs | 343.6 µs |
| force kernel per step | 334.1 µs | 278.8 µs |

Most of the 2.04× comes from the overhead, which falls by 278 µs per step. The GPU span also falls, because the readback copy and its barrier now happen once per batch instead of once per step. The force kernel itself is also about 55 µs faster batched. One likely cause is that the M1 GPU slows its clock when it is idle between submits, but this hasn't been measured directly.

Unbatched timings vary much more between runs: batch 1 frame times ranged from 735.1 to 1,016.2 µs over 5 runs, against 368.3 to 374.9 µs for batch 32. That is why every configuration is run several times and the median is reported.

## Why results stay bit-identical

Batching changes **when** the CPU waits, not **what** the GPU computes. Each step runs the same two dispatches on the same data in the same order, and floating-point results depend only on the operations and their order. `batch_equivalence` checks this: after 800 steps, K = 16, 25 and 32 all give byte-identical particles to K = 1.
