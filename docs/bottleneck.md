# Finding and removing the per-step overhead

> Draft with measured data. Rewrite the explanations in your own words.

Primary numbers are from an **NVIDIA GeForce RTX 4090** (driver 580.178.04, Ubuntu 24.04, `docs/linux/`). Apple M1 numbers (`docs/mac/`) are at the end as a secondary comparison.

## What the breakdown showed

With one step per submit (`--batch 1`), every step pays for a `vkQueueSubmit`, the GPU running the work, and a `vkWaitForFences` on the CPU. The report measures `frame` from just before the submit to just after the wait, and `gpu span` from the first to the last timestamp. Their difference, `overhead`, is time when our GPU work is not running but the CPU is still submitting or waiting.

From `docs/linux/headline.md` (naive kernel, median of 3 runs):

| N | frame µs, batch 1 | overhead µs, batch 1 | share of frame |
| --- | --- | --- | --- |
| 256 | 46.5 | 28.2 | 61% |
| 2,048 | 134.4 | 38.6 | 29% |
| 32,768 | 1,882.9 | 29.9 | 2% |

The overhead stays around 30–60 µs per step at every N, while frame time grows about 40×. At small N it is most of the frame, so the app is **overhead-bound**. At large N the force kernel dominates, so it is **compute-bound** (`docs/linux/sweep.svg`).

## The fix: batching

`--batch K` records K steps into one command buffer and submits it once. The CPU round trip between steps is replaced by a GPU pipeline barrier, which orders step k's writes before step k+1's reads without the CPU taking part. See `synchronisation.md`.

At N = 2,048 (5 runs each, `docs/linux/report_n2048_batch{1,32}_run*.csv`, medians):

| per step | batch 1 | batch 32 |
| --- | --- | --- |
| frame | 142.4 µs | 100.1 µs |
| overhead (frame − GPU span) | 40.1 µs | 1.6 µs |
| GPU span | 102.3 µs | 98.6 µs |
| force kernel | 95.1 µs | 94.4 µs |
| readback copy | 3.3 µs | 0.1 µs |
| CPU time in `vkQueueSubmit` | 3.6 µs | 0.3 µs |

The force kernel takes the same time either way, so the **1.42×** comes from the overhead, which falls from 40.1 to 1.6 µs per step, and from the readback copy now happening once per batch instead of once per step. Most of the gain arrives by K = 8 (1.36×), and K = 64 adds little over K = 32 (1.43× against 1.42×; `docs/linux/batch_size.svg`).

## Why results stay bit-identical

Batching changes **when** the CPU waits, not **what** the GPU computes. Each step runs the same two dispatches on the same data in the same order, and floating-point results depend only on the operations and their order. `batch_equivalence` checks this: after 800 steps, K = 16, 25 and 32 all give byte-identical particles to K = 1. It passed on both the RTX 4090 and the M1.

## Secondary: Apple M1 through MoltenVK

At N = 2,048 the M1 gives 2.04× (753.7 → 369.8 µs per step). Its fixed overhead is far larger (304.8 µs per step unbatched), which is consistent with MoltenVK translating every submission to Metal. Unlike on the RTX 4090, the M1's force kernel also got faster when batched (334 → 279 µs), and its unbatched timings varied by almost 2× between runs. A likely cause is the GPU lowering its clock when it idles between submits, but this hasn't been measured directly.
