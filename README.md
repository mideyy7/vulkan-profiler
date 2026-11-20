# Vulkan GPU Compute Profiler

A headless C++17 Vulkan application that runs an N-body particle simulation on the GPU and breaks GPU time down by pipeline stage using timestamp queries. It uses that breakdown to find and remove a fixed per-step submit and wait cost, tiles the force kernel through shared memory, and includes a Vulkan layer that profiles any Vulkan app without recompiling it.

## What it does

- **Simulation:** all-pairs softened gravity with semi-implicit Euler integration, written as GLSL compute shaders (`shaders/`).
- **Per-stage timing:** `vkCmdWriteTimestamp` at every stage boundary. The report gives mean, min, p50, p99, max and % of frame for each stage, plus CPU time in submit and fence wait.
- **Correctness:** GPU forces are checked against a double-precision CPU reference, including particle counts not divisible by the workgroup size. Momentum conservation is tested too, and batched runs are checked to be bit-identical to unbatched runs.
- **Validation:** every test runs with `VK_LAYER_KHRONOS_validation` and synchronization validation on, and must finish with zero messages. A canary test deliberately leaves out a barrier to prove the checker can detect it.
- **Batching:** `--batch K` records K steps in one command buffer, with one submit and one fence wait.
- **Tiling:** `--kernel tiled` stages particles through workgroup shared memory.
- **Layer:** `VK_LAYER_stage_profiler` counts dispatches, barriers and copies and times submits and waits in any Vulkan app (`layer/`).

## Build

Requirements are CMake 3.20 or newer, a C++17 compiler, the Vulkan headers and loader, `glslc`, and the Khronos validation layer. The LunarG Vulkan SDK provides all of the Vulkan parts.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

| Test | Pass condition |
| --- | --- |
| `vector_add` | Exact match for N = 1, 63, 64, 65, 1000 |
| `force_naive`, `force_tiled` | Max relative error < 1e-5 against the CPU, 5 seeds × N = 1, 63, 64, 65, 1000 |
| `momentum` | Relative momentum drift < 1e-5 over 100 steps, N = 8, 64, 256 |
| `timestamps` | Monotonic, every stage under 1 s, GPU span ≤ frame |
| `stats` | Mean and percentiles match hand-computed values |
| `batch_equivalence` | Byte-identical particles after 800 steps for K = 16, 25, 32 against K = 1 |
| `validation_modes` | Both kernels at K = 1 and 32 with zero validation messages |
| `validation_live`, `validation_leak`, `validation_canary` | A deliberate mistake **is** reported |
| `layer_check` | Layer counts equal the app's, wait times agree within 10%, zero messages when stacked with validation |

### Checking validation yourself

Every run validates unless `--no-validation` is given. Any validation message is printed to stderr, and the run then exits non-zero:

```bash
./build/gpu_profiler --report --n 2048 --batch 32
```

## Run

```bash
# Per-stage table for one configuration (add --csv out.csv to save it)
./build/gpu_profiler --report --n 2048 --batch 1 --no-validation

# Sweep N = 256 ... 32768
./build/gpu_profiler --sweep --batch 32 --kernel tiled --no-validation --csv sweep.csv

# Every benchmark in docs/, then the charts and the headline table
scripts/run_benchmarks.sh docs/<machine>
for s in sweep batching tiling; do python3 scripts/plot_$s.py docs/<machine>; done
python3 scripts/summarize.py docs/<machine> > docs/<machine>/headline.md
```

Benchmarks run with `--no-validation`, because the validation layer adds CPU time to the calls being measured.

## Profile any Vulkan app with the layer

```bash
VK_ADD_LAYER_PATH=$PWD/build/layer VK_LOADER_LAYERS_ENABLE=VK_LAYER_stage_profiler vkcube --c 300
```

The report is printed to stderr when the app destroys its device. See [docs/layer.md](docs/layer.md).

## Results

**Primary: NVIDIA GeForce RTX 4090, driver 580.178.04, Ubuntu 24.04** ([docs/linux/headline.md](docs/linux/headline.md)). All 14 tests passed on this machine.

| Measurement | Before | After | Speed-up |
| --- | --- | --- | --- |
| Batching, N = 2,048, batch 1 → 32 (median of 5 runs) | 142.4 µs/step | 100.1 µs/step | **1.42×** |
| Tiling, N = 16,384, naive → tiled, batch 32 (median of 3 runs) | 725.8 µs/step | 322.9 µs/step | **2.25×** |
| Fixed overhead per step at N = 2,048 (frame − GPU span) | 40.1 µs | 1.6 µs | |

Reproduce on a Linux NVIDIA machine with `bash scripts/cloud_gpu_run.sh` (see [linux.md](linux.md)).

**Secondary: Apple M1 through MoltenVK** ([docs/mac/headline.md](docs/mac/headline.md)): batching 2.04× at N = 2,048, tiling 1.17× at N = 16,384.

## Documentation

- [docs/bottleneck.md](docs/bottleneck.md): finding and removing the fixed per-step cost
- [docs/synchronisation.md](docs/synchronisation.md): every barrier and why it is there
- [docs/tiling.md](docs/tiling.md): the shared-memory force kernel
- [docs/layer.md](docs/layer.md): how the Vulkan layer works
