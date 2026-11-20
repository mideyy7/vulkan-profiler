# Running on the rented Linux GPU

Instructions for whoever is on the Linux machine, a person or an AI assistant. Read this whole file before running anything.

## Context

- This repository is a Vulkan GPU compute profiler: an N-body simulation with per-stage GPU timing, batching, a shared-memory tiled kernel, and a Vulkan layer. See `README.md`.
- Everything has already been built and tested on an Apple M1 through MoltenVK. Those results are in `docs/mac/` and only count as **secondary** data.
- **This machine exists to produce the primary results on a real NVIDIA GPU under Linux.** The CV numbers, batching speed-up at N = 2,048 and tiling speed-up at N = 16,384, will come from what is produced here.
- The machine is a **Vast.ai Docker container**: Ubuntu 24.04, running as `root`, with no `sudo`, and an RTX 4090. It is **billed by the hour**, so work efficiently and don't explore beyond what's needed.

## Rules

1. **Never invent, edit, round or estimate a number.** Every number must come from a CSV produced by the commands below.
2. **The GPU must be the NVIDIA one.** If Vulkan only sees `llvmpipe` or `lavapipe` (CPU renderers), stop and fix that first. Numbers from a CPU renderer are worthless here.
3. **Don't change source code to make a test pass.** If a test fails, record the output and report it. Fixes are made on the Mac and pushed.
4. **Benchmarks run with validation off** (the scripts already do this), because validation adds CPU time to what is being measured. Correctness is checked separately by `ctest` with validation on.
5. **Don't push from this machine.** It has no GitHub credentials. Results go back to the Mac as one file.

## Steps

### 1. Run everything

From the repository root:

```bash
bash scripts/cloud_gpu_run.sh 2>&1 | tee run.log
```

This takes a few minutes. It:
1. checks `nvidia-smi` works;
2. installs `cmake g++ python3 git libvulkan-dev vulkan-tools vulkan-validationlayers glslc`;
3. adds the NVIDIA Vulkan driver manifest if the container is missing it;
4. **stops if Vulkan can't see the NVIDIA GPU**;
5. builds, and runs all 14 tests with validation on;
6. runs every benchmark into `docs/linux/` and packs `linux-results.tgz`.

### 2. Check the results are real

```bash
cat docs/linux/device.txt          # must show the RTX 4090 and the NVIDIA driver
cat docs/linux/ctest.txt | tail -5 # expect: 100% tests passed, 0 tests failed out of 14
head -3 docs/linux/sweep_naive_batch1_run1.csv   # '# gpu:' line must name the NVIDIA GPU
cat docs/linux/machine.txt         # load before and after the run
```

If `device.txt` names anything other than the NVIDIA GPU, the results are invalid.

### 3. Charts and the headline table (optional, saves a step on the Mac)

```bash
apt-get install -y -qq python3-matplotlib
for s in sweep batching tiling; do python3 scripts/plot_$s.py docs/linux; done
python3 scripts/summarize.py docs/linux > docs/linux/headline.md
cat docs/linux/headline.md
tar czf linux-results.tgz docs/linux run.log
```

`headline.md` holds the numbers. Report the batch-32 against batch-1 speed-up at N = 2048 and the tiled against naive frame-time speed-up at N = 16384, exactly as printed.

### 4. Hand the results back

On the **Mac**, using the port and address Vast.ai shows for the instance:

```bash
scp -P <port> root@<address>:~/vulkan-profiler/linux-results.tgz .
```

Then **destroy** the instance on Vast.ai. Stopping it isn't enough, because a stopped instance still charges for storage.

## If something goes wrong

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `nvidia-smi: command not found` | No NVIDIA driver in the container | Recreate the instance on a GPU machine with a CUDA image |
| Step 3 stops: Vulkan sees only `llvmpipe`, or no device | Container started without graphics capability | Recreate the instance with Docker option `-e NVIDIA_DRIVER_CAPABILITIES=all` |
| Step 3 stops, and `ldconfig -p \| grep libGLX_nvidia` finds nothing | The driver's Vulkan library isn't mounted | Same fix as above; inside a container, `apt install libnvidia-gl-*` won't work |
| `vkCreateInstance failed with VkResult -6` | Validation layer not found | `apt-get install -y vulkan-validationlayers`, then rerun |
| `glslc: not found` during the build | Package missing | `apt-get install -y glslc`, then rerun |
| Only `layer_check` fails | Older Vulkan loader without `VK_ADD_LAYER_PATH` / `VK_LOADER_LAYERS_ENABLE` | Benchmarks are still valid. Record the output and report it |
| `validation_canary` fails | This validation layer version reports hazards differently | Benchmarks are still valid. Record the output and report it |
| A force, momentum or batch-equivalence test fails | A real correctness problem on this GPU | **Don't trust the numbers.** Save `run.log` and report it before anything else |
| Disk full | 10 GB disk | `rm -rf build` and rerun; results are small |

Rerunning `bash scripts/cloud_gpu_run.sh` is safe: it rebuilds and overwrites `docs/linux/`.
