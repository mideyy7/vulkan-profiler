#!/usr/bin/env bash
# Runs every benchmark used in docs/ and writes the CSVs to OUTPUT_DIR.
# Usage: scripts/run_benchmarks.sh OUTPUT_DIR
# Validation is off for all of these, because it adds CPU time to the calls
# being measured. Correctness is covered separately by ctest.
set -euo pipefail

out=${1:?usage: scripts/run_benchmarks.sh OUTPUT_DIR}
bin=${GPU_PROFILER:-build/gpu_profiler}
mkdir -p "$out"

echo "Machine load before the run (C14):" > "$out/machine.txt"
uptime >> "$out/machine.txt"
"$bin" --info > "$out/device.txt"

# Phases 5 and 6: sweeps over N. Each configuration runs 3 times, interleaved,
# and the plots and summary use the median for each N.
for run in 1 2 3; do
    for batch in 1 32; do
        "$bin" --sweep --batch "$batch" --no-validation \
            --csv "$out/sweep_naive_batch${batch}_run$run.csv"
    done
    "$bin" --sweep --batch 32 --kernel tiled --no-validation \
        --csv "$out/sweep_tiled_batch32_run$run.csv"
done

# Phase 5: effect of batch size at N = 2048. Unbatched timings vary a lot
# between runs, so every batch size is run 5 times, interleaved, and the
# summary uses the median of the 5 run means.
for run in 1 2 3 4 5; do
    for batch in 1 2 4 8 16 32 64; do
        "$bin" --report --n 2048 --batch "$batch" --no-validation \
            --csv "$out/report_n2048_batch${batch}_run$run.csv" > /dev/null
    done
done

echo "uptime after:" >> "$out/machine.txt"
uptime >> "$out/machine.txt"
echo "Wrote results to $out"
