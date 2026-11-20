"""Naive against tiled force kernel (Phase 6)."""

from plot_common import data_dir, finish, gpu_label, new_axes, plot_series, read_sweep

out = data_dir()
meta, naive = read_sweep(out, "sweep_naive_batch32")
_, tiled = read_sweep(out, "sweep_tiled_batch32")
n = [int(r["n"]) for r in naive]

fig, ax = new_axes("Shared-memory tiling against the naive force kernel",
                   f"Batch = 32, mean force-kernel time per step, median of 3 runs. {gpu_label(meta)}")
plot_series(ax, n, [float(r["force_mean_ns"]) / 1e3 for r in naive], 0, "naive", label_dy=6)
plot_series(ax, n, [float(r["force_mean_ns"]) / 1e3 for r in tiled], 1, "tiled", label_dy=-6)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("particles (N)")
ax.set_ylabel("force kernel time per step (µs, log scale)")
ax.set_xlim(right=max(n) * 3)
finish(fig, ax, out / "tiling.svg")
