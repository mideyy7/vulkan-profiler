"""Frame time, GPU time and overhead per step against N, unbatched (Phase 5)."""

from plot_common import data_dir, finish, gpu_label, new_axes, plot_series, read_sweep

out = data_dir()
meta, rows = read_sweep(out, "sweep_naive_batch1")
n = [int(r["n"]) for r in rows]
us = lambda key: [float(r[key]) / 1e3 for r in rows]  # noqa: E731

fig, ax = new_axes("Fixed overhead dominates at small N",
                   f"Naive kernel, batch = 1, median of 3 runs. {gpu_label(meta)}")
plot_series(ax, n, us("frame_mean_ns"), 0, "frame", label_dy=8)
plot_series(ax, n, us("gpu_mean_ns"), 1, "GPU span", label_dy=-8)
plot_series(ax, n, us("overhead_mean_ns"), 2, "overhead (frame - GPU)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("particles (N)")
ax.set_ylabel("time per step (µs, log scale)")
ax.set_xlim(right=max(n) * 4)
finish(fig, ax, out / "sweep.svg")
