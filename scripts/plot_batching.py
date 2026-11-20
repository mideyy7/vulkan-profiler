"""Frame time per step with and without batching (Phase 5)."""

from plot_common import (INK, SERIES, SURFACE, data_dir, finish, frame_mean_us, gpu_label,
                         new_axes, plot_series, read_sweep)

out = data_dir()
meta, unbatched = read_sweep(out, "sweep_naive_batch1")
_, batched = read_sweep(out, "sweep_naive_batch32")
n = [int(r["n"]) for r in unbatched]

fig, ax = new_axes("Batching removes the per-step overhead",
                   f"Naive kernel, mean frame time per step, median of 3 runs. {gpu_label(meta)}")
plot_series(ax, n, [float(r["frame_mean_ns"]) / 1e3 for r in unbatched], 0, "batch = 1",
            label_dy=-8)
plot_series(ax, n, [float(r["frame_mean_ns"]) / 1e3 for r in batched], 1, "batch = 32",
            label_dy=8)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("particles (N)")
ax.set_ylabel("frame time per step (µs, log scale)")
ax.set_xlim(right=max(n) * 3)
finish(fig, ax, out / "batching.svg")

# Frame time per step against batch size at N = 2048: median of 5 runs,
# with a bar from the fastest to the slowest run.
sizes = [1, 2, 4, 8, 16, 32, 64]
median, low, high = [], [], []
for k in sizes:
    means = sorted(frame_mean_us(out / f"report_n2048_batch{k}_run{r}.csv") for r in range(1, 6))
    median.append(means[2])
    low.append(means[2] - means[0])
    high.append(means[-1] - means[2])
fig, ax = new_axes("Frame time per step against batch size",
                   f"N = 2048, naive kernel, median of 5 runs, bars span fastest to slowest. "
                   f"{gpu_label(meta)}")
ax.errorbar(sizes, median, yerr=[low, high], color=SERIES[0], linewidth=2, marker="o",
            markersize=6, capsize=4, elinewidth=1.2)
for k, m in zip(sizes, median):
    ax.annotate(f"{m:.0f}", (k, m), xytext=(10, 4), textcoords="offset points",
                fontsize=8.5, color=INK)
ax.set_xscale("log", base=2)
ax.set_xticks(sizes, [str(k) for k in sizes])
ax.set_ylim(bottom=0)
ax.set_xlabel("steps per submit (K)")
ax.set_ylabel("frame time per step (µs)")
fig.tight_layout()
fig.savefig(out / "batch_size.svg", format="svg", facecolor=SURFACE)
print(f"Wrote {out / 'batch_size.svg'}")
