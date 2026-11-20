"""Shared loading and styling for the plot_*.py scripts."""

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Write files only; never open a window.
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order), surface and ink colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def data_dir() -> Path:
    if len(sys.argv) != 2:
        sys.exit(f"usage: python3 {sys.argv[0]} RESULTS_DIR")
    return Path(sys.argv[1])


def read_csv(path: Path):
    """Returns (metadata dict from '# key: value' lines, list of row dicts)."""
    meta, lines = {}, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            else:
                lines.append(line)
    return meta, list(csv.DictReader(lines))


def new_axes(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def plot_series(ax, x, y, index, label, label_dy=0):
    """One line with markers and a direct label at its right end.
    label_dy moves the label up or down (in points) when two lines end together."""
    ax.plot(x, y, color=SERIES[index], linewidth=2, marker=MARKERS[index], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.annotate(label, (x[-1], y[-1]), xytext=(6, label_dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def finish(fig, ax, path: Path):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=SURFACE)
    print(f"Wrote {path}")


def gpu_label(meta) -> str:
    return f"{meta.get('gpu', '?')}, {meta.get('driver', '?')}"


def frame_mean_us(path: Path) -> float:
    """Mean frame time per step from a --report CSV, in microseconds."""
    _, rows = read_csv(path)
    return next(float(r["mean_ns"]) for r in rows if r["stage"] == "frame") / 1e3


SWEEP_RUNS = 3


def read_sweep(out: Path, name: str):
    """Median over the sweep's runs of every numeric column, per N.
    name is e.g. "sweep_naive_batch1"; files are name_run1.csv ... name_run3.csv."""
    runs = [read_csv(out / f"{name}_run{r}.csv") for r in range(1, SWEEP_RUNS + 1)]
    meta = runs[0][0]
    rows = []
    for i, first in enumerate(runs[0][1]):
        row = dict(first)
        for key in first:
            if key.endswith("_ns"):  # Timings; n, batch and kernel are the same in every run.
                row[key] = str(statistics.median(float(run[1][i][key]) for run in runs))
        rows.append(row)
    return meta, rows
