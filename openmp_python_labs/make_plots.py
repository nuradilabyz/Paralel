"""Create all plots from raw CSV files without modifying benchmark results."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, PLOTS_DIR, ensure_output_dirs


NAVY = "#17324D"
BLUE = "#2E75B6"
TEAL = "#2A9D8F"
ORANGE = "#E07A2D"
GRAY = "#7A8591"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_group(rows: list[dict[str, str]], keys: tuple[str, ...], value: str) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value):
            grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def style_axes(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def save(figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_lab1() -> None:
    grouped = mean_group(read_csv("lab1_oversubscription.csv"), ("threads",), "seconds")
    threads = sorted(int(key[0]) for key in grouped)
    values = [grouped[(str(p),)] * 1000 for p in threads]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.plot(threads, values, marker="o", color=BLUE, linewidth=2.2)
    axis.set_xscale("log", base=2)
    axis.set_xticks(threads, labels=[str(value) for value in threads])
    style_axes(axis, "Fork join overhead by team size", "Thread team size P", "Mean elapsed time ms")
    save(figure, "lab1_oversubscription.png")


def plot_lab2() -> None:
    rows = read_csv("lab2_reduction.csv")
    by_thread: dict[int, float] = {}
    for row in rows:
        by_thread[int(row["threads"])] = float(row["speedup"])
    threads = sorted(by_thread)
    speedups = [by_thread[p] for p in threads]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.plot(threads, speedups, marker="o", linewidth=2.2, color=BLUE, label="Measured")
    axis.plot(threads, threads, linestyle="--", color=GRAY, linewidth=1.8, label="Linear ideal")
    axis.set_xticks(threads)
    axis.legend(frameon=False)
    style_axes(axis, "Private accumulator reduction scaling", "Worker processes P", "Speedup T1 divided by TP")
    save(figure, "lab2_speedup.png")


def plot_lab3() -> None:
    grouped = mean_group(read_csv("lab3_scheduling.csv"), ("policy", "threads", "chunk_size"), "seconds")
    policies = ("static", "dynamic")
    thread_counts = (2, 4, 8, 16)
    chunks = (1, 16, 64, 256)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, policy in zip(axes, policies):
        matrix = np.array([[grouped[(policy, str(p), str(c))] for c in chunks] for p in thread_counts])
        image = axis.imshow(matrix, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(chunks)), labels=[str(c) for c in chunks])
        axis.set_yticks(range(len(thread_counts)), labels=[str(p) for p in thread_counts])
        axis.set_xlabel("Chunk size C")
        axis.set_title(f"{policy.title()} scheduler", fontsize=12, fontweight="bold", color=NAVY)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                threshold = (matrix.max() + matrix.min()) / 2
                axis.text(col, row, f"{matrix[row, col]:.2f}s", ha="center", va="center", color="white" if matrix[row, col] > threshold else NAVY, fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    axes[0].set_ylabel("Worker processes P")
    figure.suptitle("Mandelbrot scheduling matrix", x=0.06, ha="left", fontsize=14, fontweight="bold", color=NAVY)
    save(figure, "lab3_scheduler_heatmap.png")


def plot_lab4() -> None:
    grouped = mean_group(read_csv("lab4_false_sharing.csv"), ("variant", "threads"), "seconds")
    threads = (1, 2, 4, 8, 16)
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    for variant, color, label in (
        ("unpadded", ORANGE, "Unpadded"),
        ("padded", BLUE, "Cache-line padded"),
        ("local", TEAL, "Thread-local accumulator"),
    ):
        axis.plot(threads, [grouped[(variant, str(p))] for p in threads], marker="o", linewidth=2.1, label=label, color=color)
    axis.set_xticks(threads)
    axis.legend(frameon=False)
    style_axes(axis, "False sharing and cache isolation", "Worker processes P", "Mean elapsed time seconds")
    save(figure, "lab4_false_sharing.png")


def plot_lab5() -> None:
    rows = [row for row in read_csv("lab5_cutoff_sweep.csv") if row["seconds"]]
    grouped = mean_group(rows, ("cutoff",), "seconds")
    cutoffs = sorted(int(key[0]) for key in grouped)
    values = [grouped[(str(cutoff),)] for cutoff in cutoffs]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.plot(cutoffs, values, marker="o", color=BLUE, linewidth=2.2)
    axis.set_xscale("log")
    axis.set_xticks(cutoffs, labels=[f"{value:,}" for value in cutoffs], rotation=25)
    style_axes(axis, "Task granularity cutoff sweep", "Sequential cutoff K log scale", "Mean elapsed time seconds")
    save(figure, "lab5_cutoff.png")


def main() -> None:
    ensure_output_dirs()
    plot_lab1()
    plot_lab2()
    plot_lab3()
    plot_lab4()
    plot_lab5()


if __name__ == "__main__":
    main()

