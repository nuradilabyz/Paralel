"""Lab 5: bounded task-based merge sort with cutoff threshold sweep."""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DATA_DIR, ensure_output_dirs, write_csv


CUTOFFS = (1, 10, 100, 1_000, 10_000, 50_000, 100_000)


def merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(left.size + right.size, dtype=left.dtype)
    i = j = k = 0
    while i < left.size and j < right.size:
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1
    if i < left.size:
        result[k:] = left[i:]
    elif j < right.size:
        result[k:] = right[j:]
    return result


def leaf_ranges(length: int, cutoff: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    stack = [(0, length)]
    while stack:
        start, end = stack.pop()
        if end - start <= cutoff:
            ranges.append((start, end))
        else:
            midpoint = start + (end - start) // 2
            stack.append((midpoint, end))
            stack.append((start, midpoint))
    return ranges


def parallel_merge_sort(array: np.ndarray, cutoff: int, workers: int, max_live_tasks: int = 50_000) -> tuple[np.ndarray, int]:
    ranges = leaf_ranges(len(array), cutoff)
    if len(ranges) > max_live_tasks:
        raise RuntimeError(
            f"cutoff {cutoff} creates {len(ranges):,} tasks, above the {max_live_tasks:,} safety cap; "
            "rerun with a smaller array or increase max_live_tasks deliberately"
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        leaves = list(executor.map(lambda bounds: np.sort(array[bounds[0] : bounds[1]], kind="mergesort"), ranges))
    task_count = len(leaves)
    while len(leaves) > 1:
        next_level: list[np.ndarray] = []
        iterator = iter(leaves)
        for left in iterator:
            right = next(iterator, None)
            next_level.append(left if right is None else merge(left, right))
        leaves = next_level
    return leaves[0] if leaves else array.copy(), task_count


def run(profile: str, max_live_tasks: int) -> None:
    ensure_output_dirs()
    n = 5_000_000 if profile == "full" else 20_000
    trials = 3
    workers = min(8, os.cpu_count() or 1)
    rng = np.random.default_rng(20260924)
    source = rng.integers(0, 10_000_000, size=n, dtype=np.int64)
    rows = []
    for cutoff in CUTOFFS:
        for trial in range(1, trials + 1):
            start = time.perf_counter()
            status = "measured"
            try:
                sorted_array, tasks = parallel_merge_sort(source, cutoff, workers, max_live_tasks=max_live_tasks)
                elapsed = time.perf_counter() - start
                verified = bool(np.all(sorted_array[:-1] <= sorted_array[1:]))
            except RuntimeError as exc:
                elapsed = math.nan
                verified = False
                tasks = len(leaf_ranges(n, cutoff))
                status = f"resource_guard: {exc}"
            rows.append(
                {
                    "cutoff": cutoff,
                    "trial": trial,
                    "array_size": n,
                    "workers": workers,
                    "tasks": tasks,
                    "seconds": "" if math.isnan(elapsed) else f"{elapsed:.6f}",
                    "verified_sorted": verified,
                    "status": status,
                }
            )
    write_csv(
        DATA_DIR / "lab5_cutoff_sweep.csv",
        ["cutoff", "trial", "array_size", "workers", "tasks", "seconds", "verified_sorted", "status"],
        rows,
    )

    t1_start = time.perf_counter()
    baseline = np.sort(source, kind="mergesort")
    t1 = time.perf_counter() - t1_start
    assert np.all(baseline[:-1] <= baseline[1:])
    work_model = n * math.log2(max(n, 2))
    # S(N) = S(N/2) + N for a sequential merge at each level, whose geometric
    # series is bounded by 2N. Constants are retained here for a numeric model.
    span_model = 2 * n - 1
    model_parallelism = work_model / span_model
    write_csv(
        DATA_DIR / "lab5_work_span.csv",
        ["array_size", "measured_t1_seconds", "work_model", "span_model", "model_parallelism", "workers"],
        [
            {
                "array_size": n,
                "measured_t1_seconds": f"{t1:.6f}",
                "work_model": f"{work_model:.3f}",
                "span_model": f"{span_model:.3f}",
                "model_parallelism": f"{model_parallelism:.6f}",
                "workers": workers,
            }
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--max-live-tasks", type=int, default=50_000)
    args = parser.parse_args()
    run(args.profile, args.max_live_tasks)


if __name__ == "__main__":
    main()
