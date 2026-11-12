"""Lab 1: fork-join lifecycle, nondeterminism, and oversubscription."""

from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil

from common import DATA_DIR, ensure_output_dirs, write_csv


THREAD_COUNTS = (1, 2, 4, 8, 16, 32, 64)


def identify_worker(rank: int, team_size: int, barrier: threading.Barrier) -> str:
    barrier.wait()
    time.sleep(random.random() * 0.004)
    role = "Master" if rank == 0 else "Worker"
    return f"[{role}] Logical Rank: {rank} of {team_size} | Native OS TID: {threading.get_native_id()}"


def capture_team(num_threads: int) -> list[str]:
    barrier = threading.Barrier(num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(identify_worker, rank, num_threads, barrier) for rank in range(num_threads)]
        return [future.result() for future in as_completed(futures)]


def time_empty_team(num_threads: int) -> float:
    barrier = threading.Barrier(num_threads)

    def rendezvous() -> None:
        barrier.wait()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(rendezvous) for _ in range(num_threads)]
        for future in futures:
            future.result()
    return time.perf_counter() - start


def cpu_kernel(work_items: int) -> float:
    value = 0.0
    for i in range(1, work_items + 1):
        value += math.sqrt(i)
    return value


def cpu_saturation(num_workers: int, work_items: int) -> tuple[float, float]:
    before = psutil.cpu_times_percent(interval=None)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(cpu_kernel, [work_items] * num_workers))
    elapsed = time.perf_counter() - start
    after = psutil.cpu_times_percent(interval=0.1)
    assert all(value > 0 for value in results)
    busy_percent = max(0.0, 100.0 - after.idle)
    _ = before
    return elapsed, busy_percent


def run(profile: str) -> None:
    ensure_output_dirs()
    runs = 10
    team_size = min(4, os.cpu_count() or 4)
    output_lines: list[str] = []
    for run_id in range(1, runs + 1):
        output_lines.append(f"Run {run_id}")
        output_lines.extend(capture_team(team_size))
        output_lines.append("")
    (DATA_DIR / "lab1_nondeterminism.txt").write_text("\n".join(output_lines), encoding="utf-8")

    trials = 7 if profile == "full" else 3
    rows: list[dict[str, object]] = []
    for p in THREAD_COUNTS:
        samples = [time_empty_team(p) for _ in range(trials)]
        for trial, elapsed in enumerate(samples, start=1):
            rows.append({"threads": p, "trial": trial, "seconds": f"{elapsed:.9f}"})
    write_csv(DATA_DIR / "lab1_oversubscription.csv", ["threads", "trial", "seconds"], rows)

    work_items = 10_000_000 if profile == "full" else 250_000
    saturation_rows: list[dict[str, object]] = []
    for p in (1, 2, 4, 8):
        elapsed, cpu_busy = cpu_saturation(p, work_items)
        saturation_rows.append(
            {"workers": p, "work_items_per_worker": work_items, "seconds": f"{elapsed:.6f}", "sampled_cpu_busy_percent": f"{cpu_busy:.1f}"}
        )
    write_csv(
        DATA_DIR / "lab1_cpu_saturation.csv",
        ["workers", "work_items_per_worker", "seconds", "sampled_cpu_busy_percent"],
        saturation_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()

