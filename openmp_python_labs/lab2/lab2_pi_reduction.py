"""Lab 2: race, critical section, and private-accumulator reduction variants."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

from common import DATA_DIR, ensure_output_dirs, write_csv


THREAD_COUNTS = (1, 2, 4, 8, 16)


def _range_for_worker(worker: int, workers: int, n: int) -> tuple[int, int]:
    start = worker * n // workers
    end = (worker + 1) * n // workers
    return start, end


def _race_worker(shared_sum, start: int, end: int, step: float) -> None:
    for i in range(start, end):
        x = (i + 0.5) * step
        shared_sum.value += 4.0 / (1.0 + x * x)


def _critical_worker(shared_sum, lock, start: int, end: int, step: float) -> None:
    for i in range(start, end):
        x = (i + 0.5) * step
        term = 4.0 / (1.0 + x * x)
        with lock:
            shared_sum.value += term


def _partial_sum(args: tuple[int, int, float]) -> float:
    start, end, step = args
    subtotal = 0.0
    for i in range(start, end):
        x = (i + 0.5) * step
        subtotal += 4.0 / (1.0 + x * x)
    return subtotal


def serial_pi(n: int) -> float:
    return _partial_sum((0, n, 1.0 / n)) / n


def race_pi(n: int, workers: int) -> float:
    ctx = mp.get_context("spawn")
    shared_sum = ctx.RawValue("d", 0.0)
    step = 1.0 / n
    processes = []
    for worker in range(workers):
        start, end = _range_for_worker(worker, workers, n)
        process = ctx.Process(target=_race_worker, args=(shared_sum, start, end, step))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"race worker exited with {process.exitcode}")
    return shared_sum.value * step


def critical_pi(n: int, workers: int) -> float:
    ctx = mp.get_context("spawn")
    shared_sum = ctx.RawValue("d", 0.0)
    lock = ctx.Lock()
    step = 1.0 / n
    processes = []
    for worker in range(workers):
        start, end = _range_for_worker(worker, workers, n)
        process = ctx.Process(target=_critical_worker, args=(shared_sum, lock, start, end, step))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"critical worker exited with {process.exitcode}")
    return shared_sum.value * step


def reduction_pi(n: int, workers: int) -> float:
    step = 1.0 / n
    ranges = [(*_range_for_worker(worker, workers, n), step) for worker in range(workers)]
    if workers == 1:
        return _partial_sum(ranges[0]) * step
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        partials = list(executor.map(_partial_sum, ranges))
    return math.fsum(partials) * step


def timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - start


def run(profile: str) -> None:
    ensure_output_dirs()
    race_n = 100_000_000 if profile == "full" else 250_000
    critical_n = 1_000_000 if profile == "full" else 100_000
    reduction_n = 100_000_000 if profile == "full" else 2_000_000
    trials = 5 if profile == "full" else 3

    race_rows = []
    for p in (1, 2, 4, 8):
        for trial in range(1, 4):
            value, elapsed = timed(race_pi, race_n, p)
            race_rows.append(
                {
                    "threads": p,
                    "trial": trial,
                    "steps": race_n,
                    "pi": f"{value:.15f}",
                    "absolute_error": f"{abs(value - math.pi):.15e}",
                    "seconds": f"{elapsed:.6f}",
                }
            )
    write_csv(DATA_DIR / "lab2_race.csv", ["threads", "trial", "steps", "pi", "absolute_error", "seconds"], race_rows)

    _, serial_time = timed(serial_pi, critical_n)
    critical_rows = []
    for p in (1, 2, 4, 8):
        value, elapsed = timed(critical_pi, critical_n, p)
        overhead = ((elapsed - serial_time) / serial_time) * 100.0
        critical_rows.append(
            {
                "threads": p,
                "steps": critical_n,
                "pi": f"{value:.15f}",
                "seconds": f"{elapsed:.6f}",
                "serial_seconds": f"{serial_time:.6f}",
                "lock_overhead_percent": f"{overhead:.2f}",
            }
        )
    write_csv(
        DATA_DIR / "lab2_critical.csv",
        ["threads", "steps", "pi", "seconds", "serial_seconds", "lock_overhead_percent"],
        critical_rows,
    )

    reduction_rows = []
    averages: dict[int, float] = {}
    raw: dict[int, list[float]] = {}
    for p in THREAD_COUNTS:
        raw[p] = []
        for trial in range(1, trials + 1):
            value, elapsed = timed(reduction_pi, reduction_n, p)
            raw[p].append(elapsed)
            reduction_rows.append(
                {
                    "threads": p,
                    "trial": trial,
                    "steps": reduction_n,
                    "pi": f"{value:.15f}",
                    "absolute_error": f"{abs(value - math.pi):.15e}",
                    "seconds": f"{elapsed:.6f}",
                }
            )
        averages[p] = sum(raw[p]) / len(raw[p])

    baseline = averages[1]
    for row in reduction_rows:
        p = int(row["threads"])
        speedup = baseline / averages[p]
        row["mean_seconds"] = f"{averages[p]:.6f}"
        row["speedup"] = f"{speedup:.6f}"
        row["efficiency"] = f"{speedup / p:.6f}"
    write_csv(
        DATA_DIR / "lab2_reduction.csv",
        ["threads", "trial", "steps", "pi", "absolute_error", "seconds", "mean_seconds", "speedup", "efficiency"],
        reduction_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
