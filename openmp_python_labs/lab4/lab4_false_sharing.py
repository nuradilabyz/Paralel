"""Lab 4: false sharing, cache-line padding, and local accumulation."""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import time

from common import DATA_DIR, ensure_output_dirs, hardware_info, write_csv


THREAD_COUNTS = (1, 2, 4, 8, 16)


def _shared_increment(counters, index: int, iterations: int) -> None:
    for _ in range(iterations):
        counters[index] += 1


def _local_increment(counters, index: int, iterations: int) -> None:
    value = 0
    for _ in range(iterations):
        value += 1
    counters[index] = value


def benchmark_variant(variant: str, workers: int, iterations: int, stride: int) -> tuple[float, bool]:
    ctx = mp.get_context("spawn")
    effective_stride = 1 if variant == "unpadded" else stride
    counters = ctx.RawArray(ctypes.c_longlong, workers * effective_stride)
    target = _local_increment if variant == "local" else _shared_increment
    processes = []
    start = time.perf_counter()
    for worker in range(workers):
        index = worker * effective_stride
        process = ctx.Process(target=target, args=(counters, index, iterations))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"counter worker exited with {process.exitcode}")
    elapsed = time.perf_counter() - start
    correct = all(counters[worker * effective_stride] == iterations for worker in range(workers))
    return elapsed, correct


def run(profile: str) -> None:
    ensure_output_dirs()
    iterations = 100_000_000 if profile == "full" else 200_000
    trials = 5 if profile == "full" else 3
    cache_line = int(hardware_info().get("cache_line_bytes", 64))
    stride = max(8, cache_line // ctypes.sizeof(ctypes.c_longlong))
    rows = []
    for variant in ("unpadded", "padded", "local"):
        for p in THREAD_COUNTS:
            for trial in range(1, trials + 1):
                elapsed, correct = benchmark_variant(variant, p, iterations, stride)
                rows.append(
                    {
                        "variant": variant,
                        "threads": p,
                        "trial": trial,
                        "iterations_per_worker": iterations,
                        "cache_line_bytes": cache_line,
                        "stride_int64": stride,
                        "seconds": f"{elapsed:.6f}",
                        "correct": correct,
                        "perf_cache_misses": "not_available_on_macos" if os.uname().sysname == "Darwin" else "not_collected",
                    }
                )
    write_csv(
        DATA_DIR / "lab4_false_sharing.csv",
        ["variant", "threads", "trial", "iterations_per_worker", "cache_line_bytes", "stride_int64", "seconds", "correct", "perf_cache_misses"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
