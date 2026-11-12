"""Lab 3: static and dynamic scheduling of irregular Mandelbrot row chunks."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from numba import njit

from common import DATA_DIR, ensure_output_dirs, write_csv


THREAD_COUNTS = (2, 4, 8, 16)
CHUNK_SIZES = (1, 16, 64, 256)


@njit(cache=True)
def compute_rows(start_row: int, end_row: int, width: int, height: int, max_iter: int) -> tuple[int, int]:
    work = 0
    checksum = 0
    for py in range(start_row, end_row):
        y0 = (py - height / 2.0) * 4.0 / height
        for px in range(width):
            x0 = (px - width / 2.0) * 4.0 / width
            x = 0.0
            y = 0.0
            iteration = 0
            while x * x + y * y <= 4.0 and iteration < max_iter:
                xtemp = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = xtemp
                iteration += 1
            work += iteration
            checksum += (iteration * (px + 1) * (py + 1)) & 0xFFFFFFFF
    return work, checksum


def _static_worker(worker_id: int, chunks: list[tuple[int, int]], width: int, height: int, max_iter: int, result_queue) -> None:
    total_work = 0
    checksum = 0
    for start, end in chunks:
        work, part_checksum = compute_rows(start, end, width, height, max_iter)
        total_work += work
        checksum += part_checksum
    result_queue.put((worker_id, total_work, checksum))


def _dynamic_worker(worker_id: int, task_queue, result_queue, width: int, height: int, max_iter: int) -> None:
    total_work = 0
    checksum = 0
    while True:
        item = task_queue.get()
        if item is None:
            break
        start, end = item
        work, part_checksum = compute_rows(start, end, width, height, max_iter)
        total_work += work
        checksum += part_checksum
    result_queue.put((worker_id, total_work, checksum))


def make_chunks(height: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, height)) for start in range(0, height, chunk_size)]


def imbalance(per_worker: dict[int, int]) -> float:
    values = list(per_worker.values())
    if not values:
        return 0.0
    average = sum(values) / len(values)
    return (max(values) - min(values)) / average if average else 0.0


def run_policy(policy: str, workers: int, chunk_size: int, width: int, height: int, max_iter: int) -> tuple[float, float, int]:
    chunks = make_chunks(height, chunk_size)
    context = mp.get_context("spawn")
    per_worker: dict[int, int] = {}
    checksum = 0
    result_queue = context.Queue()
    processes = []
    start_time = time.perf_counter()
    if policy == "dynamic":
        task_queue = context.Queue()
        for chunk in chunks:
            task_queue.put(chunk)
        for _ in range(workers):
            task_queue.put(None)
        for worker_id in range(workers):
            process = context.Process(target=_dynamic_worker, args=(worker_id, task_queue, result_queue, width, height, max_iter))
            process.start()
            processes.append(process)
    else:
        assignments = [[] for _ in range(workers)]
        for index, chunk in enumerate(chunks):
            assignments[index % workers].append(chunk)
        for worker_id, assignment in enumerate(assignments):
            process = context.Process(target=_static_worker, args=(worker_id, assignment, width, height, max_iter, result_queue))
            process.start()
            processes.append(process)
    for _ in range(workers):
        worker_id, work, part_checksum = result_queue.get()
        per_worker[worker_id] = work
        checksum += part_checksum
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"scheduler worker exited with {process.exitcode}")
    return time.perf_counter() - start_time, imbalance(per_worker), checksum


def run(profile: str) -> None:
    ensure_output_dirs()
    if profile == "full":
        width, height, max_iter, trials = 1920, 1080, 1000, 3
    else:
        width, height, max_iter, trials = 400, 240, 300, 3

    compute_rows(0, 2, 10, 10, 10)
    rows = []
    for policy in ("static", "dynamic"):
        for p in THREAD_COUNTS:
            for chunk_size in CHUNK_SIZES:
                for trial in range(1, trials + 1):
                    elapsed, imbalance_value, checksum = run_policy(policy, p, chunk_size, width, height, max_iter)
                    rows.append(
                        {
                            "policy": policy,
                            "threads": p,
                            "chunk_size": chunk_size,
                            "trial": trial,
                            "width": width,
                            "height": height,
                            "max_iter": max_iter,
                            "seconds": f"{elapsed:.6f}",
                            "work_imbalance": f"{imbalance_value:.6f}",
                            "checksum": checksum,
                        }
                    )
    write_csv(
        DATA_DIR / "lab3_scheduling.csv",
        ["policy", "threads", "chunk_size", "trial", "width", "height", "max_iter", "seconds", "work_imbalance", "checksum"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
