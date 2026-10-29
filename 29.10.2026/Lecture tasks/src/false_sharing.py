#!/usr/bin/env python3
"""Measure process-owned counters with and without cache-line sharing.

The PDF assigns K increments to *each* worker: total work is P*K.  Thus its
T1/TP speedup and Karp-Flatt columns are retained for task compatibility, but
are not a fixed-work serial-fraction estimate.  Work-normalized diagnostics
are reported separately.  Measurements are never forced to follow a trend.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import multiprocessing as mp
from pathlib import Path
from statistics import median
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VARIANTS = ("unpadded", "padded64", "padded", "privatized")


def _write_json(name: str, value: object) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _cache_line_bytes() -> int:
    profile = json.loads((RESULTS / "host_profile.json").read_text(encoding="utf-8"))
    line = int(profile["cache_line_bytes"])
    if line < 8 or line % 8:
        raise ValueError(f"Unsupported measured cache-line size: {line}")
    return line


def _counter_worker(raw, offset_bytes, slots, index, iterations, variant,
                    alignment, start_barrier, final_write_lock, messages):
    """Each process owns one counter; no lock is used in the timed shared loop."""
    try:
        counters = (ctypes.c_int64 * slots).from_buffer(raw, offset_bytes)
        if ctypes.addressof(counters) % alignment:
            raise RuntimeError("Shared array is not aligned after process spawn")
        start_barrier.wait(timeout=60)
        started = time.perf_counter_ns()
        if variant == "privatized":
            local_count = 0
            for _ in range(iterations):
                local_count += 1
            # One synchronized publication, never a lock per increment.
            with final_write_lock:
                counters[index] = local_count
        else:
            for _ in range(iterations):
                counters[index] += 1
        finished = time.perf_counter_ns()
        messages.put({"index": index, "count": int(counters[index]),
                      "start_ns": started, "end_ns": finished,
                      "duration_ns": finished - started,
                      "alignment_remainder": ctypes.addressof(counters) % alignment})
    except BaseException:
        messages.put({"error": traceback.format_exc()})
        try:
            start_barrier.abort()
        except Exception:
            pass
        raise


def run_trial(ctx, workers: int, iterations: int, variant: str, cache_line: int) -> dict:
    """Return actual timed kernel span, counters, and worker timing telemetry."""
    alignment = max(128, cache_line)
    stride = {"unpadded": 1, "padded64": 8,
              "padded": cache_line // 8, "privatized": cache_line // 8}[variant]
    slots = (workers - 1) * stride + 1
    # Extra shared storage permits alignment without relying on allocator luck.
    raw = ctx.RawArray(ctypes.c_int64, slots + alignment // 8)
    offset_bytes = (-ctypes.addressof(raw)) % alignment
    counters = (ctypes.c_int64 * slots).from_buffer(raw, offset_bytes)
    if ctypes.addressof(counters) % alignment:
        raise AssertionError("Failed to align shared counters")
    start_barrier = ctx.Barrier(workers + 1)
    final_write_lock = ctx.Lock()
    messages = ctx.Queue()
    processes = [ctx.Process(target=_counter_worker, args=(
        raw, offset_bytes, slots, worker * stride, iterations, variant,
        alignment, start_barrier, final_write_lock, messages))
        for worker in range(workers)]
    telemetry = []
    try:
        for process in processes:
            process.start()
        start_barrier.wait(timeout=60)
        for _ in processes:
            message = messages.get(timeout=600)
            if "error" in message:
                raise RuntimeError(message["error"])
            telemetry.append(message)
        for process in processes:
            process.join(timeout=60)
            if process.exitcode != 0:
                raise RuntimeError(f"Counter worker failed: exitcode={process.exitcode}")
        actual = [int(counters[worker * stride]) for worker in range(workers)]
        if actual != [iterations] * workers:
            raise AssertionError(f"Counter mismatch: {actual}, expected {iterations} each")
        elapsed_ns = max(item["end_ns"] for item in telemetry) - min(item["start_ns"] for item in telemetry)
        if elapsed_ns <= 0:
            raise AssertionError("Non-positive measured duration")
        return {"variant": variant, "p": workers, "iterations_per_worker": iterations,
                "stride_int64": stride, "spacing_bytes": stride * 8,
                "alignment_bytes": alignment, "elapsed_ns": elapsed_ns,
                "elapsed_s": elapsed_ns / 1e9, "expected_total": workers * iterations,
                "total_count": sum(actual), "counters": actual,
                "worker_timings": sorted(telemetry, key=lambda item: item["index"])}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
        messages.close()
        messages.join_thread()


def karp_flatt(speedup: float, workers: int):
    if workers == 1:
        return None  # e(1) is undefined, not zero.
    return (1.0 / speedup - 1.0 / workers) / (1.0 - 1.0 / workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5_000_000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    workers = sorted(set(args.workers))
    if args.iterations <= 0 or args.trials <= 0 or not workers or workers[0] != 1 or any(p <= 0 for p in workers):
        parser.error("Positive iterations/trials/workers required, including P=1 baseline")
    if args.iterations > (2**63 - 1):
        parser.error("Iteration count exceeds signed int64 capacity")
    cache_line = _cache_line_bytes()
    ctx = mp.get_context("spawn")
    raw_runs = []
    result = {
        "schema_version": 1,
        "cache_line_bytes": cache_line,
        "iterations_per_worker": args.iterations,
        "recorded_trials": args.trials,
        "process_start_method": "spawn",
        "timing_scope": "max(worker end_ns) - min(worker start_ns), after a synchronized barrier; excludes spawn and result IPC",
        "workload": "Each worker performs K increments; total work is P*K (weak scaling)",
        "synchronization": "No global lock in shared counter increments; privatized variant locks once for final publication",
        "variants": {"unpadded": "adjacent int64 counters", "padded64": "PDF-prescribed 64-byte spacing",
                     "padded": "measured native cache-line spacing", "privatized": "local Python int loop, one locked final write"},
        "metric_definitions": {
            "s_variant": "T_variant(1) / T_variant(P), as requested by PDF; workload is not fixed",
            "e_variant": "(1/s_variant - 1/P)/(1 - 1/P), PDF raw-time diagnostic, not a valid fixed-work serial fraction here",
            "work_normalized_speedup_variant": "P * T_variant(1) / T_variant(P), assumes serial work scales linearly with P",
            "work_normalized_e_variant": "Karp-Flatt applied to work-normalized speedup; diagnostic only, not a fitted serial fraction",
            "p1_e": "undefined (blank in TSV, null in JSON)"
        },
        "limitations": [
            "macOS process scheduling and heterogeneous cores are uncontrolled; no portable CPU pinning used",
            "Pure Python interpreter overhead can obscure coherence effects; a collapse or monotonic trend is not guaranteed",
            "64-byte spacing can leave two counters sharing one 128-byte line on this host",
            "A synthetic serial-time normalization P*T1 is not an independently measured serial P*K baseline"
        ],
        "trials": raw_runs,
    }
    # Rotate variant order across trials to reduce systematic order/thermal bias.
    for trial in range(args.trials):
        order = VARIANTS[trial % len(VARIANTS):] + VARIANTS[:trial % len(VARIANTS)]
        for p in workers:
            for variant in order:
                measured = run_trial(ctx, p, args.iterations, variant, cache_line)
                measured["trial"] = trial + 1
                raw_runs.append(measured)
                _write_json("false_sharing_trials.json", result)
                print(f"M2 trial={trial + 1} P={p} {variant}: {measured['elapsed_s']:.6f}s counts={measured['total_count']}", flush=True)
    medians = {(p, variant): median(r["elapsed_s"] for r in raw_runs if r["p"] == p and r["variant"] == variant)
               for p in workers for variant in VARIANTS}
    rows = []
    for p in workers:
        row = {"p": p, "iterations_per_worker": args.iterations, "total_work": p * args.iterations}
        for variant in VARIANTS:
            duration = medians[p, variant]
            baseline = medians[1, variant]
            speedup = baseline / duration
            normalized = p * speedup
            row[f"t_{variant}_s"] = duration
            row[f"s_{variant}"] = speedup
            row[f"e_{variant}"] = karp_flatt(speedup, p)
            row[f"work_normalized_speedup_{variant}"] = normalized
            row[f"work_normalized_e_{variant}"] = karp_flatt(normalized, p)
        rows.append(row)
    result["all_counts_exact"] = all(r["counters"] == [args.iterations] * r["p"] and r["total_count"] == r["expected_total"] for r in raw_runs)
    result["summary"] = rows
    _write_json("false_sharing_trials.json", result)
    with (RESULTS / "false_sharing_karp_flatt.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("M2 exact counter validation passed for every process in every trial.", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
