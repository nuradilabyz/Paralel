#!/usr/bin/env python3
"""Reproducible strong/weak scaling of a five-point float64 stencil.

Each process owns a disjoint block of output rows. Two SharedMemory buffers and
one barrier per timestep prevent a process reading its neighbour's next state.
Spawn/allocation/reference checks are measured or recorded separately from the
compute interval. The interval includes the start/end barriers and all timestep
barriers; it therefore measures a usable process-based implementation rather
than isolated NumPy operations. No affinity is asserted on macOS.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import queue
import statistics
import time
import traceback
import numpy as np

from common import ROOT, RESULTS, load_params, write_json, digest_array


def stencil_rows(src, dst, first, last):
    """Five input values, four additions and one multiplication per update."""
    if first >= last:
        return
    target = dst[first:last, 1:-1]
    np.add(src[first:last, 1:-1], src[first-1:last-1, 1:-1], out=target)
    np.add(target, src[first+1:last+1, 1:-1], out=target)
    np.add(target, src[first:last, :-2], out=target)
    np.add(target, src[first:last, 2:], out=target)
    np.multiply(target, 0.2, out=target)


def initial_grid(n, seed):
    grid = np.random.default_rng(seed + n).random((n, n), dtype=np.float64)
    grid[0, :] = grid[-1, :] = 0.0
    grid[:, 0] = grid[:, -1] = 0.0
    return grid


def serial_reference(initial, steps):
    """One-process reference; same arithmetic order as each parallel row."""
    buffers = [initial.copy(), np.zeros_like(initial)]
    start = time.perf_counter_ns()
    for step in range(steps):
        stencil_rows(buffers[step % 2], buffers[(step + 1) % 2],
                     1, initial.shape[0] - 1)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return buffers[steps % 2], elapsed


def worker(shm_names, n, first, last, steps, start_barrier, step_barrier,
           finish_barrier, messages, worker_id):
    memories = []
    buffers = []
    try:
        memories = [shared_memory.SharedMemory(name=name) for name in shm_names]
        buffers = [np.ndarray((n, n), dtype=np.float64, buffer=m.buf)
                   for m in memories]
        # Exercise the ufunc dispatch on unrelated data before the timed run.
        tiny = np.ones((16, 16), dtype=np.float64)
        warm = np.zeros_like(tiny)
        stencil_rows(tiny, warm, 1, 15)
        messages.put({"kind": "ready", "worker": worker_id})
        start_barrier.wait(timeout=120)
        for step in range(steps):
            stencil_rows(buffers[step % 2], buffers[(step + 1) % 2], first, last)
            step_barrier.wait(timeout=120)
        finish_barrier.wait(timeout=120)
    except BaseException:
        messages.put({"kind": "error", "worker": worker_id,
                      "traceback": traceback.format_exc()})
        for barrier in (start_barrier, step_barrier, finish_barrier):
            try:
                barrier.abort()
            except Exception:
                pass
        raise
    finally:
        buffers.clear()
        for memory in memories:
            memory.close()


def parallel_trial(initial, p, steps):
    ctx = mp.get_context("spawn")
    n = initial.shape[0]
    memories = []
    processes = []
    buffers = []
    messages = ctx.Queue()
    setup_start = time.perf_counter_ns()
    try:
        memories = [shared_memory.SharedMemory(create=True, size=initial.nbytes)
                    for _ in range(2)]
        buffers = [np.ndarray(initial.shape, dtype=np.float64, buffer=m.buf)
                   for m in memories]
        buffers[0][:] = initial
        buffers[1].fill(0.0)
        start_barrier = ctx.Barrier(p + 1)
        step_barrier = ctx.Barrier(p)
        finish_barrier = ctx.Barrier(p + 1)
        for rank in range(p):
            first = 1 + (n - 2) * rank // p
            last = 1 + (n - 2) * (rank + 1) // p
            process = ctx.Process(
                target=worker,
                args=([m.name for m in memories], n, first, last, steps,
                      start_barrier, step_barrier, finish_barrier, messages, rank),
            )
            process.start()
            processes.append(process)
        ready = set()
        while len(ready) < p:
            try:
                item = messages.get(timeout=120)
            except queue.Empty as exc:
                raise RuntimeError("Workers did not become ready in 120 seconds") from exc
            if item["kind"] == "error":
                raise RuntimeError(item["traceback"])
            ready.add(item["worker"])
        setup_s = (time.perf_counter_ns() - setup_start) / 1e9
        start = time.perf_counter_ns()
        start_barrier.wait(timeout=120)
        finish_barrier.wait(timeout=120)
        elapsed_s = (time.perf_counter_ns() - start) / 1e9
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f"Stencil worker exited with {process.exitcode}")
        result = buffers[steps % 2].copy()
        return result, elapsed_s, setup_s
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        buffers.clear()
        for memory in memories:
            memory.close()
            memory.unlink()
        messages.close()
        messages.join_thread()


def worker_counts(physical, requested):
    limit = min(physical, requested) if requested else physical
    values = [1]
    while values[-1] * 2 <= limit:
        values.append(values[-1] * 2)
    if values[-1] != limit:
        values.append(limit)
    return values


def bandwidth_probe(elements=16_777_216, trials=5):
    """Observed NumPy copy bandwidth, not a theoretical DRAM peak."""
    source = np.ones(elements, dtype=np.float64)
    target = np.empty_like(source)
    np.copyto(target, source)
    samples = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        np.copyto(target, source)
        samples.append((time.perf_counter_ns() - start) / 1e9)
    if not np.array_equal(source, target):
        raise AssertionError("Copy bandwidth probe returned incorrect data")
    median = statistics.median(samples)
    return {"method": "single-process NumPy copyto of 128 MiB float64 array",
            "elements": elements, "times_s": samples, "median_time_s": median,
            "logical_bytes_per_copy": source.nbytes * 2,
            "observed_copy_bandwidth_gb_s": source.nbytes * 2 / median / 1e9,
            "caveat": "Logical read+write bytes; no hardware DRAM counter; write allocation and cache effects are not measured."}


def fit_amdahl(strong_rows):
    """Least-squares fit T(P)/T(1)=1/P+s*(1-1/P), anchored at P=1."""
    t1 = strong_rows[0]["time_s"]
    terms = [(1.0 - 1.0 / row["p"], row["time_s"] / t1 - 1.0 / row["p"])
             for row in strong_rows if row["p"] > 1]
    denom = sum(x * x for x, _ in terms)
    unconstrained = sum(x * y for x, y in terms) / denom if denom else 0.0
    constrained = max(0.0, min(1.0, unconstrained))
    residuals = [row["time_s"] / t1 - (constrained + (1-constrained)/row["p"])
                 for row in strong_rows]
    return {"model": "T(P)/T(1) = s + (1-s)/P; least squares on normalized runtimes",
            "serial_fraction_unconstrained": unconstrained,
            "serial_fraction": constrained,
            "fit_constraint": "s restricted to [0,1]; unconstrained value retained",
            "normalized_runtime_rmse": math.sqrt(statistics.mean(x*x for x in residuals)),
            "asymptotic_speedup_ceiling": 1.0/constrained if constrained > 0 else None,
            "ceiling_note": "null means infinite in the fitted ideal model; extrapolation is not a measured hardware guarantee",
            "limitations": "Memory contention, process barriers and heterogeneous cores violate a constant serial-fraction model."}



WEAK_FORMULA_NOTE = (
    "Each mode uses its own measured P=1 median. The PDF's P*W1*T1/(WP*TP) "
    "is approximately weak efficiency when WP=P*W1; throughput speedup is "
    "(WP/W1)*(T1/TP). Exact interior-cell work is used."
)
GUSTAFSON_MODEL_NOTE = (
    "Illustrative P-s*(P-1) curve using the serial fraction fitted to strong-scaling "
    "runtimes. This reuses a strong-scaling model parameter; it does not measure "
    "the serial time fraction of each scaled weak workload and is not an empirical "
    "Gustafson-Barsis verification. Observed weak throughput is reported separately."
)


def summarize_records(records):
    """Derive all reporting rows solely from immutable raw trial measurements."""
    rows = []
    for record in records:
        n, steps = int(record["N"]), int(record["steps"])
        updates = (n - 2) ** 2 * steps
        elapsed = statistics.median(t["compute_time_s"] for t in record["trials"])
        setup = statistics.median(t["setup_spawn_time_s"] for t in record["trials"])
        rows.append({"mode": record["mode"], "p": int(record["p"]), "N": n,
                     "steps": steps, "work_updates": updates, "time_s": elapsed,
                     "setup_time_s": setup, "gflops_s": 5*updates/elapsed/1e9,
                     "updates_per_s": updates/elapsed})
    strong = sorted((row for row in rows if row["mode"] == "strong"), key=lambda r: r["p"])
    fit = fit_amdahl(strong)
    baselines = {}
    for mode in ("strong", "weak"):
        baseline = next(row for row in rows if row["mode"] == mode and row["p"] == 1)
        baselines[mode] = {"p": 1, "time_s": baseline["time_s"],
                           "work_updates": baseline["work_updates"],
                           "source": f"median of this {mode} mode's own P=1 raw compute times"}
    for row in rows:
        p = row["p"]
        baseline = baselines[row["mode"]]
        t1, base_work = baseline["time_s"], baseline["work_updates"]
        row["baseline_time_s"] = t1
        row["baseline_work_updates"] = base_work
        if row["mode"] == "strong":
            speedup = t1 / row["time_s"]
            row["speedup"] = speedup
            row["efficiency_pct"] = speedup / p * 100
            row["amdahl_serial_fraction_per_p"] = ((1/speedup-1/p)/(1-1/p)) if p > 1 else ""
            row["weak_efficiency_pdf_formula"] = ""
            row["throughput_speedup"] = speedup
        else:
            throughput = (row["work_updates"]/base_work) * t1/row["time_s"]
            row["speedup"] = ""
            row["efficiency_pct"] = throughput/p*100
            row["amdahl_serial_fraction_per_p"] = ""
            row["weak_efficiency_pdf_formula"] = p*base_work*t1/(row["work_updates"]*row["time_s"])
            row["throughput_speedup"] = throughput
        row["gustafson_model_speedup"] = p-fit["serial_fraction"]*(p-1)
    return rows, fit, baselines


def write_scaling_metrics(rows):
    fields = ["mode", "p", "N", "steps", "work_updates", "time_s", "setup_time_s",
              "baseline_time_s", "baseline_work_updates", "speedup", "efficiency_pct",
              "amdahl_serial_fraction_per_p", "weak_efficiency_pdf_formula",
              "throughput_speedup", "gustafson_model_speedup", "gflops_s", "updates_per_s"]
    with (RESULTS / "scaling_metrics.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def recompute_reports():
    """Reporting-only repair: preserve raw measurements and numerical digests."""
    data = json.loads((RESULTS / "scaling_trials.json").read_text())
    raw_before = json.dumps(data["records"], sort_keys=True)
    rows, fit, baselines = summarize_records(data["records"])
    data.update({"amdahl_fit": fit, "mode_baselines": baselines,
                 "weak_formula_note": WEAK_FORMULA_NOTE,
                 "gustafson_model_note": GUSTAFSON_MODEL_NOTE})
    assert json.dumps(data["records"], sort_keys=True) == raw_before
    write_scaling_metrics(rows)
    write_json("scaling_trials.json", data)
    roof_path = RESULTS / "roofline_summary.json"
    if roof_path.exists():
        roof = json.loads(roof_path.read_text())
        roof.update({"amdahl_fit": fit, "mode_baselines": baselines,
                     "gustafson_model_note": GUSTAFSON_MODEL_NOTE})
        write_json("roofline_summary.json", roof)
    print("Recomputed derived scaling summaries; raw timings, checksums and arrays preserved.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--recompute-only", action="store_true",
                        help="Rebuild derived summaries from existing raw trials; run no benchmarks")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    if args.recompute_only:
        recompute_reports()
        return
    if args.steps < 1 or args.trials < 1 or (args.max_workers is not None and args.max_workers < 1):
        parser.error("steps, trials and max-workers must be positive")
    RESULTS.mkdir(parents=True, exist_ok=True)
    params = load_params()
    host = json.loads((RESULTS / "host_profile.json").read_text())
    physical = int(host["physical_cores"])
    counts = worker_counts(physical, args.max_workers)
    base_n = int(params["N"])
    sid = str(params["student_id"])
    seed = int(params["seed"])
    rows, records, references = [], [], {}
    max_worker = max(counts)
    for mode in ("strong", "weak"):
        for p in counts:
            n = base_n if mode == "strong" else math.ceil(base_n * math.sqrt(p))
            initial = initial_grid(n, seed)
            if n not in references:
                reference, serial_s = serial_reference(initial, args.steps)
                references[n] = (reference, serial_s)
            reference, serial_s = references[n]
            ref_digest = digest_array(reference, sid)
            if mode == "strong" and p == 1:
                np.save(RESULTS / "stencil_reference.npy", reference)
            times, setups, trial_records = [], [], []
            for trial in range(args.trials):
                result, elapsed_s, setup_s = parallel_trial(initial, p, args.steps)
                max_abs_error = float(np.max(np.abs(result - reference)))
                exact = bool(np.array_equal(result, reference))
                if not np.allclose(result, reference, rtol=0.0, atol=1e-12):
                    raise AssertionError(f"{mode}, P={p}, trial={trial}: stencil max error {max_abs_error}")
                if not (np.all(result[0, :] == 0) and np.all(result[-1, :] == 0)
                        and np.all(result[:, 0] == 0) and np.all(result[:, -1] == 0)):
                    raise AssertionError("Stencil zero boundaries changed")
                digest = digest_array(result, sid)
                if digest != ref_digest:
                    raise AssertionError("Rounded salted stencil digest differs from reference")
                trial_records.append({"trial": trial + 1, "compute_time_s": elapsed_s,
                                      "setup_spawn_time_s": setup_s,
                                      "max_abs_error": max_abs_error, "bitwise_equal": exact,
                                      "checksum_sum_float64": float(np.sum(result, dtype=np.float64)),
                                      "salted_rounded6_sha256": digest})
                times.append(elapsed_s)
                setups.append(setup_s)
                print(f"{mode:6s} P={p} N={n} trial={trial+1}: {elapsed_s:.6f}s compute, "
                      f"{setup_s:.3f}s setup; max error={max_abs_error:.3g}", flush=True)
                if mode == "strong" and p == max_worker and trial == args.trials - 1:
                    np.save(RESULTS / "stencil_parallel.npy", result)
                del result
            updates = (n - 2) ** 2 * args.steps
            median = statistics.median(times)
            rows.append({"mode": mode, "p": p, "N": n, "steps": args.steps,
                         "work_updates": updates, "time_s": median,
                         "setup_time_s": statistics.median(setups),
                         "gflops_s": 5 * updates / median / 1e9,
                         "updates_per_s": updates / median})
            records.append({"mode": mode, "p": p, "N": n, "steps": args.steps,
                            "reference_time_s": serial_s,
                            "reference_salted_rounded6_sha256": ref_digest,
                            "trials": trial_records})
            # Retain only the small base grid reference; weak results can be large.
            if n != base_n:
                del references[n]
    rows, fit, baselines = summarize_records(records)
    serial_fraction = fit["serial_fraction"]
    write_scaling_metrics(rows)
    write_json("scaling_trials.json", {
        "student_id": sid, "salted_N": base_n, "steps": args.steps,
        "trials_per_configuration": args.trials,
        "worker_counts": counts, "start_method": "spawn", "numpy_version": np.__version__,
        "kernel": "C_new[i,j] = .2*(C[i,j]+C[i-1,j]+C[i+1,j]+C[i,j-1]+C[i,j+1]); zero boundary",
        "timing": "Parent perf_counter_ns around start/end barriers. Includes every timestep barrier. Excludes allocation, process spawn, warmup, result copy and validation; setup time recorded separately.",
        "warmup": "Each worker exercises the same ufuncs on a separate 16x16 array before ready/start.",
        "validation": "Every full output from every trial compared to its serial reference (atol=1e-12,rtol=0), boundaries checked, salted rounded6 digest checked. Strong base reference and maximum-P output persisted as .npy; large weak arrays are not persisted.",
        "initialization": "np.random.default_rng(salted_seed + grid_dimension).random float64, zero boundaries",
        "weak_formula_note": WEAK_FORMULA_NOTE,
        "mode_baselines": baselines,
        "gustafson_model_note": GUSTAFSON_MODEL_NOTE,
        "affinity": "No CPU pinning; OS chooses placement; physical core count does not imply equal-performance cores.",
        "amdahl_fit": fit, "records": records})
    copy = bandwidth_probe()
    observed_bw = copy["observed_copy_bandwidth_gb_s"]
    best = max(rows, key=lambda row: row["gflops_s"])
    ai = 5 / 16
    theoretical_bw = host.get("theoretical_memory_bandwidth_gb_s")
    write_json("roofline_summary.json", {
        "regime": "memory-bound (working hypothesis; process synchronization also limits scaling)",
        "arithmetic_intensity": ai,
        "flops_per_update": 5,
        "ideal_dram_bytes_per_update": 16,
        "ideal_dram_traffic_assumption": "Perfect spatial/row reuse: input loaded once (8 B/cell), output written once (8 B/cell), write-allocate overhead excluded; finite-grid boundary terms ignored.",
        "no_reuse_fused_kernel_bytes_per_update": 48,
        "no_reuse_fused_kernel_arithmetic_intensity": 5/48,
        "actual_numpy_ufunc_logical_bytes_per_update": 112,
        "ufunc_traffic_note": "First add 24 B, next three adds 24 B each, final in-place multiply16 B. These are logical load/store bytes, not measured DRAM traffic; chunk cache reuse reduces DRAM traffic.",
        "copy_bandwidth_probe": copy,
        "observed_copy_based_ideal_bandwidth_roof_gflops_s": observed_bw * ai,
        "theoretical_memory_bandwidth_gb_s": theoretical_bw,
        "theoretical_memory_bandwidth_source": host.get("theoretical_memory_bandwidth_source"),
        "theoretical_ideal_bandwidth_roof_gflops_s": theoretical_bw * ai if theoretical_bw else None,
        "peak_theoretical_compute_gflops_s": None,
        "best_observed_stencil_gflops_s": best["gflops_s"],
        "best_observed_configuration": {key: best[key] for key in ("mode", "p", "N", "time_s")},
        "cache_profile": {key: value for key, value in host.items() if "cache" in key or "perflevel" in key or key == "performance_levels"},
        "classification_evidence": "Low arithmetic intensity and a memory-streaming five-point stencil make bandwidth/cache traffic a plausible bottleneck. The NumPy implementation also performs multiple ufunc passes. Observed process barriers and scheduler overhead can dominate; no hardware counters establish actual DRAM saturation.",
        "classification_limit": "A complete measured Roofline needs DRAM bytes and a calibrated compute roof. Those are unavailable here; memory-bound is a model-based hypothesis, not a hardware-counter finding. Copy bandwidth is observed and not a theoretical peak.",
        "amdahl_fit": fit,
        "mode_baselines": baselines,
        "gustafson_model_note": GUSTAFSON_MODEL_NOTE,
    })
    print(f"Amdahl fitted s={serial_fraction:.6f}, unconstrained={fit['serial_fraction_unconstrained']:.6f}; "
          f"observed copy bandwidth={observed_bw:.3f} GB/s", flush=True)


if __name__ == "__main__":
    main()
