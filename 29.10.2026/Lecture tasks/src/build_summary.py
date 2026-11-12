#!/usr/bin/env python3
"""Generate a human-readable report from the completed, unaltered telemetry.

This performs no benchmark and never changes measured files. Run it after
the experiments and independent numerical verification have completed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from common import ROOT, RESULTS

STOCK_VERIFIER_SHA256 = "bce8d586fd800f888d5001cb6919fd915590f9ec25dab1ab62bd703a502ebb21"


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def read_tsv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def number(value, digits=3):
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    host = read_json("host_profile.json")
    params = read_json("salted_params.json")
    strides = {int(row["stride"]): row for row in read_tsv("stride_results.tsv")}
    dense = read_json("loop_interchange.json")
    sharing = read_tsv("false_sharing_karp_flatt.tsv")
    sharing_raw = read_json("false_sharing_trials.json")
    scaling = read_tsv("scaling_metrics.tsv")
    scale_raw = read_json("scaling_trials.json")
    roof = read_json("roofline_summary.json")
    strong = [row for row in scaling if row["mode"] == "strong"]
    weak = [row for row in scaling if row["mode"] == "weak"]
    best_strong = max(strong, key=lambda row: float(row["speedup"]))
    best_weak = max(weak, key=lambda row: float(row["throughput_speedup"]))
    max_share = max(sharing, key=lambda row: int(row["p"]))
    max_p = int(max_share["p"])
    line_bytes = int(host["cache_line_bytes"])
    line_doubles = line_bytes // 8
    cliff = strides.get(line_doubles, strides[16])
    cliff_ratio = float(cliff["latency_ns"]) / float(strides[1]["latency_ns"])
    padded_recovery = float(max_share["t_unpadded_s"]) / float(max_share["t_padded_s"])
    private_recovery = float(max_share["t_unpadded_s"]) / float(max_share["t_privatized_s"])
    fit = scale_raw["amdahl_fit"]
    weak_one = next(row for row in weak if int(row["p"]) == 1)
    uses_weak_baseline = abs(float(weak_one["throughput_speedup"]) - 1.0) < 1e-9
    baseline_note = (
        "Strong and weak metrics use the measured P=1 median from their own study; both P=1 rows normalize to one."
        if uses_weak_baseline else
        "Both studies use the strong-study P=1 median as a shared baseline; the separately measured weak P=1 row can differ from one because of timing variation."
    )
    stock = ROOT / "verify_outputs.py"
    stock_hash = hashlib.sha256(stock.read_bytes()).hexdigest()
    if stock_hash != STOCK_VERIFIER_SHA256:
        raise AssertionError("Provided verifier changed; do not report it as unmodified")
    verification_path = RESULTS / "numerical_verification.json"
    verification = read_json("numerical_verification.json") if verification_path.exists() else None
    local_status = "not yet generated"
    if verification is not None:
        value = next((verification[key] for key in
                      ("local_numerical_status", "status", "overall_status", "all_passed", "passed")
                      if key in verification), "see report")
        local_status = "PASS" if value is True else "FAIL" if value is False else str(value)
    verifier_candidates = ("src/verify_numerics.py", "src/verify_numerical.py",
                           "src/verify_results.py", "verify_numerics.py")
    local_script = next((name for name in verifier_candidates if (ROOT / name).exists()), None)
    verify_command = f"python3 {local_script}" if local_script else "# Run the independent verifier linked in the verification report."
    console_logs = ("stride_console.txt", "dense_console.txt", "false_sharing_console.txt", "scaling_console.txt")
    links = " · ".join(f"[{name.removesuffix('_console.txt')} log](results/{name})"
                       for name in console_logs if (RESULTS / name).exists())
    advertised = roof.get("theoretical_memory_bandwidth_gb_s")
    advertised_source = roof.get("theoretical_memory_bandwidth_source")
    advertised_note = (
        f"Apple advertises up to [{number(advertised, 0)} GB/s unified-memory bandwidth]({advertised_source}); "
        "this is a manufacturer maximum shared by the system, not this run's measured DRAM bandwidth."
        if advertised and advertised_source else
        "A manufacturer peak bandwidth was not supplied; it is not inferred from timings."
    )
    frequency_note = "reported in the host profile" if host.get("base_frequency_hz") else "unreported"
    l3_note = "reported in the host profile" if host.get("l3_cache_bytes") else "unreported"
    cache_rows = []
    for level in host.get("performance_levels", []):
        cache_rows.append([level.get("name", "unknown"), level.get("physicalcpu", "—"),
                           number(float(level.get("l1dcachesize", 0)) / 1024, 0),
                           number(float(level.get("l2cachesize", 0)) / 1024**2, 0)])
    cache_table = table(["Core group", "Physical cores", "L1D KiB/core", "L2 MiB/group"], cache_rows) if cache_rows else "See the measured cache fields in the host profile."
    stride_table = table(["Stride", "Median ns/read", "Logical read GB/s"],
                         [[s, number(strides[s]["latency_ns"]), number(strides[s]["effective_read_GB_s"])]
                          for s in (1, 8, 16)])
    dense_table = table(["Loop order", "Three samples (s)", "Median (s)"], [
        ["i-j-k", ", ".join(number(t, 6) for t in dense["trials_s"]["naive"]), number(dense["naive_time_s"], 6)],
        ["i-k-j", ", ".join(number(t, 6) for t in dense["trials_s"]["interchanged"]), number(dense["interchanged_time_s"], 6)],
    ])
    sharing_table = table(["P", "Adjacent (s)", "64 B padding (s)", f"{line_bytes} B padding (s)", "Private (s)"],
                          [[row["p"], *(number(row[f"t_{variant}_s"], 6) for variant in
                                        ("unpadded", "padded64", "padded", "privatized"))]
                           for row in sharing])
    sharing_metrics = table(["P", "Raw e adjacent", "Raw e native padding", "Work-normalized speedup adjacent", "Work-normalized speedup native padding"],
                            [[row["p"], number(row["e_unpadded"]), number(row["e_padded"]),
                              number(row["work_normalized_speedup_unpadded"]), number(row["work_normalized_speedup_padded"])]
                             for row in sharing])
    strong_table = table(["P", "N", "Median (s)", "Speedup", "Efficiency %"],
                         [[row["p"], row["N"], number(row["time_s"], 6), number(row["speedup"]), number(row["efficiency_pct"], 1)]
                          for row in strong])
    weak_table = table(["P", "N", "Median (s)", "Throughput speedup", "Efficiency %", "PDF formula"],
                       [[row["p"], row["N"], number(row["time_s"], 6), number(row["throughput_speedup"]),
                         number(row["efficiency_pct"], 1), number(row["weak_efficiency_pdf_formula"])] for row in weak])
    counter_note = (f"All {len(sharing_raw['trials'])} trials returned exactly P × K increments, including {max_p * int(sharing_raw['iterations_per_worker']):,} at P={max_p}."
                    if sharing_raw.get("all_counts_exact") else "Counter verification status is recorded per trial; inspect the raw telemetry.")
    stock_status = (f"**Provided verifier: incompatible with this host.** It requires `cache_line_bytes == 64`; the measured value is {line_bytes}. Its actual failure is retained in [provided_verifier.log](results/provided_verifier.log)."
                    if line_bytes != 64 else
                    "The provided verifier's actual status is retained in [provided_verifier.log](results/provided_verifier.log).")
    receipt_note = (
        "[verification_receipt.sig](results/verification_receipt.sig) is a local numerical self-audit receipt, not an authenticated instructor receipt or proof that the supplied checker passed."
        if (RESULTS / "verification_receipt.sig").exists() else
        "No receipt has yet been generated; this report does not imply that verification passed."
    )
    text = f"""# CS482 practicum — 29.10.2026

On the physical {host['cpu_model']}, stride-{line_doubles} reads cost {cliff_ratio:.2f}× the stride-1 time per element, consistent with its {line_bytes}-byte cache line; changing dense multiplication from i-j-k to i-k-j gave {dense['speedup']:.2f}× speedup. At P={max_p}, native padding gave {padded_recovery:.2f}× and privatization {private_recovery:.2f}× the adjacent-counter throughput for the same work. Strong stencil scaling peaked at {float(best_strong['speedup']):.2f}× with P={best_strong['p']}; weak throughput peaked at {float(best_weak['throughput_speedup']):.2f}× with P={best_weak['p']}. Memory traffic, interpreter work, barriers and heterogeneous-core scheduling limit further scaling; these measurements do not isolate DRAM saturation.

Student: **Nuradil Abyz**, ID **{params['student_id']}**. Salted N={params['N']}, seed={params['seed']}. Implementation and analysis prepared with AI assistance; timings collected on the physical host.

## Host and method

[Host profile](results/host_profile.json), [raw sysctl](results/sysctl_raw.txt), [salted parameters](results/salted_params.json). Captured {host['captured_at_utc']}; {host['platform']}; {host['python_implementation']} {host['python_version']}, NumPy {host['numpy_version']}; {host['physical_cores']} physical cores / {host['logical_threads']} threads, {float(host['memory_bytes'])/1024**3:.0f} GiB memory. No core pinning. Experiments ran sequentially, with parallel workers only within each experiment.

{cache_table}

Base frequency: {frequency_note}. L3: {l3_note}. Cache-hit/miss counters were not collected. Generic sysctl cache sizes are supplemented by the heterogeneous per-core-group values above; unavailable telemetry is not invented.

## M1 — locality and loop interchange

The contiguous float64 buffer contains {params['buffer_elements']:,} elements ({params['buffer_elements']*8/1024**2:.0f} MiB). Each stride has five warmups and five recorded trials; the median is normalized by the actual number of elements touched. On this host, a line holds {line_doubles} doubles, so the PDF's eight-double/64-byte cache-line assumption does not apply literally.

{stride_table}

Dense tests allocate full N×N inputs and multiply their leading M×M blocks (M={dense['M']}, original row pitch N), with three recorded runs per order. Explicit [C loops](src/m1_kernels.c) are called by Python `ctypes`; benchmark loops do not call BLAS. Allocation and output zeroing are excluded. Compiler: {dense['compiler']}; flags: `{' '.join(dense['compiler_flags'])}`.

{dense_table}

Maximum absolute error between loop orders: {dense['max_abs_error_interchanged_vs_naive']:.3g}; against NumPy: {dense['max_abs_error_interchanged_vs_reference']:.3g}, with `atol=1e-12, rtol=0`. Raw timings and salted rounded-six-decimal digests: [stride TSV](results/stride_results.tsv), [stride trials](results/stride_trials.json), [dense results](results/loop_interchange.json).

## M2 — false sharing

K={sharing_raw['iterations_per_worker']:,} increments per worker, {sharing_raw['recorded_trials']} trials; medians below exclude spawn and result IPC, and span the synchronized worker interval. Counters use aligned shared storage and distinct ownership with no global per-increment lock; private counters publish once under a lock. {counter_note}

{sharing_table}

64-byte spacing can leave two counters on one {line_bytes}-byte cache line. Native padding separates them; observed recovery is limited, not assumed perfect. At P={max_p}, adjacent/native-padding runtime ratio is {padded_recovery:.3f}×.

{sharing_metrics}

Every worker performs K operations, so total work grows as P×K. The PDF-requested `S=T1/TP` and `e=(1/S−1/P)/(1−1/P)` are raw-time diagnostics here, **not fixed-work serial fractions**. Work-normalized speedup is `P*T1/TP`; its serial baseline is estimated, not independently measured. e(1) is undefined. Python dispatch, OS scheduling and mixed core speeds can obscure coherence effects. [Raw trials and exact counters](results/false_sharing_trials.json) · [all metrics](results/false_sharing_karp_flatt.tsv).

## M3 — stencil scaling and Roofline

Five-point float64 stencil, {scale_raw['steps']} timesteps, {scale_raw['trials_per_configuration']} trials per configuration. Processes own disjoint row ranges in two shared buffers; every timestep ends at a barrier. Timings include barriers, excluding allocation, spawn, reference checks and result copies; setup times are recorded separately. Each complete trial is checked against its matching serial reference, including zero boundaries and salted digest.

Strong scaling keeps N={params['N']} fixed:

{strong_table}

Weak scaling uses `N(P)=ceil(N*sqrt(P))` and actual interior-cell work:

{weak_table}

{baseline_note} Throughput speedup is `(WP/W1)*(T1/TP)`; efficiency is that value/P. The PDF's `P*W1*T1/(WP*TP)` approximates weak efficiency rather than scaled speedup when WP≈P×W1.

Amdahl least-squares fit: s={fit['serial_fraction']:.6f} (unconstrained {fit['serial_fraction_unconstrained']:.6f}), ideal asymptotic ceiling {number(fit['asymptotic_speedup_ceiling'])}×, normalized-time RMSE {fit['normalized_runtime_rmse']:.4f}. This is an effective model, not a measured immutable serial fraction. The reported Gustafson `P−s(P−1)` is an **illustrative prediction using the strong-fit s**; a scaled-run serial share was not separately measured. [Scaling trials and fit](results/scaling_trials.json) · [scaling TSV](results/scaling_metrics.tsv).

Roofline uses {roof['flops_per_update']} FLOPs/update and an ideal {roof['ideal_dram_bytes_per_update']} B/update with perfect row reuse and no write-allocation overhead: AI={roof['arithmetic_intensity']:.4f} FLOPs/B. The implemented NumPy ufunc passes make {roof['actual_numpy_ufunc_logical_bytes_per_update']} logical B/update; logical traffic is not measured DRAM traffic. Observed copy bandwidth was {roof['copy_bandwidth_probe']['observed_copy_bandwidth_gb_s']:.3f} GB/s; best stencil throughput was {roof['best_observed_stencil_gflops_s']:.3f} GFLOP/s. {advertised_note} A compute roof and hardware DRAM counters are unavailable, so “memory-bound” remains a model-based hypothesis with synchronization also limiting performance. [Roofline assumptions and raw copy trials](results/roofline_summary.json).

## Verification and task differences

{stock_status}

The [original verifier](verify_outputs.py) is byte-for-byte preserved (SHA-256 `{stock_hash}`). It checks telemetry fields and trends but does not itself verify dense arrays, exact counter totals or stencil arrays. Independent numerical verification status: **{local_status}**; see [numerical_verification.json](results/numerical_verification.json). {receipt_note}

The [original handout](ParallelComputingLabPracticum_Paradigm.pdf) assumes 64-byte cache lines and describes an eight-digit ID; the established repository ID has nine digits and is salted without truncation. Python/NumPy orchestrates every experiment; transparent native C loops are used for dense and stride measurements. No frequencies, L3 size, hardware counters or performance trends were fabricated. The requested submission path is `29.10.2026/Lecture tasks`.

## Reproduce

From this directory, install NumPy in a Python 3.10+ environment and provide `clang` or another C compiler (`CC` may select it). The captured run used CPython {host['python_version']}, NumPy {host['numpy_version']} and {dense['compiler']}. Build products remain in ignored `.build/`; all raw results are retained. These commands overwrite results with new genuine host measurements:

```bash
python3 src/profile_env.py --student-id {params['student_id']}
python3 src/stride_bench.py
python3 src/dense_kernel.py
python3 src/false_sharing.py
python3 src/scaling_study.py
{verify_command}
python3 src/build_summary.py
# The unchanged instructor checker exits unsuccessfully on this 128-byte host:
python3 verify_outputs.py
```

{links}
"""
    if len(text.splitlines()) > 150:
        raise AssertionError("Report unexpectedly exceeds the compact report budget")
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print(f"Wrote README.md ({len(text.splitlines())} lines) from completed telemetry")


if __name__ == "__main__":
    main()
