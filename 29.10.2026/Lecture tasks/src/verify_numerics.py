#!/usr/bin/env python3
"""Independent local numerical audit; never impersonates the supplied verifier.

The supplied verifier is executed unchanged. Its status is recorded separately.
The .sig emitted here is an unsigned deterministic local integrity receipt,
not instructor authentication or evidence that the supplied verifier passed.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ATOL = 1e-12
ORIGINAL_VERIFIER_SHA256 = "bce8d586fd800f888d5001cb6919fd915590f9ec25dab1ab62bd703a502ebb21"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write_json(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_tsv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def close(actual, expected, message, rel=1e-12, absolute=1e-15):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f"{message}: nonfinite value")
    require(math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=absolute),
            f"{message}: {actual} != {expected}")


def positive(values, message):
    require(all(math.isfinite(float(x)) and float(x) > 0 for x in values), message)


def salted_digest(array, student_id):
    # Implement the documented canonical serialization here, independently of common.py.
    canonical = np.asarray(np.round(array, decimals=6), dtype="<f8", order="C")
    return hashlib.sha256(student_id.encode("utf-8") + canonical.tobytes(order="C")).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_environment():
    params, host = read_json("salted_params.json"), read_json("host_profile.json")
    sid = str(params["student_id"])
    require(bool(sid.strip()), "Empty student ID")
    h = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
    require(params == {"student_id": sid, "N": 1024 + (h % 9) * 64,
                       "buffer_elements": 16_777_216, "seed": h % 100_007}, "Salted parameters do not match SHA-256 formula")
    for field in ("cpu_model", "physical_cores", "logical_threads", "cache_line_bytes"):
        require(field in host, f"Missing host field: {field}")
    require(int(host["cache_line_bytes"]) > 0 and int(host["cache_line_bytes"]) % 8 == 0, "Invalid actual cache-line size")
    require(int(host["physical_cores"]) > 0 and int(host["logical_threads"]) >= int(host["physical_cores"]), "Invalid topology")
    require(file_digest(ROOT / "verify_outputs.py") == ORIGINAL_VERIFIER_SHA256, "Supplied verifier differs from the attached original")
    return params, host


def audit_stride(params):
    raw, table = read_json("stride_trials.json"), read_tsv("stride_results.tsv")
    strides = (1, 2, 4, 8, 16, 32, 64, 128)
    require(raw["buffer_elements"] == params["buffer_elements"] and raw["buffer_bytes"] == params["buffer_elements"] * 8, "Stride allocation mismatch")
    require(raw["warmup_iterations_per_stride"] == 5 and raw["recorded_iterations_per_stride"] == 5, "Stride run count mismatch")
    require(len(table) == len(strides) and {int(r["stride"]) for r in table} == set(strides), "Stride TSV coverage mismatch")
    require(len(raw["trials"]) == 40, "Expected 40 recorded stride trials")
    for row in table:
        stride = int(row["stride"])
        trials = [r for r in raw["trials"] if r["stride"] == stride]
        require(len(trials) == 5 and {r["trial"] for r in trials} == set(range(1, 6)), f"Stride {stride}: trial coverage")
        touched = (params["buffer_elements"] + stride - 1) // stride
        require(all(r["elements_touched"] == touched and r["sum"] == touched for r in trials), f"Stride {stride}: incorrect reduction")
        times = [r["elapsed_ns"] for r in trials]
        positive(times, f"Stride {stride}: nonpositive duration")
        med = statistics.median(times)
        require(int(row["elements_touched"]) == touched and float(row["sum"]) == touched, f"Stride {stride}: TSV reduction mismatch")
        require(json.loads(row["trials_ns"]) == times, f"Stride {stride}: raw sample mismatch")
        close(row["median_time_ns"], med, "Stride median")
        close(row["latency_ns"], med / touched, "Stride normalized latency")
        close(row["effective_read_GB_s"], touched * 8 / med, "Stride effective bandwidth")
    require(raw["source_sha256"] == file_digest(ROOT / raw["source"]), "Stride native-source digest mismatch")
    return {"status": "PASS", "recorded_trials_checked": 40, "reduction": "Every stored sum equals exact touched-element count", "timing_medians_recomputed": True}


def audit_dense(params):
    result = read_json("loop_interchange.json")
    n, m, seed, sid = params["N"], params["N"] // 2, params["seed"], params["student_id"]
    require((result["N"], result["M"], result["seed"]) == (n, m, seed), "Dense problem size or seed mismatch")
    rng = np.random.default_rng(seed)
    # Reconstruct full inputs in precisely documented random draw order and pitch.
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)
    independent = a[:m, :m] @ b[:m, :m]
    arrays = {name: np.load(RESULTS / f"dense_{name}.npy", allow_pickle=False)
              for name in ("naive", "interchanged", "reference")}
    errors, hashes = {}, {}
    for name, array in arrays.items():
        require(array.shape == (m, m) and array.dtype == np.float64, f"Dense {name}: shape/dtype mismatch")
        require(bool(np.isfinite(array).all()), f"Dense {name}: nonfinite output")
        np.testing.assert_allclose(array, independent, atol=ATOL, rtol=0)
        errors[name] = float(np.max(np.abs(array - independent)))
        hashes[name] = salted_digest(array, sid)
        require(hashes[name] == result["rounded6_salted_sha256"][name], f"Dense {name}: salted digest mismatch")
    np.testing.assert_allclose(arrays["naive"], arrays["interchanged"], atol=ATOL, rtol=0)
    require(salted_digest(independent, sid) == result["rounded6_salted_sha256"]["reference"], "Independent dense reference digest mismatch")
    require(result["absolute_tolerance"] == ATOL and result["relative_tolerance"] == 0, "Dense tolerance metadata mismatch")
    for variant in ("naive", "interchanged"):
        times = result["trials_s"][variant]
        require(len(times) == 3, "Dense expected three timing trials")
        positive(times, "Dense nonpositive timing")
        close(result[f"{variant}_time_s"], statistics.median(times), "Dense median")
    close(result["speedup"], result["naive_time_s"] / result["interchanged_time_s"], "Dense speedup")
    require(result["source_sha256"] == file_digest(ROOT / result["source"]), "Dense native-source digest mismatch")
    return {"status": "PASS", "independent_reconstruction": "Full seeded N-by-N A and B; leading M-by-M NumPy product", "N": n, "M": m, "atol": ATOL, "rtol": 0, "max_abs_error_vs_independent": errors, "salted_rounded6_sha256": hashes}


def audit_false_sharing(host):
    raw, rows = read_json("false_sharing_trials.json"), read_tsv("false_sharing_karp_flatt.tsv")
    variants, workers, k = ("unpadded", "padded64", "padded", "privatized"), (1, 2, 4, 8), 5_000_000
    require(raw["iterations_per_worker"] == k and raw["recorded_trials"] == 3, "M2 default K/trials not recorded")
    require(raw["cache_line_bytes"] == host["cache_line_bytes"], "M2 cache-line metadata mismatch")
    require(len(raw["trials"]) == len(variants) * len(workers) * 3, "M2 expected 48 raw trials")
    require(len(rows) == 4 and {int(r["p"]) for r in rows} == set(workers), "M2 P coverage mismatch")
    medians = {}
    for p in workers:
        for variant in variants:
            trials = [r for r in raw["trials"] if r["p"] == p and r["variant"] == variant]
            require(len(trials) == 3 and {r["trial"] for r in trials} == {1, 2, 3}, "M2 trial coverage mismatch")
            stride = {"unpadded": 1, "padded64": 8, "padded": host["cache_line_bytes"] // 8,
                      "privatized": host["cache_line_bytes"] // 8}[variant]
            for trial in trials:
                require(trial["iterations_per_worker"] == k and trial["counters"] == [k] * p, "M2 individual counter mismatch")
                require(trial["expected_total"] == p*k == trial["total_count"] == sum(trial["counters"]), "M2 total count mismatch")
                require(trial["stride_int64"] == stride and trial["spacing_bytes"] == stride*8, "M2 spacing mismatch")
                telemetry = trial["worker_timings"]
                require(len(telemetry) == p and {r["index"] for r in telemetry} == {i*stride for i in range(p)}, "M2 worker index coverage mismatch")
                require(all(r["count"] == k and r["alignment_remainder"] == 0 and r["end_ns"] > r["start_ns"]
                            and r["duration_ns"] == r["end_ns"] - r["start_ns"] for r in telemetry), "M2 worker timing/count/alignment mismatch")
                span = max(r["end_ns"] for r in telemetry) - min(r["start_ns"] for r in telemetry)
                require(trial["elapsed_ns"] == span, "M2 elapsed span mismatch")
                close(trial["elapsed_s"], span / 1e9, "M2 seconds conversion")
            medians[p, variant] = statistics.median(r["elapsed_s"] for r in trials)
    for row in rows:
        p = int(row["p"])
        require(int(row["iterations_per_worker"]) == k and int(row["total_work"]) == p*k, "M2 TSV workload mismatch")
        for variant in variants:
            t = medians[p, variant]
            s = medians[1, variant] / t
            close(row[f"t_{variant}_s"], t, "M2 median")
            close(row[f"s_{variant}"], s, "M2 PDF speedup")
            close(row[f"work_normalized_speedup_{variant}"], p*s, "M2 work-normalized speedup")
            if p == 1:
                require(row[f"e_{variant}"] == row[f"work_normalized_e_{variant}"] == "", "M2 e(1) must be undefined")
            else:
                close(row[f"e_{variant}"], (1/s - 1/p)/(1 - 1/p), "M2 PDF Karp-Flatt")
                close(row[f"work_normalized_e_{variant}"], (1/(p*s) - 1/p)/(1 - 1/p), "M2 normalized Karp-Flatt")
    return {"status": "PASS", "raw_trials_checked": 48, "increments_per_worker": k, "all_individual_and_total_counts_exact": True,
            "timing_medians_and_metrics_recomputed": True, "metric_limit": "P*K grows with P; PDF e is not a fixed-work serial-fraction estimate"}


def independent_stencil(n, seed, steps):
    source = np.random.default_rng(seed + n).random((n, n), dtype=np.float64)
    source[0, :] = source[-1, :] = 0.0
    source[:, 0] = source[:, -1] = 0.0
    # Separate expression implementation; no import of the benchmark kernel.
    for _ in range(steps):
        target = np.zeros_like(source)
        target[1:-1, 1:-1] = ((((source[1:-1, 1:-1] + source[:-2, 1:-1])
                               + source[2:, 1:-1]) + source[1:-1, :-2])
                               + source[1:-1, 2:]) * 0.2
        source = target
    return source


def audit_scaling(params, host):
    raw, rows = read_json("scaling_trials.json"), read_tsv("scaling_metrics.tsv")
    n, sid, steps = params["N"], params["student_id"], int(raw["steps"])
    require(steps == 128 and raw["trials_per_configuration"] == 3, "M3 expected 128 steps and three trials")
    require(raw["salted_N"] == n and raw["student_id"] == sid, "M3 salting mismatch")
    counts = raw["worker_counts"]
    require(counts == [1, 2, 4, 8] and max(counts) <= host["physical_cores"], "M3 worker coverage mismatch")
    require(len(rows) == len(raw["records"]) == 2*len(counts), "M3 configuration count mismatch")
    row_map = {(r["mode"], int(r["p"])): r for r in rows}
    require(set(row_map) == {(mode, p) for mode in ("strong", "weak") for p in counts}, "M3 configuration coverage mismatch")
    require({(r["mode"], int(r["p"])) for r in raw["records"]} == set(row_map),
            "M3 raw record coverage mismatch or duplicated configuration")
    independent = independent_stencil(n, params["seed"], steps)
    independent_sha = salted_digest(independent, sid)
    saved_hashes = {}
    for name in ("reference", "parallel"):
        array = np.load(RESULTS / f"stencil_{name}.npy", allow_pickle=False)
        require(array.shape == (n, n) and array.dtype == np.float64 and bool(np.isfinite(array).all()), "Saved stencil shape/dtype/value mismatch")
        np.testing.assert_allclose(array, independent, atol=ATOL, rtol=0)
        require(bool((array[0, :] == 0).all() and (array[-1, :] == 0).all() and (array[:, 0] == 0).all() and (array[:, -1] == 0).all()), "Saved stencil boundary mismatch")
        saved_hashes[name] = salted_digest(array, sid)
        require(saved_hashes[name] == independent_sha, "Saved stencil digest differs from independent reconstruction")
    for record in raw["records"]:
        mode, p = record["mode"], record["p"]
        require(mode in ("strong", "weak") and p in counts, "Unknown M3 record")
        expected_n = n if mode == "strong" else math.ceil(n*math.sqrt(p))
        require(record["N"] == expected_n and record["steps"] == steps, "M3 dimension or steps mismatch")
        trials, row = record["trials"], row_map[mode, p]
        require(len(trials) == 3 and {r["trial"] for r in trials} == {1, 2, 3}, "M3 raw trial coverage mismatch")
        positive([r["compute_time_s"] for r in trials] + [r["setup_spawn_time_s"] for r in trials], "M3 nonpositive duration")
        ref_sha = record["reference_salted_rounded6_sha256"]
        require(len(ref_sha) == 64 and all(c in "0123456789abcdef" for c in ref_sha), "M3 invalid digest")
        for trial in trials:
            require(0 <= trial["max_abs_error"] <= ATOL and isinstance(trial["bitwise_equal"], bool), "M3 recorded numerical comparison failed")
            require(trial["salted_rounded6_sha256"] == ref_sha, "M3 trial/reference digest mismatch")
            require(math.isfinite(trial["checksum_sum_float64"]), "M3 invalid reduction checksum")
            if expected_n == n:
                require(ref_sha == independent_sha, "M3 base record differs from independent stencil digest")
                close(trial["checksum_sum_float64"], float(np.sum(independent, dtype=np.float64)), "M3 base reduction checksum")
        require(int(row["N"]) == expected_n and int(row["steps"]) == steps, "M3 TSV dimension mismatch")
        work = (expected_n - 2)**2 * steps
        require(int(row["work_updates"]) == work, "M3 work-count mismatch")
        duration = statistics.median(r["compute_time_s"] for r in trials)
        close(row["time_s"], duration, "M3 compute median")
        close(row["setup_time_s"], statistics.median(r["setup_spawn_time_s"] for r in trials), "M3 setup median")
        close(row["gflops_s"], 5*work/duration/1e9, "M3 GFLOP/s")
        close(row["updates_per_s"], work/duration, "M3 update throughput")
    t1, w1 = float(row_map["strong", 1]["time_s"]), int(row_map["strong", 1]["work_updates"])
    xs = [(1-1/p, float(row_map["strong", p]["time_s"])/t1-1/p) for p in counts if p > 1]
    unconstrained = sum(x*y for x, y in xs)/sum(x*x for x, _ in xs)
    serial_fraction = min(1.0, max(0.0, unconstrained))
    close(raw["amdahl_fit"]["serial_fraction_unconstrained"], unconstrained, "Amdahl unconstrained fit")
    close(raw["amdahl_fit"]["serial_fraction"], serial_fraction, "Amdahl constrained fit")
    for mode in ("strong", "weak"):
        own_t1 = float(row_map[mode, 1]["time_s"])
        own_w1 = int(row_map[mode, 1]["work_updates"])
        close(raw["mode_baselines"][mode]["time_s"], own_t1, "Mode-specific raw baseline time")
        require(raw["mode_baselines"][mode]["work_updates"] == own_w1, "Mode-specific raw baseline work")
    for (mode, p), row in row_map.items():
        duration, work = float(row["time_s"]), int(row["work_updates"])
        own_t1 = float(row_map[mode, 1]["time_s"])
        own_w1 = int(row_map[mode, 1]["work_updates"])
        close(row["baseline_time_s"], own_t1, "Mode-specific TSV baseline time")
        require(int(row["baseline_work_updates"]) == own_w1, "Mode-specific TSV baseline work")
        throughput = (work/own_w1)*own_t1/duration
        close(row["throughput_speedup"], throughput, "M3 throughput speedup")
        close(row["efficiency_pct"], throughput/p*100, "M3 efficiency")
        close(row["gustafson_model_speedup"], p-serial_fraction*(p-1), "Gustafson model")
        if mode == "strong":
            speedup = t1/duration
            close(row["speedup"], speedup, "Strong speedup")
            if p > 1:
                close(row["amdahl_serial_fraction_per_p"], (1/speedup-1/p)/(1-1/p), "Strong per-P serial fraction")
        else:
            close(row["weak_efficiency_pdf_formula"], p*own_w1*own_t1/(work*duration), "PDF weak formula")
    return {"status": "PASS", "saved_arrays_independently_reconstructed": ["stencil_reference.npy", "stencil_parallel.npy"],
            "N": n, "steps": steps, "atol": ATOL, "rtol": 0, "saved_salted_rounded6_sha256": saved_hashes,
            "raw_trial_records_checked": sum(len(r["trials"]) for r in raw["records"]),
            "scope": "Base strong and weak-P1 outputs independently reconstructed. For larger weak grids, this audit checks all recorded per-trial reference digests, error tolerances and timing metrics; the full weak arrays were checked during execution and are not persisted."}


def audit_roofline():
    roof = read_json("roofline_summary.json")
    require("bound" in roof["regime"], "Roofline regime absent")
    close(roof["arithmetic_intensity"], 5/16, "Roofline ideal arithmetic intensity")
    require(roof["flops_per_update"] == 5 and roof["ideal_dram_bytes_per_update"] == 16, "Roofline operation/byte counts")
    require(roof["actual_numpy_ufunc_logical_bytes_per_update"] == 112, "Roofline ufunc logical bytes")
    probe = roof["copy_bandwidth_probe"]
    require(len(probe["times_s"]) == 5, "Copy probe trial count")
    positive(probe["times_s"], "Copy probe nonpositive timing")
    median = statistics.median(probe["times_s"])
    close(probe["median_time_s"], median, "Copy probe median")
    close(probe["observed_copy_bandwidth_gb_s"], probe["logical_bytes_per_copy"]/median/1e9, "Copy logical bandwidth")
    return {"status": "PASS", "classification": roof["regime"], "limitation": "Low-AI model hypothesis; not a hardware-counter measurement of DRAM saturation"}


def run_provided_verifier(host):
    # Remove stale local/previous receipts before invoking the original program.
    (RESULTS / "verification_receipt.sig").unlink(missing_ok=True)
    completed = subprocess.run([sys.executable, str(ROOT / "verify_outputs.py")], cwd=ROOT,
                               text=True, capture_output=True, timeout=120)
    output = completed.stdout + completed.stderr
    (RESULTS / "provided_verifier.log").write_text(output, encoding="utf-8")
    status = "PASS" if completed.returncode == 0 else "FAIL"
    expected = completed.returncode != 0 and host["cache_line_bytes"] != 64 and f"Expected cache_line_bytes == 64, got {host['cache_line_bytes']}" in output
    if completed.returncode == 0:
        require((RESULTS / "verification_receipt.sig").is_file(), "Supplied verifier reported success without its receipt")
        (RESULTS / "verification_receipt.sig").replace(RESULTS / "provided_verification_receipt.sig")
    elif not expected:
        raise RuntimeError(f"Unexpected supplied-verifier failure; see results/provided_verifier.log (exit {completed.returncode})")
    print(f"[PROVIDED VERIFIER {status}] exit={completed.returncode}; log=results/provided_verifier.log", flush=True)
    return {"status": status, "returncode": completed.returncode, "unchanged_script_sha256": ORIGINAL_VERIFIER_SHA256,
            "expected_cache_line_portability_failure": expected,
            "reason": f"Original verifier hard-codes 64-byte cache lines; actual host reports {host['cache_line_bytes']} bytes" if expected else "Original verifier completed successfully",
            "execution_scope": "Stops at M0 cache-line check; later supplied checks were not executed" if expected else "All supplied checks executed",
            "log": "results/provided_verifier.log"}


def write_manifest_and_receipt(params, provided):
    paths = [ROOT / "verify_outputs.py"]
    if (ROOT / "run_all.py").is_file():
        paths.append(ROOT / "run_all.py")
    paths += [p for p in (ROOT / "src").iterdir() if p.is_file() and p.suffix in (".py", ".c")]
    stable_results = (
        "host_profile.json", "salted_params.json", "sysctl_raw.txt",
        "stride_results.tsv", "stride_trials.json", "loop_interchange.json",
        "dense_naive.npy", "dense_interchanged.npy", "dense_reference.npy",
        "false_sharing_karp_flatt.tsv", "false_sharing_trials.json",
        "scaling_metrics.tsv", "scaling_trials.json", "roofline_summary.json",
        "stencil_reference.npy", "stencil_parallel.npy",
    )
    paths += [RESULTS / name for name in stable_results]
    entries = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": file_digest(p)}
               for p in sorted(set(paths))]
    write_json("verification_manifest.json", {"type": "LOCAL_FILE_INTEGRITY_MANIFEST", "hash": "SHA-256",
               "scope": "Source .py/.c, unchanged supplied verifier, run_all.py and an explicit list of stable scientific result JSON/TSV/NPY plus sysctl capture. README, all logs, numerical audit report, this manifest and the local receipt are excluded to avoid changing or circular hashes.",
               "files": entries, "authentication": "Unsigned integrity digests; no instructor or third-party authentication"})
    manifest_sha = file_digest(RESULTS / "verification_manifest.json")
    text = ("ISSUER=src/verify_numerics.py\nTYPE=LOCAL_NUMERICAL_AUDIT\n"
            f"STUDENT_ID={params['student_id']}\nSALTED_N={params['N']}\nLOCAL_NUMERICAL_STATUS=PASS\n"
            f"PROVIDED_VERIFIER_STATUS={provided['status']}\nMANIFEST_SHA256={manifest_sha}\n"
            "AUTHENTICATION=NONE_UNSIGNED_DETERMINISTIC_INTEGRITY_DIGEST\n"
            "NOTE=Local audit receipt; not instructor-issued and not a claim of supplied-verifier success.\n")
    (RESULTS / "verification_receipt.sig").write_text(text, encoding="utf-8")
    return manifest_sha


def main():
    for name in ("verification_receipt.sig", "verification_manifest.json", "numerical_verification.json"):
        (RESULTS / name).unlink(missing_ok=True)
    params, host = audit_environment()
    checks = {}
    for name, audit in (("stride", lambda: audit_stride(params)), ("dense", lambda: audit_dense(params)),
                        ("false_sharing", lambda: audit_false_sharing(host)),
                        ("scaling", lambda: audit_scaling(params, host)), ("roofline", audit_roofline)):
        checks[name] = audit()
        print(f"[LOCAL PASS] {name}", flush=True)
    provided = run_provided_verifier(host)
    report = {"issuer": "src/verify_numerics.py", "type": "LOCAL_NUMERICAL_AUDIT", "local_numerical_status": "PASS",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "student_id": params["student_id"], "salted_N": params["N"],
              "provided_verifier": provided, "checks": checks,
              "receipt_note": "Local receipt is unsigned; it does not override the provided verifier failure or authenticate who authored/performed the work."}
    write_json("numerical_verification.json", report)
    manifest_sha = write_manifest_and_receipt(params, provided)
    print(f"[LOCAL NUMERICAL AUDIT PASS] Manifest SHA-256: {manifest_sha}", flush=True)
    print(f"Provided verifier status remains {provided['status']}; see results/provided_verifier.log.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (RESULTS / "verification_receipt.sig").unlink(missing_ok=True)
        (RESULTS / "numerical_verification.json").unlink(missing_ok=True)
        (RESULTS / "verification_manifest.json").unlink(missing_ok=True)
        print(f"[LOCAL AUDIT FAIL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
