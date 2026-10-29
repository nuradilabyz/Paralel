#!/usr/bin/env python3
"""M1: explicit i-j-k versus i-k-j matrix multiplication, with real checks."""
import os

# Avoid a threaded BLAS reference competing with the benchmark processes.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import statistics
import time

import numpy as np

from common import RESULTS, load_params, write_json, digest_array
from stride_bench import native_library, pointer


def main():
    params = load_params()
    n, seed = int(params["N"]), int(params["seed"])
    m = n // 2
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)
    lib, metadata = native_library()
    pa, pb = pointer(a), pointer(b)
    outputs = {name: np.zeros((m, m), dtype=np.float64) for name in ("naive", "interchanged")}
    timings = {name: [] for name in outputs}
    functions = {"naive": lib.dense_ijk, "interchanged": lib.dense_ikj}
    # Allocate/zero the output outside the timing for both methods. Alternate
    # order to reduce a systematic thermal or first-touch bias across trials.
    for trial in range(3):
        order = ("naive", "interchanged") if trial % 2 == 0 else ("interchanged", "naive")
        for name in order:
            out = outputs[name]
            out.fill(0.0)
            pc = pointer(out)
            begin = time.perf_counter_ns()
            functions[name](pa, pb, pc, n, m)
            elapsed_ns = time.perf_counter_ns() - begin
            timings[name].append(elapsed_ns / 1e9)
            print(f"{name} trial {trial + 1}: {elapsed_ns/1e9:.6f} s", flush=True)
    reference = a[:m, :m] @ b[:m, :m]
    atol, rtol = 1e-12, 0.0
    naive, interchanged = outputs["naive"], outputs["interchanged"]
    np.testing.assert_allclose(interchanged, naive, atol=atol, rtol=rtol)
    np.testing.assert_allclose(naive, reference, atol=atol, rtol=rtol)
    np.testing.assert_allclose(interchanged, reference, atol=atol, rtol=rtol)
    RESULTS.mkdir(exist_ok=True)
    for name, out in {**outputs, "reference": reference}.items():
        np.save(RESULTS / f"dense_{name}.npy", out, allow_pickle=False)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    hashes = {name: digest_array(out, params["student_id"]) for name, out in {**outputs, "reference": reference}.items()}
    result = {
        **metadata, "N": n, "M": m, "seed": seed,
        "allocation": "full N-by-N A and B; leading M-by-M sub-block product; original row pitch N",
        "naive_time_s": medians["naive"],
        "interchanged_time_s": medians["interchanged"],
        "speedup": medians["naive"] / medians["interchanged"],
        "trials_s": timings, "recorded_iterations": 3,
        "timer": "time.perf_counter_ns", "timing_excludes": ["allocation", "output zero fill", "NumPy reference", "verification", "file output"],
        "absolute_tolerance": atol, "relative_tolerance": rtol,
        "max_abs_error_interchanged_vs_naive": float(np.max(np.abs(interchanged - naive))),
        "max_abs_error_naive_vs_reference": float(np.max(np.abs(naive - reference))),
        "max_abs_error_interchanged_vs_reference": float(np.max(np.abs(interchanged - reference))),
        "numerical_checks_passed": True,
        "rounded6_salted_sha256": hashes,
        "output_files": {name: f"results/dense_{name}.npy" for name in ("naive", "interchanged", "reference")},
        "note": "Floating-point comparison is authoritative; a rounded digest may differ if a value lies at a six-decimal rounding boundary.",
    }
    write_json("loop_interchange.json", result)
    print(f"Loop interchange speedup: {result['speedup']:.3f}x; max error={result['max_abs_error_interchanged_vs_naive']:.3g}", flush=True)


if __name__ == "__main__":
    main()
