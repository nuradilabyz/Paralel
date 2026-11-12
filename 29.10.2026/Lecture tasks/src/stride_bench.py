#!/usr/bin/env python3
"""M1: measured scalar reductions over a 128 MiB float64 buffer.

The actual loops are in the supplied C source, exposed using stdlib ctypes,
so the memory experiment is not dominated by Python element dispatch.
"""
import ctypes
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import numpy as np

from common import ROOT, RESULTS, load_params, write_json

DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


def native_library():
    """Compile transparently; source/flags determine the cache key."""
    source = Path(__file__).with_name("m1_kernels.c")
    compiler = shutil.which(os.environ.get("CC", "clang")) or shutil.which("cc")
    if compiler is None:
        raise RuntimeError("A C compiler is required for the explicit memory kernels")
    flags = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off"]
    flags += ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
    key = hashlib.sha256(source.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    build_dir = ROOT / ".build"
    build_dir.mkdir(exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    target = build_dir / ("m1_" + key + suffix)
    command = [compiler, *flags, str(source), "-o", str(target)]
    if not target.exists():
        subprocess.run(command, check=True)
    compiler_version = subprocess.run([compiler, "--version"], text=True, capture_output=True, check=True).stdout.splitlines()[0]
    lib = ctypes.CDLL(str(target))
    lib.stride_sum.argtypes = [DOUBLE_PTR, ctypes.c_size_t, ctypes.c_size_t]
    lib.stride_sum.restype = ctypes.c_double
    for name in ("dense_ijk", "dense_ikj"):
        fn = getattr(lib, name)
        fn.argtypes = [DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR, ctypes.c_size_t, ctypes.c_size_t]
        fn.restype = None
    metadata = {
        "compiler": compiler_version,
        "compiler_flags": flags,
        "source": "src/m1_kernels.c",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "implementation": "explicit C loops called through Python ctypes; no BLAS",
    }
    return lib, metadata


def pointer(array):
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise ValueError("Kernel input must be a contiguous float64 buffer")
    return array.ctypes.data_as(DOUBLE_PTR)


def main():
    params = load_params()
    elements = int(params["buffer_elements"])
    data = np.ones(elements, dtype=np.float64)
    ptr = pointer(data)
    lib, metadata = native_library()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    all_trials = []
    for stride in (1, 2, 4, 8, 16, 32, 64, 128):
        touched = (elements + stride - 1) // stride
        expected = float(touched)
        for _ in range(5):
            if lib.stride_sum(ptr, elements, stride) != expected:
                raise AssertionError("Stride reduction produced an incorrect sum")
        samples = []
        for trial in range(5):
            begin = time.perf_counter_ns()
            total = lib.stride_sum(ptr, elements, stride)
            elapsed = time.perf_counter_ns() - begin
            if total != expected:
                raise AssertionError("Stride reduction produced an incorrect sum")
            samples.append(elapsed)
            all_trials.append({"stride": stride, "trial": trial + 1,
                               "elapsed_ns": elapsed, "elements_touched": touched,
                               "sum": total})
        median = statistics.median(samples)
        rows.append({"stride": stride, "elements_touched": touched,
                     "median_time_ns": median, "latency_ns": median / touched,
                     "effective_read_GB_s": touched * 8 / median,
                     "sum": expected, "trials_ns": json.dumps(samples)})
        print(f"stride={stride:3d}: {median/touched:.3f} ns/element, sum={expected:.0f}", flush=True)
    with (RESULTS / "stride_results.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_json("stride_trials.json", {
        **metadata, "buffer_elements": elements, "buffer_bytes": data.nbytes,
        "warmup_iterations_per_stride": 5, "recorded_iterations_per_stride": 5,
        "timer": "time.perf_counter_ns", "cache_flush": False,
        "note": "Repeated streaming passes; no fabricated cache-hit or hardware-counter values. Actual cache line comes from host_profile.json.",
        "trials": all_trials,
    })


if __name__ == "__main__":
    main()
