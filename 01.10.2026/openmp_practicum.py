"""OpenMP Multi-Core Scaling in Python - 01.10.2026.

Student: Abyz Nuradil | Student ID: 230103188
Implements all three challenges from OpenMP_Python_Lab_Practicum.pdf.
Run with no arguments for the full assigned workloads, or --quick for a smoke run.
Timings exclude JIT compilation and use the median of three fresh repetitions.
The generated report records the actual benchmark hardware and environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import njit, prange

STUDENT_NAME = "Abyz Nuradil"
STUDENT_ID = "230103188"
LAB_DATE = "01.10.2026"


@njit(parallel=True, nogil=True)
def monte_carlo_pi(n_samples):
    """Challenge 1: thread-private counts are combined by a sum reduction."""
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


@njit(parallel=True, nogil=True)
def render_mandelbrot_rows(h, w, max_iter):
    """Challenge 2: parallel rows with contiguous writes along each row."""
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while z_real * z_real + z_imag * z_imag <= 4.0 and it < max_iter:
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img


@njit(parallel=True, nogil=True)
def render_mandelbrot_cols(h, w, max_iter):
    """Challenge 2: same pixels; parallel columns with strided inner writes."""
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while z_real * z_real + z_imag * z_imag <= 4.0 and it < max_iter:
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img


@njit(parallel=True, nogil=True)
def heat_step(u, u_next, alpha=0.20):
    """Challenge 3: write only the interior; boundaries remain fixed."""
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                - 4.0 * u[i, j]
            )


def initial_heat_grid(size, dtype):
    u = np.zeros((size, size), dtype=dtype)
    u[0, :] = 100.0
    u[:, 0] = 100.0
    return u, u.copy()


def time_heat(size, steps, dtype):
    # Reinitialize both arrays every repetition, after the separate JIT warmup.
    u, u_next = initial_heat_grid(size, dtype)
    alpha = dtype(0.20)
    start = time.perf_counter()
    for _ in range(steps):
        heat_step(u, u_next, alpha)
        u, u_next = u_next, u
    elapsed = time.perf_counter() - start
    return elapsed, u


def available_thread_counts():
    maximum = numba.config.NUMBA_NUM_THREADS
    return sorted({t for t in (1, 2, 4, 8, maximum) if t <= maximum})


def hardware_info(environment):
    cpu = platform.processor() or platform.machine()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    elif platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        )
        cpu = result.stdout.strip() or cpu
    quota_cores = None
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    if quota_path.exists():
        quota, period = quota_path.read_text().split()
        if quota != "max":
            quota_cores = int(quota) / int(period)
    return {
        "environment": environment,
        "cpu_model": cpu,
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "matplotlib_version": matplotlib.__version__,
        "logical_cpus_visible": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_quota_cores": quota_cores,
        "numba_max_threads": numba.config.NUMBA_NUM_THREADS,
        "threading_layer": numba.threading_layer(),
    }


def warmup():
    numba.set_num_threads(1)
    monte_carlo_pi(10_000)
    render_mandelbrot_rows(100, 100, 50)
    render_mandelbrot_cols(100, 100, 50)
    for dtype in (np.float64, np.float32):
        u, u_next = initial_heat_grid(32, dtype)
        heat_step(u, u_next, dtype(0.20))


def add_scaling(rows):
    baseline = next(row["seconds"] for row in rows if row["threads"] == 1)
    for row in rows:
        row["speedup"] = baseline / row["seconds"]
        row["efficiency_percent"] = 100.0 * row["speedup"] / row["threads"]


def benchmark(args):
    samples = 500_000 if args.quick else 120_000_000
    h = w = 180 if args.quick else 2500
    max_iter = 100 if args.quick else 1000
    size = 100 if args.quick else 1500
    steps = 12 if args.quick else 300
    counts = available_thread_counts()
    maximum = counts[-1]
    print("Compiling and warming up every kernel and precision...", flush=True)
    warmup()
    result = {
        "student": {"name": STUDENT_NAME, "id": STUDENT_ID, "lab_date": LAB_DATE},
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "quick smoke run" if args.quick else "full assigned workloads",
        "hardware": hardware_info(args.environment),
        "methodology": {
            "repetitions": args.repeats, "aggregation": "median",
            "jit_compilation_timed": False,
            "heat_initialization_timed": False,
            "heat_time_includes": "300 Python dispatches and pointer swaps in full mode",
            "mandelbrot_time_includes": "allocation and pixel computation",
            "randomness": "Numba thread-local random streams; estimates vary per run",
            "heat_throughput_formula": "grid_size^2 * steps / seconds / 1e6 (lab convention)",
        },
        "workloads": {"monte_carlo_samples": samples, "mandelbrot_height": h,
                      "mandelbrot_width": w, "mandelbrot_max_iter": max_iter,
                      "heat_grid_size": size, "heat_steps": steps},
        "monte_carlo": [], "mandelbrot": [], "heat": [], "validation": {},
    }
    print(json.dumps(result["hardware"], indent=2), flush=True)
    print("\nChallenge 1: Monte Carlo pi", flush=True)
    for threads in counts:
        numba.set_num_threads(threads)
        monte_carlo_pi(10_000)  # Also warm up the newly selected worker count.
        times, estimates = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            estimate = monte_carlo_pi(samples)
            times.append(time.perf_counter() - start)
            estimates.append(float(estimate))
        if not all(abs(estimate - math.pi) < (0.05 if args.quick else 0.005) for estimate in estimates):
            raise AssertionError("Monte Carlo estimate outside the expected range")
        row = {"threads": threads, "seconds": statistics.median(times),
               "times_seconds": times, "pi_estimates": estimates}
        result["monte_carlo"].append(row)
        print(f"  {threads:2d} threads: {row['seconds']:.6f} s; pi={estimates[-1]:.7f}", flush=True)
    add_scaling(result["monte_carlo"])

    print("\nChallenge 2: Mandelbrot", flush=True)
    numba.set_num_threads(maximum)
    reference = None
    for axis, render in (("rows", render_mandelbrot_rows), ("cols", render_mandelbrot_cols)):
        render(100, 100, 50)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            grid = render(h, w, max_iter)
            times.append(time.perf_counter() - start)
        if reference is None:
            reference = grid
        else:
            np.testing.assert_array_equal(reference, grid)
        row = {"axis": axis, "threads": maximum,
               "seconds": statistics.median(times), "times_seconds": times}
        result["mandelbrot"].append(row)
        print(f"  {axis:4s}: {row['seconds']:.6f} s", flush=True)
    result["validation"]["mandelbrot_rows_equal_cols"] = True
    save_mandelbrot(reference, result, args.output_dir)

    print("\nChallenge 3: heat diffusion", flush=True)
    final_states = {}
    for dtype in (np.float64, np.float32):
        precision_rows = []
        baseline_grid = None
        for threads in counts:
            numba.set_num_threads(threads)
            time_heat(32, 2, dtype)
            times = []
            for _ in range(args.repeats):
                elapsed, grid = time_heat(size, steps, dtype)
                times.append(elapsed)
            if baseline_grid is None:
                baseline_grid = grid.copy()
            else:
                np.testing.assert_array_equal(baseline_grid, grid)
            assert np.all(grid[0, :] == 100.0) and np.all(grid[:, 0] == 100.0)
            assert np.all(grid[-1, 1:] == 0.0) and np.all(grid[1:, -1] == 0.0)
            assert np.isfinite(grid).all() and grid.min() >= 0 and grid.max() <= 100
            elapsed = statistics.median(times)
            row = {"precision": np.dtype(dtype).name, "threads": threads,
                   "seconds": elapsed, "times_seconds": times,
                   "megacells_per_second": size * size * steps / elapsed / 1e6,
                   "two_array_megabytes": 2 * grid.nbytes / 1e6}
            precision_rows.append(row)
            print(f"  {row['precision']} {threads:2d} threads: {elapsed:.6f} s; "
                  f"{row['megacells_per_second']:.2f} Mcells/s", flush=True)
        add_scaling(precision_rows)
        result["heat"].extend(precision_rows)
        final_states[np.dtype(dtype).name] = grid
    difference = float(np.abs(final_states["float64"] - final_states["float32"]).max())
    np.testing.assert_allclose(final_states["float64"], final_states["float32"], rtol=2e-5, atol=2e-5)
    result["validation"].update({
        "monte_carlo_estimates_plausible": True,
        "heat_thread_counts_identical_within_precision": True,
        "heat_boundaries_preserved": True,
        "heat_finite_and_in_range": True,
        "float32_matches_float64_within_tolerance": True,
        "heat_max_absolute_precision_difference_celsius": difference,
    })
    numba.set_num_threads(maximum)
    return result


def save_mandelbrot(grid, result, output):
    rows = result["mandelbrot"][0]
    fig, ax = plt.subplots(figsize=(8, 8))
    # r=0 corresponds to cy=-1.2; origin='lower' preserves the coordinate mapping.
    ax.imshow(grid, cmap="magma", origin="lower", extent=(-2.0, 0.5, -1.2, 1.2))
    ax.set_title(f"Mandelbrot {grid.shape[0]}x{grid.shape[1]} (Rows: {rows['seconds']:.2f}s)")
    ax.set_axis_off()
    fig.savefig(output / "mandelbrot_output.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, heat=False):
    header = "| Threads | Time (s) | Speedup | Efficiency (%) |"
    divider = "| ---: | ---: | ---: | ---: |"
    if heat:
        header += " Throughput (Mcells/s) |"
        divider += " ---: |"
    lines = [header, divider]
    for row in rows:
        line = f"| {row['threads']} | {row['seconds']:.6f} | {row['speedup']:.3f}x | {row['efficiency_percent']:.2f} |"
        if heat:
            line += f" {row['megacells_per_second']:.2f} |"
        lines.append(line)
    return "\n".join(lines)


def build_report(result):
    hw, work = result["hardware"], result["workloads"]
    mc = result["monte_carlo"]
    maximum = hw["numba_max_threads"]
    base, last = mc[0], mc[-1]
    best = max(mc, key=lambda row: row["speedup"])
    rows, cols = result["mandelbrot"]
    faster = min(result["mandelbrot"], key=lambda row: row["seconds"])
    slower = max(result["mandelbrot"], key=lambda row: row["seconds"])
    heat64 = [row for row in result["heat"] if row["precision"] == "float64"]
    heat32 = [row for row in result["heat"] if row["precision"] == "float32"]
    end64, end32 = heat64[-1], heat32[-1]
    precision_factor = end64["seconds"] / end32["seconds"]
    precision_description = "decreased" if precision_factor >= 1 else "increased"
    quota = f"; container CPU quota: {hw['cpu_quota_cores']:g} core equivalents" if hw["cpu_quota_cores"] else ""
    efficiency_description = "decreased below" if last["efficiency_percent"] < 100 else "did not decrease below"
    scorecard = [
        "| Benchmark | Active threads | Size / samples | Time (s) | Speedup | Efficiency (%) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in mc:
        scorecard.append(f"| Challenge 1: Monte Carlo | {row['threads']} | {work['monte_carlo_samples']:,} | {row['seconds']:.6f} | {row['speedup']:.3f}x | {row['efficiency_percent']:.2f} |")
    for row in result["mandelbrot"]:
        scorecard.append(f"| Challenge 2: Mandelbrot ({row['axis']}) | {maximum} | {work['mandelbrot_height']} x {work['mandelbrot_width']} | {row['seconds']:.6f} | N/A | N/A |")
    for row in (end64, end32):
        scorecard.append(f"| Challenge 3: Heat ({row['precision']}) | {maximum} | {work['heat_grid_size']} x {work['heat_grid_size']} x {work['heat_steps']} | {row['seconds']:.6f} | N/A | N/A |")
    return f"""# OpenMP Multi-Core Scaling in Python - completed practicum

**Student:** {STUDENT_NAME}  
**Student ID:** {STUDENT_ID}  
**Lab date:** {LAB_DATE}  
**Section:** Not supplied in the assignment request.  
**Benchmark environment:** {hw['environment']}  
**Actual run timestamp (UTC):** {result['run_timestamp_utc']}  
**Mode:** {result['mode']}

## Hardware and measurement method

CPU: **{hw['cpu_model']}**. Visible logical CPUs: {hw['logical_cpus_visible']}; Numba maximum: **{maximum} threads**{quota}. Backend: **{hw['threading_layer']}**. OS: {hw['operating_system']}. Python {hw['python_version']}; NumPy {hw['numpy_version']}; Numba {hw['numba_version']}; Matplotlib {hw['matplotlib_version']}.

Every result is the median of {result['methodology']['repetitions']} measured repetitions. Each compiled kernel and both heat precisions are warmed up before timing. Heat grids are reset outside each timed region, so every run starts from identical initial conditions. Heat timing includes the Python timestep loop and pointer swaps; Mandelbrot timing includes image allocation. Raw repetition timings and all Monte Carlo estimates are preserved in `benchmark_results.json`.

These measurements describe the environment named above. They do not measure a different laptop. No claim of measured peak RAM bandwidth, SMT effects, or physical-core topology is made from these timings alone.

## Table 1 - Student result scorecard

{chr(10).join(scorecard)}

## Challenge 1 - answers A-D

**A. Baseline and maximum-thread runtime.** T_1 = **{base['seconds']:.6f} s**. T_max = **{last['seconds']:.6f} s** at {maximum} threads. Maximum-thread speedup is **{last['speedup']:.3f}x**, with **{last['efficiency_percent']:.2f}%** efficiency. The best observed speedup is **{best['speedup']:.3f}x** at {best['threads']} threads. T_max means runtime at the maximum worker count, not the largest elapsed time.

**B. Why warmup is required.** The first invocation compiles Python into native code, specializes the argument types, and initializes parallel runtime resources. Including that work would compare compilation/startup cost with already compiled executions. The 10,000-sample warmup excludes JIT compilation from the measurements.

**C. Parallel efficiency and Amdahl's Law.** Efficiency {efficiency_description} 100% at the maximum worker count: **{last['efficiency_percent']:.2f}%**. Speedup S = T_1 / T_p and efficiency E = 100 S / p. Amdahl's Law is S(p) = 1 / (s + (1-s)/p), where s is the serial fraction. Worker coordination, reductions and serial launch overhead limit scaling. CPU quotas and shared execution resources also matter here. On other hardware, memory/cache contention, thermal/frequency changes, and SMT threads sharing physical cores can contribute. Monte Carlo is largely compute-bound, so memory bandwidth is not automatically the dominant explanation. Timing alone does not identify which architectural factor caused a particular loss of efficiency.

**D. Why the count is safe.** Numba recognizes `inside_circle += 1` as an integer sum reduction. Each worker accumulates a private partial count, and those counts are combined after the parallel loop. With OpenMP selected, this has the semantics of `reduction(+:inside_circle)`; workers do not race on an ordinary shared increment. Numba can lower this to private accumulators and a combine phase rather than literal generated C source.

## Challenge 2 - answers A-D

**A. Measured comparison.** Row-parallel time = **{rows['seconds']:.6f} s**; column-parallel time = **{cols['seconds']:.6f} s**. The **{faster['axis']}** decomposition was **{slower['seconds']/faster['seconds']:.3f}x** faster in this run. Both output arrays are exactly equal. Changing the decomposition changes both memory access and how expensive pixels are distributed. Contiguous row writes generally favor locality, but workload imbalance can offset that advantage. This run does not separately instrument cache misses or per-thread idle time, so the timing difference is consistent with those factors rather than proof of a single cause.

**B. C-contiguous layout.** A C-order int32 array stores adjacent columns in the same row at consecutive 4-byte addresses. The row kernel varies c inside the inner loop, writing `img[r, c]` contiguously. The column kernel varies r inside the inner loop, also writing `img[r, c]`, but consecutive writes are separated by w * 4 bytes. At width {work['mandelbrot_width']}, that stride is {work['mandelbrot_width'] * 4:,} bytes. Strided writes use cache lines less effectively and can increase cache/TLB pressure. The PDF's `img[c, r]` example illustrates swapped indexing; our actual kernels retain the same coordinate mapping and array indexing.

**C. Load imbalance and dynamic scheduling.** An outer band may escape after a few iterations, while central points reach max_iter. Equal row counts therefore need not mean equal computation. With static contiguous chunks, a worker assigned cheap rows can finish while a worker assigned central rows continues. OpenMP `schedule(dynamic, chunk)` assigns another chunk to each available worker, reducing idle time at the cost of scheduling overhead. A row/column swap by itself does not enable dynamic scheduling. Numba's default `prange` scheduling is static; its positive chunk-size dynamic scheduling is supported by the TBB backend, so changing the chunk size with an OpenMP backend would not demonstrate dynamic scheduling. The recorded backend is **{hw['threading_layer']}**, and these kernels retain default scheduling. See the [official Numba scheduling documentation](https://numba.readthedocs.io/en/stable/user/parallel.html#scheduling).

**D. Visual artifact.** `mandelbrot_output.png` is exported at 300 DPI and accompanies this report and the completed task sheet. The lower image origin matches the coordinates used by the kernel.

![Mandelbrot output](mandelbrot_output.png)

## Challenge 3 - answers A-C

**A. Runtime and throughput.** At {maximum} threads, float64 ran in **{end64['seconds']:.6f} s**, producing **{end64['megacells_per_second']:.2f} Megacells/s**. Throughput follows the lab formula grid_size^2 * steps / seconds / 1e6. Only (grid_size-2)^2 interior cells are updated at each step; using the full grid is the worksheet's reporting convention.

**B. Float32 comparison.** At the same {maximum} threads, float32 ran in **{end32['seconds']:.6f} s**, producing **{end32['megacells_per_second']:.2f} Megacells/s**. The runtime ratio T_float64 / T_float32 = **{precision_factor:.3f}**; runtime {precision_description} in this measurement. The two buffers occupy **{end64['two_array_megabytes']:.2f} MB** in float64 and **{end32['two_array_megabytes']:.2f} MB** in float32. Halving stored element size can reduce data movement and fit more cells in cache. A 2x speedup is not guaranteed because cache reuse, arithmetic, dispatch/synchronization, and CPU quotas also affect total time. The maximum final-state precision difference is **{result['validation']['heat_max_absolute_precision_difference_celsius']:.8f} C**, within the checked tolerance. alpha=0.20 satisfies the explicit 2D stability limit alpha <= 0.25. Top and left boundaries remain 100 C; bottom and right remain 0 C except their intersections with the heated walls.

**C. Memory roofline.** The roofline bound is P <= min(P_peak, B * I), where B is memory bandwidth and I is arithmetic intensity (operations per byte moved). This stencil performs relatively little arithmetic for its array traffic. Once the relevant memory bandwidth is saturated, doubling cores cannot double available bytes per second. Cache reuse can shift the relevant bottleneck toward cache bandwidth, and repeated parallel launches add overhead. The scaling tables below show observed runtimes; identifying RAM saturation specifically would require bandwidth or hardware-counter measurements.

### Float64 scaling

{markdown_table(heat64, heat=True)}

### Float32 scaling

{markdown_table(heat32, heat=True)}

## Validation and leaderboard data

Full-resolution row and column fractals match exactly. Heat grids match exactly across thread counts within each precision; heated and cold boundaries stay fixed, and temperatures remain finite and between 0 and 100 C. Float32 and float64 states agree within the configured tolerance. Monte Carlo estimates are checked against pi using a stochastic tolerance.

Leaderboard entry for this benchmark environment: **{hw['cpu_model']}**, Monte Carlo best speedup **{best['speedup']:.3f}x** ({best['threads']} threads). Best float64 heat speedup: **{max(heat64, key=lambda r: r['speedup'])['speedup']:.3f}x**. Record laptop results separately after running on that laptop.
"""


def save_results(result, output):
    (output / "benchmark_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "benchmark_report.md").write_text(build_report(result), encoding="utf-8")
    with (output / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["challenge", "variant", "threads", "seconds", "speedup", "efficiency_percent", "megacells_per_second"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for challenge, rows in ((1, result["monte_carlo"]), (2, result["mandelbrot"]), (3, result["heat"])):
            for row in rows:
                writer.writerow({"challenge": challenge,
                                 "variant": row.get("axis", row.get("precision", "pi")),
                                 **{key: row.get(key, "") for key in fields[2:]}})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use small workloads; these are not final lab results.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timing repetitions (default: 3).")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--environment", default="Local machine running this script", help="Describe where the measurements were made.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.output_dir = args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = benchmark(args)
    save_results(result, args.output_dir)
    print("\nAll numerical checks PASSED.", flush=True)
    print(f"Saved report, CSV, JSON, and Mandelbrot image to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
