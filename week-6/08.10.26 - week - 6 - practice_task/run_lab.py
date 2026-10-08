"""Execute every required size and generate the report from measured results."""
from pathlib import Path
import contextlib, io, json, re, subprocess, sys
from datetime import datetime, timezone
import numpy as np
from numba import cuda
import task1_divergence as t1
import task2_stencil_1d as t2
import task3_grid_stride as t3
import task4_sobel_2d as t4

STUDENT_ID = "230103188"
assert cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU"
assert not cuda.config.ENABLE_CUDASIM, "Real CUDA GPU required"
device = cuda.get_current_device()
name = device.name.decode() if isinstance(device.name, bytes) else device.name
cc = ".".join(map(str, device.compute_capability))
telemetry = subprocess.check_output(["nvidia-smi"], text=True)
print(telemetry)
Path("nvidia-smi.txt").write_text(telemetry)
print(f"GPU: {name}; compute capability: {cc}")
bench = t1.benchmark()
Path("benchmark_results.json").write_text(json.dumps(bench, indent=2)+"\n")
checks = []
for task in ["task2_stencil_1d.py", "task3_grid_stride.py", "task4_sobel_2d.py"]:
    result = subprocess.run([sys.executable, task], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    checks.append(result.stdout.strip())
# Additional cases verify halos, non-multiple grids, arbitrary factors and Sobel sign.
for n in (1, 2, 255, 257, 10007):
    a = np.random.default_rng(n).random(n, dtype=np.float32)
    assert np.allclose(t2.run_stencil(a), t2.cpu_stencil(a), atol=1e-4)
for n in (1, 16383, 16385, 100000):
    a = np.linspace(-2, 3, n, dtype=np.float32)
    assert np.allclose(t3.run_grid_stride(a, -1.75), a*np.float32(-1.75), atol=1e-5)
for shape in ((1, 1), (2, 7), (17, 31), (33, 19)):
    img = np.random.default_rng(7).random(shape, dtype=np.float32)
    assert np.allclose(t4.run_sobel(img), t4.cpu_sobel(img), atol=1e-4)
ramp = np.tile(np.arange(31, dtype=np.float32), (17, 1))
assert np.all(t4.run_sobel(ramp)[1:-1, 1:-1] == 8)
print("ADDITIONAL BOUNDARY AND NONUNIFORM-INPUT CHECKS PASSED")
verified = subprocess.run([sys.executable, "verify_submission.py"], input=STUDENT_ID+"\n", capture_output=True, text=True)
print(verified.stdout)
assert verified.returncode == 0, verified.stderr
Path("verification_output.txt").write_text(verified.stdout)
token = re.search(r"OFFICIAL SUBMISSION TOKEN: ([A-F0-9]{20})", verified.stdout).group(1)
rows = "\n".join(f"| {key} | {val['mean_ms']:.6f} |" for key, val in bench.items())
ratio = bench["B_interleaved"]["mean_ms"]/bench["C_warp_aligned"]["mean_ms"]
readme = f"""# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** {STUDENT_ID}  
**Allocated GPU Node:** {name}  
**CUDA Compute Capability:** {cc}  
**Official Verification Token:** {token}

## Execution

Executed on Google Colab, UTC: {datetime.now(timezone.utc).isoformat()}.
The source assignment is `CUDA Python Lab 01.pdf` (its internal title is Lab 02).
Files are placed in the requested `Paralel/week-6/08.10.26 - week - 6 - practice_task` folder.
The PDF separately specifies a repository named `cuda-lab-02-230103188`; this submission uses the folder requested by the student.

Open `CUDA_Lab02_230103188.ipynb` in Colab, choose **Runtime > Change runtime type > T4 GPU**, then run all cells.
The notebook writes the Python modules, runs the full-size tasks and instructor verifier, and regenerates this report.
For standalone execution in a CUDA environment: `python run_lab.py`.
For the instructor check alone: `python verify_submission.py`, then enter `{STUDENT_ID}`.

## Task 1: Warp divergence benchmark

N = 1,048,576 float32 values; 1,000 iterations per value; 256 threads per block.
Each kernel has one untimed warm-up and 10 timed trials. Timers use `time.perf_counter()` with `cuda.synchronize()`.
Device buffers are allocated before timing; transfers and validation occur outside timing.
Wall-clock measurements include launch and synchronization overhead, as specified by the assignment.
Every trial reads the same input and writes a separate output.

| Kernel | Mean time (ms), 10 trials |
| --- | ---: |
{rows}

Interleaved / warp-aligned time ratio: **{ratio:.3f}x**.
A uses `v = v * 1.0001 + 0.0001`. B and C use that same path or `v = (v - 0.0001) / 1.0001`.
B assigns opposite paths to adjacent lanes; C assigns one path to each complete 32-thread warp.
B and C process equal numbers of values on each path, so their comparison isolates branch arrangement more directly than A versus B.
Divergence can serialize paths, but the measured slowdown also depends on compiler transformations, arithmetic cost and scheduling; an exact 2x penalty is not guaranteed.
All three outputs were checked against CPU arithmetic; small fused-operation rounding differences are allowed.
Raw trial times and maximum validation errors are in `benchmark_results.json`.

## Tasks 2-4: Results

{chr(10).join('- '+c for c in checks)}

- Stencil: replicated left/right boundary, 256 threads per block, CPU reference uses edge padding.
- Grid-stride: exactly 64 blocks x 256 threads = 16,384 threads; each thread processes 1,024 values at N = 2**24.
- Sobel-X: 16x16 threads per block; dynamic grid is 128x128 blocks for 2048x2048 input. All four output borders are zero.
- Sobel follows the signed formula in the PDF (cross-correlation convention), with a positive derivative along increasing columns.
- Extra GPU checks passed for singleton/two-element arrays, odd dimensions, partial blocks, negative scaling and a column ramp with expected gradient +8.

## Official verification

The provided verifier is preserved, with PDF line wrapping and indentation repaired.
Its smaller smoke-test sizes are run in addition to the full-size task requirements.

```text
{verified.stdout.strip()}
```

## Environment

See `nvidia-smi.txt` and `environment.txt` for runtime telemetry and package versions.
Numba-CUDA installation reference: https://nvidia.github.io/numba-cuda/user/installation.html
"""
Path("README.md").write_text(readme)
versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
Path("environment.txt").write_text(versions)
print("README.md generated from real GPU results.")
