# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103188  
**Allocated GPU Node:** Tesla T4  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** EC7BDC7A66732072FE14

## Execution

Executed on Google Colab, UTC: 2026-10-08T18:17:13.581219+00:00.
The source assignment is `CUDA Python Lab 01.pdf` (its internal title is Lab 02).
Files are placed in the requested `Paralel/week-6/08.10.26 - week - 6 - practice_task` folder.
The PDF separately specifies a repository named `cuda-lab-02-230103188`; this submission uses the folder requested by the student.

[Executed Google Colab notebook](https://colab.research.google.com/drive/1f04XrfEphbzAnWrbM0hrR0BygBe1D0s8) (owner: university account).

Open `CUDA_Lab02_230103188.ipynb` in Colab, choose **Runtime > Change runtime type > T4 GPU**, then run all cells.
The notebook writes the Python modules, runs the full-size tasks and instructor verifier, and regenerates this report.
For standalone execution in a CUDA environment: `python run_lab.py`.
For the instructor check alone: `python verify_submission.py`, then enter `230103188`.

## Task 1: Warp divergence benchmark

N = 1,048,576 float32 values; 1,000 iterations per value; 256 threads per block.
Each kernel has one untimed warm-up and 10 timed trials. Timers use `time.perf_counter()` with `cuda.synchronize()`.
Device buffers are allocated before timing; transfers and validation occur outside timing.
Wall-clock measurements include launch and synchronization overhead, as specified by the assignment.
Every trial reads the same input and writes a separate output.

| Kernel | Mean time (ms), 10 trials |
| --- | ---: |
| A_uniform | 3.704192 |
| B_interleaved | 9.477729 |
| C_warp_aligned | 4.090734 |

Interleaved / warp-aligned time ratio: **2.317x**.
A uses `v = v * 1.0001 + 0.0001`. B and C use that same path or `v = (v - 0.0001) / 1.0001`.
B assigns opposite paths to adjacent lanes; C assigns one path to each complete 32-thread warp.
B and C process equal numbers of values on each path, so their comparison isolates branch arrangement more directly than A versus B.
Divergence can serialize paths, but the measured slowdown also depends on compiler transformations, arithmetic cost and scheduling; an exact 2x penalty is not guaranteed.
All three outputs were checked against CPU arithmetic; small fused-operation rounding differences are allowed.
Raw trial times and maximum validation errors are in `benchmark_results.json`.

## Tasks 2-4: Results

- TASK 2 PASSED: MAX DELTA = 5.960464477539063e-08
- TASK 3 PASSED: 16777216 elements equal 4.25; 64 x 256 = 16384 threads
- TASK 4 PASSED: shape=(2048, 2048); MAX DELTA = 4.76837158203125e-07

- Stencil: replicated left/right boundary, 256 threads per block, CPU reference uses edge padding.
- Grid-stride: exactly 64 blocks x 256 threads = 16,384 threads; each thread processes 1,024 values at N = 2**24.
- Sobel-X: 16x16 threads per block; dynamic grid is 128x128 blocks for 2048x2048 input. All four output borders are zero.
- Sobel follows the signed formula in the PDF (cross-correlation convention), with a positive derivative along increasing columns.
- Extra GPU checks passed for singleton/two-element arrays, odd dimensions, partial blocks, negative scaling and a column ramp with expected gradient +8.

## Official verification

The provided verifier is preserved, with PDF line wrapping and indentation repaired.
Its smaller smoke-test sizes are run in addition to the full-size task requirements.

```text
============================================================
RUNNING CUDA LAB 02 AUTONOMOUS VERIFICATION
============================================================
Enter your Student ID: [PASS] Task 2 (1D Stencil & Clamping)
[PASS] Task 3 (Grid-Stride Scaling)
[PASS] Task 4 (2D Sobel Horizontal)

============================================================
VERIFICATION SUCCESSFUL
OFFICIAL SUBMISSION TOKEN: EC7BDC7A66732072FE14
============================================================
Copy this token directly into your README.md.
```

## Environment

See `nvidia-smi.txt` and `environment.txt` for runtime telemetry and package versions.
Numba-CUDA installation reference: https://nvidia.github.io/numba-cuda/user/installation.html

Installation note: the CUDA 12 Numba dependencies emitted pip conflict warnings for preinstalled CUDA 13 RAPIDS/PyTorch packages. Those packages are not used by this lab; all CUDA kernels and the official verifier completed successfully. The recorded notebook preserves the installation output.
