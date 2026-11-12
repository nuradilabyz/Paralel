# CS482 practicum — 29.10.2026

On the physical Apple M2, stride-16 reads cost 3.30× the stride-1 time per element, consistent with its 128-byte cache line; changing dense multiplication from i-j-k to i-k-j gave 4.97× speedup. At P=8, native padding gave 1.15× and privatization 2.51× the adjacent-counter throughput for the same work. Strong stencil scaling peaked at 1.52× with P=2; weak throughput peaked at 2.68× with P=4. Memory traffic, interpreter work, barriers and heterogeneous-core scheduling limit further scaling; these measurements do not isolate DRAM saturation.

Student: **Nuradil Abyz**, ID **230103188**. Salted N=1408, seed=87657. Implementation and analysis prepared with AI assistance; timings collected on the physical host.

## Host and method

[Host profile](results/host_profile.json), [raw sysctl](results/sysctl_raw.txt), [salted parameters](results/salted_params.json). Captured 2026-10-29T09:18:52.535233+00:00; macOS-26.5.1-arm64-arm-64bit; CPython 3.12.14, NumPy 2.3.5; 8 physical cores / 8 threads, 8 GiB memory. No core pinning. Experiments ran sequentially, with parallel workers only within each experiment.

| Core group | Physical cores | L1D KiB/core | L2 MiB/group |
| --- | --- | --- | --- |
| Performance | 4 | 128 | 16 |
| Efficiency | 4 | 64 | 4 |

Base frequency: unreported. L3: unreported. Cache-hit/miss counters were not collected. Generic sysctl cache sizes are supplemented by the heterogeneous per-core-group values above; unavailable telemetry is not invented.

## M1 — locality and loop interchange

The contiguous float64 buffer contains 16,777,216 elements (128 MiB). Each stride has five warmups and five recorded trials; the median is normalized by the actual number of elements touched. On this host, a line holds 16 doubles, so the PDF's eight-double/64-byte cache-line assumption does not apply literally.

| Stride | Median ns/read | Logical read GB/s |
| --- | --- | --- |
| 1 | 0.922 | 8.673 |
| 8 | 1.127 | 7.097 |
| 16 | 3.040 | 2.631 |

Dense tests allocate full N×N inputs and multiply their leading M×M blocks (M=704, original row pitch N), with three recorded runs per order. Explicit [C loops](src/m1_kernels.c) are called by Python `ctypes`; benchmark loops do not call BLAS. Allocation and output zeroing are excluded. Compiler: Apple clang version 17.0.0 (clang-1700.0.13.5); flags: `-O3 -std=c11 -fno-fast-math -ffp-contract=off -dynamiclib`.

| Loop order | Three samples (s) | Median (s) |
| --- | --- | --- |
| i-j-k | 0.296709, 0.303281, 0.424756 | 0.303281 |
| i-k-j | 0.061036, 0.059249, 0.063319 | 0.061036 |

Maximum absolute error between loop orders: 0; against NumPy: 1.14e-13, with `atol=1e-12, rtol=0`. Raw timings and salted rounded-six-decimal digests: [stride TSV](results/stride_results.tsv), [stride trials](results/stride_trials.json), [dense results](results/loop_interchange.json).

## M2 — false sharing

K=5,000,000 increments per worker, 3 trials; medians below exclude spawn and result IPC, and span the synchronized worker interval. Counters use aligned shared storage and distinct ownership with no global per-increment lock; private counters publish once under a lock. All 48 trials returned exactly P × K increments, including 40,000,000 at P=8.

| P | Adjacent (s) | 64 B padding (s) | 128 B padding (s) | Private (s) |
| --- | --- | --- | --- | --- |
| 1 | 0.272895 | 0.278561 | 0.282101 | 0.109758 |
| 2 | 0.300062 | 0.274662 | 0.277050 | 0.105662 |
| 4 | 0.338877 | 0.308477 | 0.339582 | 0.126612 |
| 8 | 0.658341 | 0.587528 | 0.570489 | 0.261846 |

64-byte spacing can leave two counters on one 128-byte cache line. Native padding separates them; observed recovery is limited, not assumed perfect. At P=8, adjacent/native-padding runtime ratio is 1.154×.

| P | Raw e adjacent | Raw e native padding | Work-normalized speedup adjacent | Work-normalized speedup native padding |
| --- | --- | --- | --- | --- |
| 1 | — | — | 1.000 | 1.000 |
| 2 | 1.199 | 0.964 | 1.819 | 2.036 |
| 4 | 1.322 | 1.272 | 3.221 | 3.323 |
| 8 | 2.614 | 2.168 | 3.316 | 3.956 |

Every worker performs K operations, so total work grows as P×K. The PDF-requested `S=T1/TP` and `e=(1/S−1/P)/(1−1/P)` are raw-time diagnostics here, **not fixed-work serial fractions**. Work-normalized speedup is `P*T1/TP`; its serial baseline is estimated, not independently measured. e(1) is undefined. Python dispatch, OS scheduling and mixed core speeds can obscure coherence effects. [Raw trials and exact counters](results/false_sharing_trials.json) · [all metrics](results/false_sharing_karp_flatt.tsv).

## M3 — stencil scaling and Roofline

Five-point float64 stencil, 128 timesteps, 3 trials per configuration. Processes own disjoint row ranges in two shared buffers; every timestep ends at a barrier. Timings include barriers, excluding allocation, spawn, reference checks and result copies; setup times are recorded separately. Each complete trial is checked against its matching serial reference, including zero boundaries and salted digest.

Strong scaling keeps N=1408 fixed:

| P | N | Median (s) | Speedup | Efficiency % |
| --- | --- | --- | --- | --- |
| 1 | 1408 | 1.195822 | 1.000 | 100.0 |
| 2 | 1408 | 0.784170 | 1.525 | 76.2 |
| 4 | 1408 | 0.814322 | 1.468 | 36.7 |
| 8 | 1408 | 0.795102 | 1.504 | 18.8 |

Weak scaling uses `N(P)=ceil(N*sqrt(P))` and actual interior-cell work:

| P | N | Median (s) | Throughput speedup | Efficiency % | PDF formula |
| --- | --- | --- | --- | --- | --- |
| 1 | 1408 | 1.174618 | 1.000 | 100.0 | 1.000 |
| 2 | 1992 | 1.334258 | 1.764 | 88.2 | 0.879 |
| 4 | 2816 | 1.757795 | 2.677 | 66.9 | 0.667 |
| 8 | 3983 | 3.699664 | 2.545 | 31.8 | 0.317 |

Strong and weak metrics use the measured P=1 median from their own study; both P=1 rows normalize to one. Throughput speedup is `(WP/W1)*(T1/TP)`; efficiency is that value/P. The PDF's `P*W1*T1/(WP*TP)` approximates weak efficiency rather than scaled speedup when WP≈P×W1.

Amdahl least-squares fit: s=0.553518 (unconstrained 0.553518), ideal asymptotic ceiling 1.807×, normalized-time RMSE 0.0670. This is an effective model, not a measured immutable serial fraction. The reported Gustafson `P−s(P−1)` is an **illustrative prediction using the strong-fit s**; a scaled-run serial share was not separately measured. [Scaling trials and fit](results/scaling_trials.json) · [scaling TSV](results/scaling_metrics.tsv).

Roofline uses 5 FLOPs/update and an ideal 16 B/update with perfect row reuse and no write-allocation overhead: AI=0.3125 FLOPs/B. The implemented NumPy ufunc passes make 112 logical B/update; logical traffic is not measured DRAM traffic. Observed copy bandwidth was 18.955 GB/s; best stencil throughput was 2.883 GFLOP/s. Apple advertises up to [100 GB/s unified-memory bandwidth](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/); this is a manufacturer maximum shared by the system, not this run's measured DRAM bandwidth. A compute roof and hardware DRAM counters are unavailable, so “memory-bound” remains a model-based hypothesis with synchronization also limiting performance. [Roofline assumptions and raw copy trials](results/roofline_summary.json).

## Verification and task differences

**Provided verifier: incompatible with this host.** It requires `cache_line_bytes == 64`; the measured value is 128. Its actual failure is retained in [provided_verifier.log](results/provided_verifier.log).

The [original verifier](verify_outputs.py) is byte-for-byte preserved (SHA-256 `bce8d586fd800f888d5001cb6919fd915590f9ec25dab1ab62bd703a502ebb21`). It checks telemetry fields and trends but does not itself verify dense arrays, exact counter totals or stencil arrays. Independent numerical verification status: **PASS**; see [numerical_verification.json](results/numerical_verification.json). [verification_receipt.sig](results/verification_receipt.sig) is a local numerical self-audit receipt, not an authenticated instructor receipt or proof that the supplied checker passed.

The [original handout](ParallelComputingLabPracticum_Paradigm.pdf) assumes 64-byte cache lines and describes an eight-digit ID; the established repository ID has nine digits and is salted without truncation. Python/NumPy orchestrates every experiment; transparent native C loops are used for dense and stride measurements. No frequencies, L3 size, hardware counters or performance trends were fabricated. The requested submission path is `29.10.2026/Lecture tasks`.

## Reproduce

From this directory, install NumPy in a Python 3.10+ environment and provide `clang` or another C compiler (`CC` may select it). The captured run used CPython 3.12.14, NumPy 2.3.5 and Apple clang version 17.0.0 (clang-1700.0.13.5). Build products remain in ignored `.build/`; all raw results are retained. These commands overwrite results with new genuine host measurements:

```bash
python3 src/profile_env.py --student-id 230103188
python3 src/stride_bench.py
python3 src/dense_kernel.py
python3 src/false_sharing.py
python3 src/scaling_study.py
python3 src/verify_numerics.py
python3 src/build_summary.py
# The unchanged instructor checker exits unsuccessfully on this 128-byte host:
python3 verify_outputs.py
```

[stride log](results/stride_console.txt) · [dense log](results/dense_console.txt) · [false_sharing log](results/false_sharing_console.txt) · [scaling log](results/scaling_console.txt)
