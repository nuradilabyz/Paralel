# Empirical Scalability Diagnostics: Karp-Flatt

**Student:** Nuradil Abyz, **ID:** 230103188. **Date:** 29 October 2026. **Branch:** `practice-230103188`. Submission location: `29.10.2026/practice tasks` in `nuradilabyz/Paralel`.

## Part 0 - host and measurement method

Physical host: **Apple M2**, 8 physical cores, 8 logical execution cores; 4 Performance cores; 4 Efficiency cores. Platform: macOS-26.5.1-arm64-arm-64bit. Python: 3.12.14 (main, Aug 25 2026, 13:50:33) [Clang 22.1.3 ]. [Recorded host profile](results/host_profile.json).

The three programs use Python's standard library and the `spawn` start method. Each P in {1,2,4} has five genuine trials; table values are medians. Worker-count order rotates between trials. Experiments run sequentially. Timers include pool/process creation, computation, IPC, final reduction where applicable, and process cleanup; A also includes the specified sleep. B allocates its counter before timing, as in the handout. No CPU pinning or thermal controls were imposed, so background activity, startup and mixed core types can affect timing. No example runtimes were substituted for measurements. Code and analysis were prepared with AI assistance; all timings were collected on this host.

## Metric calculation and requested synthesis table

For each experiment independently, `S(P)=T1/TP`, `E(P)=100*S(P)/P`, and

`e(P)=(1/S(P)-1/P)/(1-1/P)=(P*TP/T1-1)/(P-1)` for P>=2.

Work is fixed across P within each experiment. e(1) is undefined. The two Karp-Flatt expressions are algebraically equivalent; calculations below use unrounded medians.

| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A: serial tail | 3.258407 | 1.884376 | 1.246746 | 0.156624 | 0.176832 | Approximately flat effective serialization; intentional sleep plus process overhead |
| B: contention | 4.177759 | 10.994604 | 11.568103 | 4.263399 | 3.358631 | Per-increment mutex contention; observe actual speedup and e below |
| C: privatization | 0.433531 | 0.242111 | 0.150482 | 0.116929 | 0.129478 | Private counters remove the per-increment shared lock; startup remains timed |

## Part 1 - Experiment A: the stubborn serial tail

[`exp_a_serial.py`](exp_a_serial.py) computes the handout's 16 chunks of `sum(sin(i)*cos(i), i=1..1,999,999)`, with `sleep(0.40)` before creating the pool. Total useful work is unchanged when P changes. Every chunk is checked against the independent trigonometric sum identity `0.5*sin(m)*sin(m+1)/sin(1)`, where m=1,999,999, with absolute tolerance 1e-8.

The observed speedups are S2=1.7292 and S4=2.6135. e(4)/e(2)=1.1290; the rubric's [0.80,1.25] flatness criterion is **MET**. The deliberate pause alone is 12.28% of the measured P=1 runtime. Effective Karp-Flatt includes process startup and IPC as well as that pause, so it need not equal this fraction or fall exactly in the handout's illustrative 0.10-0.14 interval. These values are compatible with a roughly fixed serial contribution, not proof that sleep is the only overhead.

## Part 2 - Experiment B: the contention traffic jam

[`exp_b_contention.py`](exp_b_contention.py) performs exactly **2,000,000 total increments**, partitioned among P processes. The shared `Value('i',0)` and explicit `Lock` are preserved from the handout: Value has its own internal lock, and the explicit mutex protects the complete read-modify-write. Every trial verifies the final count exactly. No artificial sleep is used.

T4/T1=2.7690; S4=0.3611. The negative-speedup condition T4>T1 is **MET**. e(2)=4.263399, e(4)=3.358631; e4/e2=0.7878 (**NOT MET** for e4>e2). The controlled implementation repeatedly acquires locks around a single shared counter. A monotonic rise in e is not guaranteed: if lock-limited runtimes plateau while P grows, the metric's denominator also grows, and e can decline despite obvious negative speedup. We did not measure cache misses, bus traffic, kernel sleep states or lock spinning, so these are mechanisms consistent with the experiment rather than separately measured facts.

## Part 3 - Experiment C: the privatization fix

[`exp_c_fixed.py`](exp_c_fixed.py) performs exactly **20,000,000 total increments** in private Python integer variables. Each worker returns one partial result through Pool.map; the parent sums the results once. Every partial and the final 20,000,000 count are checked. Pool IPC and startup remain, but no shared lock is acquired per increment. A Python integer is not guaranteed to remain in a hardware register.

S4=2.8809; E4=72.02%. T4<T1: **MET**. E4>70%: **MET**. The illustrative e<=0.03 target is **NOT MET** for both P=2 and P=4. Finite tasks and timed process startup can prevent ideal scaling even after the per-increment mutex is removed. B and C use different total workloads (2M versus 20M), so their raw seconds are not a fixed-work speedup comparison.

## Part 4 - verification, interpretation and provenance

Local numerical and metric verification: **PASS**. [`verify_results.py`](verify_results.py) checks all 45 recorded trials, independently recomputes medians, speedups, efficiencies and Karp-Flatt, verifies A's analytic sums and B/C's exact counts, and records SHA-256 digests of sources and measured data in [verification.json](results/verification.json). Numerical correctness and hardware-dependent performance targets are reported separately.

Karp-Flatt is an effective overhead measure, not a unique hardware diagnosis. A flat curve can be compatible with fixed serialization; a rising curve can reflect locks, scheduling, bandwidth, cache effects or other P-dependent costs. The controlled A/B/C implementations support the interpretations above without claiming that e alone proves a particular physical mechanism. The example table in the [handout](Practice%20Lab%20Handout.pdf) also contains arithmetic inconsistencies: its B times (0.65,1.10,2.40 s) imply e2=2.384615 and e4=4.589744 under its printed formula. Our results use the formula and actual times rather than those example metric values.

The source PDF is preserved unchanged. The handout asks for RESULTS.md at a repository root; this submission places it at the root of the requested `practice tasks` folder. Genuine staged commits record A, B and C/results, and the completed solution is available on both `main` and `practice-230103188` after push. No instructor sign-off is implied.

## Complete raw timing tables

### Experiment A: speedup, efficiency and spread

| P | Median (s) | Min (s) | Max (s) | Speedup | Efficiency (%) | e(P) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.258407 | 3.145820 | 3.312341 | 1.0000 | 100.00 | undefined |
| 2 | 1.884376 | 1.857944 | 1.947626 | 1.7292 | 86.46 | 0.156624 |
| 4 | 1.246746 | 1.236878 | 1.278995 | 2.6135 | 65.34 | 0.176832 |

[Raw trials](results/experiment_a.json) / [metrics TSV](results/experiment_a.tsv) / [console](results/experiment_a_console.txt)

### Experiment B: speedup, efficiency and spread

| P | Median (s) | Min (s) | Max (s) | Speedup | Efficiency (%) | e(P) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.177759 | 4.063046 | 4.238481 | 1.0000 | 100.00 | undefined |
| 2 | 10.994604 | 10.374377 | 11.743802 | 0.3800 | 19.00 | 4.263399 |
| 4 | 11.568103 | 11.248104 | 11.809339 | 0.3611 | 9.03 | 3.358631 |

[Raw trials](results/experiment_b.json) / [metrics TSV](results/experiment_b.tsv) / [console](results/experiment_b_console.txt)

### Experiment C: speedup, efficiency and spread

| P | Median (s) | Min (s) | Max (s) | Speedup | Efficiency (%) | e(P) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.433531 | 0.427303 | 0.491746 | 1.0000 | 100.00 | undefined |
| 2 | 0.242111 | 0.237421 | 0.243429 | 1.7906 | 89.53 | 0.116929 |
| 4 | 0.150482 | 0.147454 | 0.162023 | 2.8809 | 72.02 | 0.129478 |

[Raw trials](results/experiment_c.json) / [metrics TSV](results/experiment_c.tsv) / [console](results/experiment_c_console.txt)

