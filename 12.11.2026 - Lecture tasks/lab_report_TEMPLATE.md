# Parallel Computing Lab Report
## Performance Evaluation and Optimization in Parallel Computing
### Solo Python Systems Lab

**Student ID:** [YOUR_ID]
**Date:** 12.11.2026
**Mode:** Solo Laptop Execution (Offline)
**Execution Time:** 120+ Minutes

---

## Table of Contents

1. [Hardware Baseline & Verification Hashes](#1-hardware-baseline--verification-hashes)
2. [Phase 1: False Sharing & Cache-Line Contention](#2-phase-1-false-sharing--cache-line-contention)
3. [Phase 2: Strong vs. Weak Scaling](#3-phase-2-strong-vs-weak-scaling)
4. [Phase 3: Empirical Roofline & GEMM Optimization](#4-phase-3-empirical-roofline--gemm-optimization)
5. [Phase 4: Scheduling & Load Imbalance](#5-phase-4-scheduling--load-imbalance)
6. [Source Code Appendix](#6-source-code-appendix)

---

## 1. Hardware Baseline & Verification Hashes

### CPU/Cache Topology

| Hardware Parameter | Measured Value |
|-------------------|----------------|
| CPU Model String | [Fill from OS] |
| Physical Cores (Pphys) | [Fill integer] |
| Logical Threads (Plog) | [Fill integer] |
| L1 Data Cache (per core) | [Fill in KB] |
| L2 Cache Size | [Fill in KB/MB] |
| L3 Cache Size (Shared) | [Fill in MB] |
| Cache Line Size (Lbytes) | [Fill in Bytes] |
| Python Version & OS | [e.g., Python 3.11 / Linux] |

### Timer Noise Floor Calibration

| Metric | Value |
|--------|-------|
| Min | [ ] ns |
| Max | [ ] ns |
| Median | [ ] ns |
| Mean | [ ] ns |
| Std Dev | [ ] ns |

### Verification Hashes

```
VERIFY_HASH[PHASE0]: [Your hash here] (P_log=[value])
VERIFY_HASH[PHASE1]: [Your hash here] (P_log=[value])
VERIFY_HASH[PHASE2]: [Your hash here] (P_log=[value])
VERIFY_HASH[PHASE3]: [Your hash here] (P_log=[value])
VERIFY_HASH[PHASE4]: [Your hash here] (P_log=[value])
```

---

## 2. Phase 1: False Sharing & Cache-Line Contention

### 1A. Stride Sensitivity at p = Pphys

| Stride (words) | Stride (bytes) | Median (ms) | IQR (ms) | Throughput (Mops/s) |
|----------------|----------------|-------------|----------|---------------------|
| 1 | 8 | | | |
| 2 | 16 | | | |
| 4 | 32 | | | |
| 8 | 64 | | | |
| 16 | 128 | | | |
| 32 | 256 | | | |

**Cache-Line Boundary:** [Detected at __ bytes]

**Analysis:** The performance improves significantly at stride = [ ] words ([ ] bytes), which corresponds to the cache line size of this CPU. This demonstrates that when processes write to the same cache line (stride < cache line size), false sharing causes severe performance degradation due to cache coherence traffic.

### 1B. Karp-Flatt Diagnostic Series

#### Stride = 1 (Severe False Sharing)

| p | Tp (ms) | Sp = T1/Tp | Ep = Sp/p | e(p) = (1/p)(1/Sp - 1) |
|---|---------|-----------|-----------|------------------------|
| 1 | | 1.00 | 1.000 | 0.000 |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| ... | ... | ... | ... | ... |

**Explanation:** As p increases with stride=1, all processes write to adjacent words on the same cache line, causing continuous invalidation. The serial fraction e(p) rises because the effective "serial" portion (cache line contention) grows with p.

#### Stride = 16 (128-byte Isolation)

| p | Tp (ms) | Sp = T1/Tp | Ep = Sp/p | e(p) = (1/p)(1/Sp - 1) |
|---|---------|-----------|-----------|------------------------|
| 1 | | 1.00 | 1.000 | 0.000 |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| ... | ... | ... | ... | ... |

**Explanation:** With stride=16, each process writes to its own cache line, eliminating false sharing. The serial fraction e(p) remains low, demonstrating true parallel scaling.

---

## 3. Phase 2: Strong vs. Weak Scaling

### Strong Scaling (Amdahl's Law) - Fixed Problem Size

Problem size: 100,000,000 iterations

| p | Median (ms) | Speedup (Sp) | Efficiency (Ep) | Work/Worker | Notes |
|---|-------------|--------------|-----------------|-------------|-------|
| 1 | | 1.00 | 1.000 | 100,000,000 | Baseline |
| 2 | | | | 50,000,000 | |
| 3 | | | | 33,333,333 | |
| 4 | | | | 25,000,000 | |
| ... | ... | ... | ... | ... | ... |

**Amdahl Ceiling:** [Estimate from data]

**Analysis:** Strong scaling shows diminishing returns as p increases because the fixed problem size cannot fully utilize all cores. The speedup is limited by the serial fraction of the algorithm.

### Weak Scaling (Gustafson-Barsis's Law) - Scaled Problem Size

Problem size: p × 1,000,000 iterations

| p | Median (ms) | Speedup (Sp) | Efficiency (Ep) | Total Work | Notes |
|---|-------------|--------------|-----------------|------------|-------|
| 1 | | 1.00 | 1.000 | 1,000,000 | Baseline |
| 2 | | | | 2,000,000 | |
| 3 | | | | 3,000,000 | |
| 4 | | | | 4,000,000 | |
| ... | ... | ... | ... | ... | ... |

**Analysis:** Weak scaling shows better scalability as the problem size grows proportionally with p. This demonstrates that with sufficient parallel work, efficiency can be maintained.

### SMT/Hyper-Threading Inflection

**Observation:** At p = [Pphys value], the efficiency [increases/decreases/stays same] when crossing from physical cores to logical threads (SMT/Hyper-Threading). This indicates that [explain: whether SMT provides benefit or causes contention].

---

## 4. Phase 3: Empirical Roofline & GEMM Optimization

### DRAM Streaming Bandwidth Measurement

| n | Bandwidth (GB/s) | Time (ms) |
|---|-------------------|-----------|
| 1,000,000 | | |
| 5,000,000 | | |
| 10,000,000 | | |
| 20,000,000 | | |

**Sustained DRAM Bandwidth (BW): [ ] GB/s**

### FMA Peak Measurement

| n | GFLOPS | Time (ms) |
|---|--------|-----------|
| 1,000 | | |
| 10,000 | | |
| 100,000 | | |
| 1,000,000 | | |
| 5,000,000 | | |

**Sustained FMA Peak (P): [ ] GFLOPS**

### Empirical Roofline Model

- **Bandwidth (BW):** [ ] GB/s
- **Peak FLOPS (P):** [ ] GFLOPS
- **Ridge Point (I"):** [ ] FLOPS/byte

**Interpretation:**
- For arithmetic intensity < I": **Memory-bound**
- For arithmetic intensity > I": **Compute-bound**

### GEMM Optimization with Tile Size Sweep

Matrix size: 500×500

| Tile Size | Time (ms) | Speedup vs Naive | Footprint (bytes) | GFLOPS |
|-----------|-----------|-------------------|-------------------|--------|
| naive | | 1.00 | N/A | |
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 24 | | | | |
| 32 | | | | |
| 48 | | | | |
| 64 | | | | |
| 128 | | | | |

**Optimal Tile Size (C"):** [ ]

**Explanation:** The optimal tile size balances cache utilization (keeping working set in cache) with loop overhead. Tile size [ ] achieves the best performance because [explain: footprint fits in L1/L2 cache, reduces cache misses, etc.].

---

## 5. Phase 4: Scheduling & Load Imbalance

### Workload Description

- **Total items (N):** 12,000
- **Skew pattern:** K(i) = 1 + floor((i/60)^2)
- **Range:** Item 0 performs 1 iteration, Item 11,999 performs ~40,000 iterations

### Strategy 1: Static Block

| Metric | Value |
|--------|-------|
| Wall-clock time | [ ] ms |
| Load imbalance λ | [ ] |
| Worker times | [list] |

### Strategy 2: Cyclic (Round-Robin)

| Metric | Value |
|--------|-------|
| Wall-clock time | [ ] ms |
| Load imbalance λ | [ ] |
| Worker times | [list] |

### Strategy 3: Dynamic Chunk (Work-Stealing)

| C | Wall Time (ms) | λ |
|---|----------------|---|
| 1 | | |
| 4 | | |
| 16 | | |
| 64 | | |
| 256 | | |
| 1024 | | |

### Optimal Chunk Size Analysis

**Best C":** [ ]

**Trade-off Explanation:**
- **C = 1:** Minimum tail skew (all chunks same size), but maximum lock contention (every iteration requires synchronization)
- **C = 1024:** Minimum synchronization overhead (fewer lock acquisitions), but high load imbalance (last chunks may have skew)
- **C":** Optimal balance between synchronization overhead and load imbalance

### Summary Comparison

| Strategy | Wall Time | λ | Notes |
|----------|-----------|---|-------|
| Static Block | | | Fastest for uniform workloads, worst for skewed |
| Cyclic | | | Better balance for moderate skew |
| Dynamic (C") | | | Best for highly skewed workloads |

---

## 6. Source Code Appendix

### File List

1. **harness.py** - Zero-dependency Python 3 telemetry & verification framework
   - `BenchStats` dataclass
   - `fnv1a_hash()` - FNV-1a 64-bit hash
   - `run_benchmark()` - Benchmark execution with warmup and timing
   - `emit_verify_tag()` - Verification hash generation

2. **task0_cal.py** - Phase 0: Hardware topology audit & calibration
   - CPU topology detection
   - Timer noise floor measurement
   - VERIFY_HASH[PHASE0] generation

3. **task1_falsesharing.py** - Phase 1: False sharing diagnosis
   - `fs_worker()` - Worker function with stride parameter
   - Stride sweep (1-32 words)
   - Process scaling sweep
   - Karp-Flatt metric calculation
   - Cache line boundary detection
   - VERIFY_HASH[PHASE1] generation

4. **task2_scaling.py** - Phase 2: Strong vs. weak scaling
   - Strong scaling with fixed problem size
   - Weak scaling with scaled problem size
   - Amdahl & Gustafson-Barsis analysis
   - SMT inflection detection
   - VERIFY_HASH[PHASE2] generation

5. **task3_roofline.py** - Phase 3: Roofline & GEMM
   - STREAM Triad bandwidth measurement
   - FMA peak measurement
   - Ridge point calculation
   - GEMM with blocking/tiling
   - Tile size sweep (10 points)
   - VERIFY_HASH[PHASE3] generation

6. **task4_scheduling.py** - Phase 4: Scheduling strategies
   - Static block scheduling
   - Cyclic (round-robin) scheduling
   - Dynamic chunk queue with work-stealing
   - Chunk size sweep (1, 4, 16, 64, 256, 1024)
   - Load imbalance factor calculation
   - VERIFY_HASH[PHASE4] generation

---

## Execution Instructions

1. **Disconnect from internet:** Disable Wi-Fi/Ethernet
2. **Run Phase 0:** `python task0_cal.py [STUDENT_ID]`
3. **Run Phase 1:** `python task1_falsesharing.py [STUDENT_ID]`
4. **Run Phase 2:** `python task2_scaling.py [STUDENT_ID]`
5. **Run Phase 3:** `python task3_roofline.py [STUDENT_ID]`
6. **Run Phase 4:** `python task4_scheduling.py [STUDENT_ID]`
7. **Compile report:** Copy all results into this template

---

## Notes

- All experiments use `multiprocessing.Process` to bypass the Python GIL
- All shared memory uses `multiprocessing.shared_memory` for true inter-process communication
- All timing uses `time.perf_counter_ns()` for nanosecond precision
- All verification hashes use FNV-1a 64-bit hash with hardware-bound payloads
- No external libraries or AI tools were used (per lab constraints)

---

**Submission Checklist:**

- [ ] Hardware baseline table complete
- [ ] All 5 VERIFY_HASH values included
- [ ] Phase 1: Tables 1A and 1B with cache-line identification
- [ ] Phase 2: Strong and weak scaling tables with analysis
- [ ] Phase 3: Roofline model and GEMM tile sweep
- [ ] Phase 4: Scheduling comparison and optimal chunk size
- [ ] All Python source code included or referenced
- [ ] Raw terminal output saved (if applicable)
