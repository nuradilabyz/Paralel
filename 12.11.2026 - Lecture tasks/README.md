# 12.11.2026 - Lecture Tasks

## Parallel Computing Python Solo Lab v5

This directory contains all files and results for the **Performance Evaluation and Optimization in Parallel Computing** lab as described in `parallelcomputingpythonsololabv5.pdf`.

### Lab Overview

- **Minimum Runtime:** 120+ Minutes (5 Sequential Stages)
- **Mode:** Solo Laptop Execution (Offline)
- **Constraints:** No Paper, No Internet, No AI Tools
- **Language:** Pure Python 3 (Standard Library Only)

### Directory Structure

```
12.11.2026 - Lecture tasks/
├── README.md                          # This file
├── lab_report_TEMPLATE.md              # Complete lab report template
├── harness.py                          # Phase 0: Telemetry & verification framework
├── task0_cal.py                        # Phase 0: Hardware topology & calibration
├── task1_falsesharing.py               # Phase 1: False sharing diagnosis
├── task2_scaling.py                    # Phase 2: Strong/weak scaling
├── task3_roofline.py                   # Phase 3: Roofline & GEMM optimization
└── task4_scheduling.py                 # Phase 4: Scheduling strategies
```

### Lab Phases

#### Phase 0: Hardware Topology Audit & Harness Calibration (15 mins)
- CPU/cache topology interrogation
- Timer noise-floor calibration
- Building the `harness.py` module
- Generates: `VERIFY_HASH[PHASE0]`

#### Phase 1: False Sharing & Cache-Line Contention (25 mins)
- Shared-memory false sharing stride sweep (1-32 words)
- Cache-line boundary detection
- Karp-Flatt metric calculation
- Generates: `VERIFY_HASH[PHASE1]`

#### Phase 2: Strong vs. Weak Scaling (25 mins)
- Amdahl's Law (strong scaling)
- Gustafson-Barsis's Law (weak scaling)
- SMT/Hyper-Threading inflection analysis
- Generates: `VERIFY_HASH[PHASE2]`

#### Phase 3: Empirical Roofline & GEMM Optimization (35 mins)
- DRAM streaming bandwidth measurement
- FMA peak measurement
- 3-stage GEMM optimization
- 10-point tile size sweep
- Generates: `VERIFY_HASH[PHASE3]`

#### Phase 4: Skewed Workload Scheduling (20 mins)
- Static block scheduling
- Cyclic (round-robin) scheduling
- Dynamic chunk queue (work-stealing)
- Load imbalance factor analysis
- Generates: `VERIFY_HASH[PHASE4]`

### Key Design Principles

1. **True Hardware Concurrency:** All parallel kernels use `multiprocessing.Process` to bypass Python's GIL
2. **Shared Memory:** Uses `multiprocessing.shared_memory` for inter-process communication
3. **Verification:** Cryptographic hashes (FNV-1a 64-bit) ensure result integrity
4. **Deterministic:** All measurements are reproducible with the same hardware

### Technical Requirements

- **Python Version:** 3.8+
- **Standard Library Modules:**
  - `multiprocessing` (with `shared_memory`)
  - `ctypes`
  - `time` (with `perf_counter_ns`)
  - `math`
  - `dataclasses` (Python 3.7+)

### Execution Instructions

1. **Disconnect from internet** (required by lab protocol)
2. **Run each phase in order:**
   ```bash
   python task0_cal.py YOUR_STUDENT_ID
   python task1_falsesharing.py YOUR_STUDENT_ID
   python task2_scaling.py YOUR_STUDENT_ID
   python task3_roofline.py YOUR_STUDENT_ID
   python task4_scheduling.py YOUR_STUDENT_ID
   ```
3. **Compile results** into `lab_report_TEMPLATE.md`

### Verification Hash Format

All hashes follow the format:
```
VERIFY_HASH[PHASE#]: XXXXXXXXXXXXXXXXXX (P_log=#)
```

Where:
- `XXXXXXXXXXXXXXXX` is a 64-bit hexadecimal FNV-1a hash
- `P_log` is the number of logical CPUs detected

### Important Notes

1. **macOS/Windows Compatibility:** The code handles `spawn` mode for multiprocessing
2. **Shared Memory Cleanup:** All shared memory is properly unlinked after use
3. **Cross-Platform:** Hardware detection works on Linux, macOS, and Windows
4. **No External Dependencies:** Pure Python 3 standard library only

### Reference Material

- **Lab Manual:** `parallelcomputingpythonsololabv5.pdf`
- **Original Location:** `/Users/nuradilabyz/Desktop/week-11_tasks/`

### Contributing

This is a solo lab. No external contributions permitted per the lab constraints.

---

**Prepared by:** [Your Name]
**Student ID:** [Your ID]
**Date:** 12.11.2026
