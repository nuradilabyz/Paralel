"""
Phase 1: Forensic Diagnosis of False Sharing & Cache-Line Contention

This script performs:
1. False sharing stride sweep (1 to 32 64-bit words)
2. Cache-line boundary detection
3. Karp-Flatt metric calculation
4. Process scaling sweep

The experiment demonstrates how shared cache lines cause performance degradation
when multiple CPU cores write to adjacent memory locations.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import sys
import os
import math

# Add parent directory to path to import harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_benchmark, emit_verify_tag, fnv1a_hash


def fs_worker(shm_name: str, slot_idx: int, iters: int, stride: int = 1) -> None:
    """
    Worker function for false sharing experiment.
    Each worker updates a specific slot in shared memory.
    
    Args:
        shm_name: Name of the shared memory block
        slot_idx: Index of the slot this worker will update
        iters: Number of iterations to perform
        stride: Stride between slots (in 64-bit words)
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    # Cast buffer to array of 64-bit unsigned integers
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    
    # Initialize accumulator with slot-specific seed
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005  # Multiplicative constant for LCG
    
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    
    del arr
    shm.close()


def run_false_sharing_experiment(p: int, stride: int, total_iters: int = 24_000_000):
    """
    Run the false sharing experiment with p processes and given stride.
    
    Args:
        p: Number of processes
        stride: Stride between slots (in 64-bit words)
        total_iters: Total number of iterations across all processes
    
    Returns:
        Median execution time in milliseconds
    """
    iters_per_worker = total_iters // p
    
    def benchmark_fn():
        # Create shared memory block (4096 bytes = 64 cache lines on x86)
        shm = shared_memory.SharedMemory(create=True, size=4096)
        
        # Initialize shared memory
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        for i in range(512):
            arr[i] = i * 0xDEADBEEF
        del arr
        
        processes = []
        for t in range(p):
            slot_idx = t * stride
            proc = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters_per_worker, stride)
            )
            processes.append(proc)
            proc.start()
        
        for proc in processes:
            proc.join()
        
        shm.close()
        shm.unlink()
    
    # Run benchmark
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    return stats.median_ms


def calculate_karp_flatt(p: int, t1_ms: float, tp_ms: float) -> float:
    """
    Calculate the Karp-Flatt serial fraction e(p).
    
    e(p) = (1/p) * (1/S_p - 1)
    where S_p = T1/Tp
    
    Args:
        p: Number of processes
        t1_ms: Execution time with 1 process
        tp_ms: Execution time with p processes
    
    Returns:
        Karp-Flatt serial fraction e(p)
    """
    if tp_ms <= 0 or p <= 1:
        return 0.0
    
    speedup = t1_ms / tp_ms
    if speedup <= 0:
        return 0.0
    
    e_p = (1.0 / p) * (1.0 / speedup - 1.0)
    return e_p


def main():
    print("=" * 70)
    print("PHASE 1: False Sharing & Cache-Line Contention")
    print("=" * 70)
    print()
    
    # Get student ID
    if len(sys.argv) > 1:
        student_id = int(sys.argv[1])
    else:
        student_id = int(input("Enter your Student ID: ").strip())
    
    # Get physical core count
    Pphys = mp.cpu_count()
    print(f"Student ID: {student_id}")
    print(f"Logical CPUs (Plog): {Pphys}")
    print()
    
    # Determine physical cores (assume at least half of logical are physical)
    # This is a simplification - in practice, use topology detection
    Pphys_actual = max(1, Pphys // 2) if Pphys > 4 else Pphys
    print(f"Assuming Physical Cores (Pphys): {Pphys_actual}")
    print()
    
    total_iters = 24_000_000  # 2.4e7 as specified
    
    # ========================================================================
    # STRIDE SWEEP at p = Pphys
    # ========================================================================
    print("STRIDE SWEEP (p = Pphys)")
    print("-" * 70)
    
    strides = [1, 2, 4, 8, 16, 32]  # 64-bit word strides
    stride_bytes = [s * 8 for s in strides]  # Convert to bytes
    
    stride_results = []
    for stride in strides:
        print(f"  Testing stride = {stride} words ({stride * 8} bytes)...", end=" ")
        median_ms = run_false_sharing_experiment(Pphys_actual, stride, total_iters)
        throughput = (total_iters / 1e6) / (median_ms / 1000)  # Mops/s
        stride_results.append({
            'stride': stride,
            'stride_bytes': stride * 8,
            'median_ms': median_ms,
            'iqr_ms': 0,  # Will be calculated properly
            'throughput': throughput
        })
        print(f"Done ({median_ms:.2f} ms)")
    
    print()
    print("Table 1A: Stride Sensitivity at p = Pphys")
    print("-" * 70)
    print(f"{'Stride (words)':<15} {'Stride (bytes)':<15} {'Median (ms)':<15} {'Throughput (Mops/s)':<20}")
    print("-" * 70)
    for result in stride_results:
        print(f"{result['stride']:<15} {result['stride_bytes']:<15} {result['median_ms']:<15.4f} {result['throughput']:<20.2f}")
    print()
    
    # Find cache line boundary (where performance improves significantly)
    # Look for the point where median_ms drops significantly
    cache_line_boundary = None
    for i in range(1, len(stride_results)):
        if stride_results[i]['median_ms'] < stride_results[i-1]['median_ms'] * 0.7:
            cache_line_boundary = stride_results[i]['stride_bytes']
            print(f"Cache line boundary detected at {cache_line_boundary} bytes")
            break
    
    if cache_line_boundary is None:
        cache_line_boundary = 64  # Default assumption for x86
        print(f"Assuming standard cache line size: {cache_line_boundary} bytes")
    
    print()
    
    # ========================================================================
    # PROCESS SCALING SWEEP
    # ========================================================================
    print("PROCESS SCALING SWEEP")
    print("-" * 70)
    
    # Test all process counts from 1 to Pphys
    process_counts = list(range(1, Pphys_actual + 1))
    
    # Test with stride = 1 (severe false sharing)
    print("  Testing with stride = 1 (severe false sharing)...")
    scaling_stride1 = []
    t1_ms = None
    for p in process_counts:
        median_ms = run_false_sharing_experiment(p, 1, total_iters)
        if p == 1:
            t1_ms = median_ms
        speedup = t1_ms / median_ms if t1_ms else 1.0
        efficiency = speedup / p
        karp_flatt = calculate_karp_flatt(p, t1_ms, median_ms) if p > 1 and t1_ms else 0.0
        scaling_stride1.append({
            'p': p,
            'median_ms': median_ms,
            'speedup': speedup,
            'efficiency': efficiency,
            'karp_flatt': karp_flatt
        })
        print(f"    p={p}: {median_ms:.2f} ms, Speedup={speedup:.2f}, Eff={efficiency:.4f}, e(p)={karp_flatt:.4f}")
    
    print()
    
    # Test with stride = 16 (128-byte isolation)
    print("  Testing with stride = 16 (128-byte isolation)...")
    scaling_stride16 = []
    t1_ms_16 = None
    for p in process_counts:
        median_ms = run_false_sharing_experiment(p, 16, total_iters)
        if p == 1:
            t1_ms_16 = median_ms
        speedup = t1_ms_16 / median_ms if t1_ms_16 else 1.0
        efficiency = speedup / p
        karp_flatt = calculate_karp_flatt(p, t1_ms_16, median_ms) if p > 1 and t1_ms_16 else 0.0
        scaling_stride16.append({
            'p': p,
            'median_ms': median_ms,
            'speedup': speedup,
            'efficiency': efficiency,
            'karp_flatt': karp_flatt
        })
        print(f"    p={p}: {median_ms:.2f} ms, Speedup={speedup:.2f}, Eff={efficiency:.4f}, e(p)={karp_flatt:.4f}")
    
    print()
    
    # ========================================================================
    # TABLE 1B: Karp-Flatt Diagnostic Series
    # ========================================================================
    print("Table 1B: Karp-Flatt Diagnostic Series")
    print("-" * 70)
    print(f"{'p':<5} {'Stride=1':<20} {'Stride=16':<20}")
    print(f"{'':<5} {'Sp':<8} {'Ep':<8} {'e(p)':<8} {'Sp':<8} {'Ep':<8} {'e(p)':<8}")
    print("-" * 70)
    
    for i in range(len(process_counts)):
        p = process_counts[i]
        s1 = scaling_stride1[i]
        s16 = scaling_stride16[i]
        print(f"{p:<5} {s1['speedup']:<8.2f} {s1['efficiency']:<8.4f} {s1['karp_flatt']:<8.4f} {s16['speedup']:<8.2f} {s16['efficiency']:<8.4f} {s16['karp_flatt']:<8.4f}")
    
    print()
    
    # ========================================================================
    # VERIFICATION HASH
    # ========================================================================
    print("VERIFICATION HASH")
    print("-" * 70)
    
    # Get median_ms for stride=1 and stride=16 at p = Pphys
    median_stride1 = stride_results[0]['median_ms']  # stride=1
    median_stride16 = stride_results[4]['median_ms']  # stride=16
    
    verify_hash = emit_verify_tag(
        "PHASE1",
        student_id,
        median_stride1,
        median_stride16
    )
    print(verify_hash)
    
    print()
    
    # ========================================================================
    # FORENSIC EXPLANATION
    # ========================================================================
    print("FORENSIC EXPLANATION")
    print("-" * 70)
    print("Under stride=1:")
    print("  All processes write to adjacent 8-byte words on the same cache line.")
    print("  This causes continuous cache line invalidation (MESI/MOESI coherence traffic).")
    print("  As p increases, more cores compete for the same cache line, increasing e(p).")
    print("  The Karp-Flatt serial fraction rises because the 'serial' part (cache line")
    print("  contention) grows with p.")
    print()
    print("Under stride=16:")
    print("  Processes write to slots 128 bytes apart (2 cache lines on x86).")
    print("  Each process works on its own cache line, eliminating false sharing.")
    print("  e(p) remains low as p increases, showing true parallel scaling.")
    print()
    print(f"Cache line boundary detected at: {cache_line_boundary} bytes")
    
    # Save results
    with open(f"phase1_results_{student_id}.txt", "w") as f:
        f.write("PHASE 1 RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Student ID: {student_id}\n")
        f.write(f"Pphys: {Pphys_actual}\n")
        f.write(f"Plog: {Pphys}\n\n")
        
        f.write("Table 1A: Stride Sensitivity\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Stride (words)':<15} {'Stride (bytes)':<15} {'Median (ms)':<15} {'Throughput (Mops/s)':<20}\n")
        f.write("-" * 70 + "\n")
        for result in stride_results:
            f.write(f"{result['stride']:<15} {result['stride_bytes']:<15} {result['median_ms']:<15.4f} {result['throughput']:<20.2f}\n")
        f.write("\n")
        
        f.write("Table 1B: Karp-Flatt Diagnostic Series\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'p':<5} {'Stride=1':<20} {'Stride=16':<20}\n")
        f.write(f"{'':<5} {'Sp':<8} {'Ep':<8} {'e(p)':<8} {'Sp':<8} {'Ep':<8} {'e(p)':<8}\n")
        f.write("-" * 70 + "\n")
        for i in range(len(process_counts)):
            p = process_counts[i]
            s1 = scaling_stride1[i]
            s16 = scaling_stride16[i]
            f.write(f"{p:<5} {s1['speedup']:<8.2f} {s1['efficiency']:<8.4f} {s1['karp_flatt']:<8.4f} {s16['speedup']:<8.2f} {s16['efficiency']:<8.4f} {s16['karp_flatt']:<8.4f}\n")
        f.write("\n")
        
        f.write("Verification Hash\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
        f.write("\n")
        
        f.write("Cache Line Boundary\n")
        f.write("-" * 70 + "\n")
        f.write(f"Detected at: {cache_line_boundary} bytes\n")
        f.write(f"Median (stride=1): {median_stride1:.4f} ms\n")
        f.write(f"Median (stride=16): {median_stride16:.4f} ms\n")
    
    print(f"\nResults saved to: phase1_results_{student_id}.txt")
    print("\nPHASE 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
