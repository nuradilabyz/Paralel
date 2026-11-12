"""
Phase 2: Strong Scaling vs. Weak Scaling & Architectural Ceilings

This script performs:
1. Strong scaling experiment (Amdahl's Law)
2. Weak scaling experiment (Gustafson's Law)
3. Architectural ceiling analysis
4. SMT/Hyper-Threading inflection detection

Strong scaling: Fixed problem size, increasing processors
Weak scaling: Problem size scales with processor count
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import sys
import os
import math

# Add parent directory to path to import harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_benchmark, emit_verify_tag


def strong_scaling_worker(shm_name: str, worker_id: int, p: int, iters: int) -> None:
    """
    Worker for strong scaling: each worker processes a fixed portion of work.
    Total work is constant regardless of p.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 1024).from_buffer(shm.buf)
    
    # Calculate work range for this worker
    work_per_worker = iters // p
    start = worker_id * work_per_worker
    end = (worker_id + 1) * work_per_worker if worker_id < p - 1 else iters
    
    acc = 0
    for i in range(start, end):
        acc = (acc * 6364136223846793005 + i) & 0xFFFFFFFFFFFFFFFF
    
    arr[worker_id] = acc
    del arr
    shm.close()


def weak_scaling_worker(shm_name: str, worker_id: int, p: int, base_iters: int) -> None:
    """
    Worker for weak scaling: each worker processes the same amount of work.
    Total work scales linearly with p.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 1024).from_buffer(shm.buf)
    
    # Each worker does base_iters work
    acc = 0
    for i in range(base_iters):
        acc = (acc * 6364136223846793005 + (i + worker_id * base_iters)) & 0xFFFFFFFFFFFFFFFF
    
    arr[worker_id] = acc
    del arr
    shm.close()


def run_strong_scaling(p: int, total_iters: int = 100_000_000) -> float:
    """
    Run strong scaling experiment.
    Total work (total_iters) is constant.
    Each of p workers processes total_iters/p iterations.
    
    Returns: Median execution time in milliseconds
    """
    def benchmark_fn():
        shm = shared_memory.SharedMemory(create=True, size=8192)
        arr = (ctypes.c_uint64 * 1024).from_buffer(shm.buf)
        for i in range(1024):
            arr[i] = 0
        del arr
        
        processes = []
        for t in range(p):
            proc = mp.Process(
                target=strong_scaling_worker,
                args=(shm.name, t, p, total_iters)
            )
            processes.append(proc)
            proc.start()
        
        for proc in processes:
            proc.join()
        
        shm.close()
        shm.unlink()
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    return stats.median_ms


def run_weak_scaling(p: int, base_iters: int = 1_000_000) -> float:
    """
    Run weak scaling experiment.
    Each worker does base_iters work, so total work = p * base_iters.
    
    Returns: Median execution time in milliseconds
    """
    def benchmark_fn():
        shm = shared_memory.SharedMemory(create=True, size=8192)
        arr = (ctypes.c_uint64 * 1024).from_buffer(shm.buf)
        for i in range(1024):
            arr[i] = 0
        del arr
        
        processes = []
        for t in range(p):
            proc = mp.Process(
                target=weak_scaling_worker,
                args=(shm.name, t, p, base_iters)
            )
            processes.append(proc)
            proc.start()
        
        for proc in processes:
            proc.join()
        
        shm.close()
        shm.unlink()
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    return stats.median_ms


def main():
    print("=" * 70)
    print("PHASE 2: Strong Scaling vs. Weak Scaling")
    print("=" * 70)
    print()
    
    # Get student ID
    if len(sys.argv) > 1:
        student_id = int(sys.argv[1])
    else:
        student_id = int(input("Enter your Student ID: ").strip())
    
    # Get physical core count
    Plog = mp.cpu_count()
    print(f"Student ID: {student_id}")
    print(f"Logical CPUs (Plog): {Plog}")
    
    # Determine physical cores
    Pphys = max(1, Plog // 2) if Plog > 4 else Plog
    print(f"Assuming Physical Cores (Pphys): {Pphys}")
    print()
    
    # Test all process counts from 1 to Plog
    process_counts = list(range(1, Plog + 1))
    
    # ========================================================================
    # STRONG SCALING (Amdahl)
    # ========================================================================
    print("STRONG SCALING (Amdahl's Law)")
    print("Fixed problem size: 100,000,000 iterations")
    print("-" * 70)
    
    strong_results = []
    t1_strong = None
    for p in process_counts:
        print(f"  Testing p = {p}...", end=" ")
        median_ms = run_strong_scaling(p)
        if p == 1:
            t1_strong = median_ms
        
        speedup = t1_strong / median_ms if t1_strong else 1.0
        efficiency = speedup / p
        
        strong_results.append({
            'p': p,
            'median_ms': median_ms,
            'speedup': speedup,
            'efficiency': efficiency,
            'work_per_worker': 100_000_000 // p
        })
        print(f"Done ({median_ms:.2f} ms, Speedup={speedup:.2f}, Eff={efficiency:.4f})")
    
    print()
    
    # ========================================================================
    # WEAK SCALING (Gustafson)
    # ========================================================================
    print("WEAK SCALING (Gustafson-Barsis's Law)")
    print("Scaled problem size: p * 1,000,000 iterations")
    print("-" * 70)
    
    weak_results = []
    t1_weak = None
    for p in process_counts:
        print(f"  Testing p = {p}...", end=" ")
        median_ms = run_weak_scaling(p)
        if p == 1:
            t1_weak = median_ms
        
        speedup = t1_weak / median_ms if t1_weak else 1.0
        efficiency = speedup / p
        
        weak_results.append({
            'p': p,
            'median_ms': median_ms,
            'speedup': speedup,
            'efficiency': efficiency,
            'total_work': p * 1_000_000
        })
        print(f"Done ({median_ms:.2f} ms, Speedup={speedup:.2f}, Eff={efficiency:.4f})")
    
    print()
    
    # ========================================================================
    # DISPLAY TABLES
    # ========================================================================
    print("STRONG SCALING TABLE")
    print("-" * 70)
    print(f"{'p':<5} {'Median (ms)':<12} {'Speedup':<10} {'Efficiency':<12} {'Work/Worker':<15}")
    print("-" * 70)
    for r in strong_results:
        print(f"{r['p']:<5} {r['median_ms']:<12.4f} {r['speedup']:<10.4f} {r['efficiency']:<12.4f} {r['work_per_worker']:<15,}")
    print()
    
    print("WEAK SCALING TABLE")
    print("-" * 70)
    print(f"{'p':<5} {'Median (ms)':<12} {'Speedup':<10} {'Efficiency':<12} {'Total Work':<15}")
    print("-" * 70)
    for r in weak_results:
        print(f"{r['p']:<5} {r['median_ms']:<12.4f} {r['speedup']:<10.4f} {r['efficiency']:<12.4f} {r['total_work']:<15,}")
    print()
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("ANALYSIS")
    print("-" * 70)
    
    # Find Amdahl ceiling for strong scaling
    if len(strong_results) > 1:
        # Use the last few points to estimate ceiling
        last_strong = strong_results[-1]
        amdahl_ceiling = last_strong['speedup'] * last_strong['p'] / last_strong['p']
        print(f"Strong Scaling:")
        print(f"  Max speedup at p={Pphys}: {strong_results[Pphys-1]['speedup']:.2f}")
        print(f"  Efficiency at p={Pphys}: {strong_results[Pphys-1]['efficiency']:.4f}")
        print(f"  Estimated Amdahl ceiling: {amdahl_ceiling:.4f}")
    
    print()
    
    # Check for SMT inflection
    print("SMT/Hyper-Threading Inflection:")
    if Pphys < Plog:
        print(f"  Physical cores: {Pphys}")
        print(f"  Logical cores: {Plog}")
        print(f"  Comparing p={Pphys} vs p={Pphys+1}...")
        
        if Pphys < len(strong_results):
            eff_phys = strong_results[Pphys-1]['efficiency']
            if Pphys < len(strong_results):
                eff_phys_plus = strong_results[Pphys]['efficiency']
                print(f"  Efficiency at p={Pphys}: {eff_phys:.4f}")
                print(f"  Efficiency at p={Pphys+1}: {eff_phys_plus:.4f}")
                if eff_phys_plus < eff_phys * 0.95:
                    print(f"  -> INFLECTION DETECTED: Efficiency drops when crossing physical core boundary")
                else:
                    print(f"  -> No significant inflection")
    
    print()
    
    # ========================================================================
    # VERIFICATION HASH
    # ========================================================================
    print("VERIFICATION HASH")
    print("-" * 70)
    
    # Use Pphys for verification
    strong_median = strong_results[Pphys-1]['median_ms'] if Pphys <= len(strong_results) else strong_results[-1]['median_ms']
    weak_median = weak_results[Pphys-1]['median_ms'] if Pphys <= len(weak_results) else weak_results[-1]['median_ms']
    
    verify_hash = emit_verify_tag(
        "PHASE2",
        student_id,
        strong_median,
        weak_median
    )
    print(verify_hash)
    
    print()
    
    # Save results
    with open(f"phase2_results_{student_id}.txt", "w") as f:
        f.write("PHASE 2 RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Student ID: {student_id}\n")
        f.write(f"Pphys: {Pphys}\n")
        f.write(f"Plog: {Plog}\n\n")
        
        f.write("STRONG SCALING (Amdahl)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'p':<5} {'Median (ms)':<12} {'Speedup':<10} {'Efficiency':<12} {'Work/Worker':<15}\n")
        f.write("-" * 70 + "\n")
        for r in strong_results:
            f.write(f"{r['p']:<5} {r['median_ms']:<12.4f} {r['speedup']:<10.4f} {r['efficiency']:<12.4f} {r['work_per_worker']:<15,}\n")
        f.write("\n")
        
        f.write("WEAK SCALING (Gustafson)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'p':<5} {'Median (ms)':<12} {'Speedup':<10} {'Efficiency':<12} {'Total Work':<15}\n")
        f.write("-" * 70 + "\n")
        for r in weak_results:
            f.write(f"{r['p']:<5} {r['median_ms']:<12.4f} {r['speedup']:<10.4f} {r['efficiency']:<12.4f} {r['total_work']:<15,}\n")
        f.write("\n")
        
        f.write("Verification Hash\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
        f.write("\n")
        
        f.write("Analysis\n")
        f.write("-" * 70 + "\n")
        f.write(f"Strong Scaling Max Speedup (p={Pphys}): {strong_results[Pphys-1]['speedup'] if Pphys <= len(strong_results) else strong_results[-1]['speedup']:.4f}\n")
        f.write(f"Weak Scaling Speedup (p={Pphys}): {weak_results[Pphys-1]['speedup'] if Pphys <= len(weak_results) else weak_results[-1]['speedup']:.4f}\n")
    
    print(f"\nResults saved to: phase2_results_{student_id}.txt")
    print("\nPHASE 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
