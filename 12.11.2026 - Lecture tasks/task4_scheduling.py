"""
Phase 4: Skewed Workload Scheduling & Dynamic Work-Stealing

Simplified version using threading and shared counters for actual execution.
"""

import multiprocessing as mp
import numpy as np
import sys
import os
import math
import time
from multiprocessing import shared_memory
import ctypes

# Add parent directory to path to import harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_benchmark, emit_verify_tag


def compute_work_item(i: int) -> float:
    """Compute a single work item with triangular skew."""
    k = 1 + (i // 60) ** 2
    result = 0.0
    for _ in range(k):
        result += math.sin(_) * math.cos(_) * math.sqrt(_ + 1)
    return result


def static_block_worker(args):
    """Static block scheduling worker."""
    task_id, p, N, result_arr = args
    work_per_worker = N // p
    start = task_id * work_per_worker
    end = (task_id + 1) * work_per_worker if task_id < p - 1 else N
    
    t_start = time.perf_counter()
    for i in range(start, end):
        compute_work_item(i)
    t_end = time.perf_counter()
    
    result_arr[task_id] = (t_end - t_start) * 1000  # ms


def cyclic_worker(args):
    """Cyclic scheduling worker."""
    task_id, p, N, result_arr = args
    
    t_start = time.perf_counter()
    for i in range(task_id, N, p):
        compute_work_item(i)
    t_end = time.perf_counter()
    
    result_arr[task_id] = (t_end - t_start) * 1000  # ms


def dynamic_worker(args):
    """Dynamic chunk scheduling worker."""
    task_id, p, N, result_arr, chunk_size, counter, lock = args
    
    t_start = time.perf_counter()
    
    while True:
        with lock:
            current = counter.value
            counter.value += chunk_size
        
        if current >= N:
            break
        
        end = min(current + chunk_size, N)
        for i in range(current, end):
            compute_work_item(i)
    
    t_end = time.perf_counter()
    result_arr[task_id] = (t_end - t_start) * 1000  # ms


def run_scheduling(strategy: str, p: int, N: int, chunk_size: int = 1) -> tuple:
    """
    Run scheduling experiment.
    Returns: (wall_time_ms, worker_times, lambda_imbalance)
    """
    # Create result array
    result_arr = mp.Array('d', p)  # Array of doubles
    
    if strategy == "block":
        workers = []
        for t in range(p):
            args = (t, p, N, result_arr)
            proc = mp.Process(target=static_block_worker, args=(args,))
            workers.append(proc)
    
    elif strategy == "cyclic":
        workers = []
        for t in range(p):
            args = (t, p, N, result_arr)
            proc = mp.Process(target=cyclic_worker, args=(args,))
            workers.append(proc)
    
    elif strategy == "dynamic":
        # Create shared counter and lock
        counter = mp.Value('i', 0)
        lock = mp.Lock()
        
        workers = []
        for t in range(p):
            args = (t, p, N, result_arr, chunk_size, counter, lock)
            proc = mp.Process(target=dynamic_worker, args=(args,))
            workers.append(proc)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Start all workers
    for proc in workers:
        proc.start()
    
    # Wait for all workers
    t_start = time.perf_counter()
    for proc in workers:
        proc.join()
    t_end = time.perf_counter()
    
    wall_time = (t_end - t_start) * 1000  # ms
    
    # Get worker times
    worker_times = [result_arr[i] for i in range(p)]
    
    # Calculate load imbalance
    max_time = max(worker_times)
    avg_time = sum(worker_times) / p
    lambda_imbalance = max_time / avg_time if avg_time > 0 else 1.0
    
    return wall_time, worker_times, lambda_imbalance


def main():
    print("=" * 70)
    print("PHASE 4: Skewed Workload Scheduling & Dynamic Work-Stealing")
    print("=" * 70)
    print()
    
    # Get student ID
    if len(sys.argv) > 1:
        student_id = int(sys.argv[1])
    else:
        student_id = int(input("Enter your Student ID: ").strip())
    
    # Parameters
    N = 12_000  # 12,000 work items
    Pphys = max(1, mp.cpu_count() // 2) if mp.cpu_count() > 4 else mp.cpu_count()
    
    print(f"Student ID: {student_id}")
    print(f"Physical Cores (Pphys): {Pphys}")
    print(f"Total work items (N): {N:,}")
    print()
    
    # Chunk sizes to test
    chunk_sizes = [1, 4, 16, 64, 256, 1024]
    
    # ========================================================================
    # STRATEGY 1: STATIC BLOCK
    # ========================================================================
    print("Strategy 1: Static Block Scheduling")
    print("-" * 70)
    
    wall_time_block, worker_times_block, lambda_block = run_scheduling(
        "block", Pphys, N
    )
    
    print(f"Wall-clock time: {wall_time_block:.2f} ms")
    print(f"Load imbalance λ: {lambda_block:.4f}")
    print(f"Worker times: {[f'{t:.2f}' for t in worker_times_block]}")
    print()
    
    # ========================================================================
    # STRATEGY 2: CYCLIC
    # ========================================================================
    print("Strategy 2: Cyclic (Round-Robin) Scheduling")
    print("-" * 70)
    
    wall_time_cyclic, worker_times_cyclic, lambda_cyclic = run_scheduling(
        "cyclic", Pphys, N
    )
    
    print(f"Wall-clock time: {wall_time_cyclic:.2f} ms")
    print(f"Load imbalance λ: {lambda_cyclic:.4f}")
    print(f"Worker times: {[f'{t:.2f}' for t in worker_times_cyclic]}")
    print()
    
    # ========================================================================
    # STRATEGY 3: DYNAMIC CHUNK
    # ========================================================================
    print("Strategy 3: Dynamic Chunk Scheduling")
    print("-" * 70)
    
    dynamic_results = []
    
    for C in chunk_sizes:
        print(f"  Testing chunk size C = {C}...", end=" ")
        wall_time, worker_times, lambda_val = run_scheduling(
            "dynamic", Pphys, N, C
        )
        dynamic_results.append({
            'C': C,
            'wall_time': wall_time,
            'worker_times': worker_times,
            'lambda': lambda_val
        })
        print(f"Done ({wall_time:.2f} ms, λ={lambda_val:.4f})")
    
    print()
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print("SCHEDULING COMPARISON")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Wall Time (ms)':<15} {'λ':<10}")
    print("-" * 70)
    print(f"{'Static Block':<20} {wall_time_block:<15.2f} {lambda_block:<10.4f}")
    print(f"{'Cyclic':<20} {wall_time_cyclic:<15.2f} {lambda_cyclic:<10.4f}")
    print()
    
    print("Dynamic Chunk Sweep:")
    print(f"{'C':<5} {'Wall Time (ms)':<15} {'λ':<10}")
    print("-" * 70)
    for r in dynamic_results:
        print(f"{r['C']:<5} {r['wall_time']:<15.2f} {r['lambda']:<10.4f}")
    print()
    
    # ========================================================================
    # FIND OPTIMAL CHUNK SIZE
    # ========================================================================
    print("OPTIMAL CHUNK SIZE ANALYSIS")
    print("-" * 70)
    
    best_C = min(dynamic_results, key=lambda x: x['wall_time'])
    print(f"Best wall-clock time: C = {best_C['C']} ({best_C['wall_time']:.2f} ms)")
    
    best_balance = min(dynamic_results, key=lambda x: x['lambda'])
    print(f"Best load balance: C = {best_balance['C']} (λ = {best_balance['lambda']:.4f})")
    print()
    print("Trade-off analysis:")
    print("  C = 1: Minimum tail skew, but maximum lock contention")
    print("  C = 1024: Minimum synchronization overhead, but high load imbalance")
    print(f"  C* = {best_C['C']}: Optimal balance for this system")
    print()
    
    # ========================================================================
    # VERIFICATION HASH
    # ========================================================================
    print("VERIFICATION HASH")
    print("-" * 70)
    
    verify_hash = emit_verify_tag(
        "PHASE4",
        student_id,
        wall_time_block,
        best_C['wall_time']
    )
    print(verify_hash)
    
    print()
    
    # Save results
    with open(f"phase4_results_{student_id}.txt", "w") as f:
        f.write("PHASE 4 RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Student ID: {student_id}\n")
        f.write(f"Pphys: {Pphys}\n")
        f.write(f"N: {N:,}\n\n")
        
        f.write("Static Block Scheduling\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Wall-clock time: {wall_time_block:.2f} ms\n")
        f.write(f"  Load imbalance λ: {lambda_block:.4f}\n")
        f.write(f"  Worker times: {worker_times_block}\n\n")
        
        f.write("Cyclic Scheduling\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Wall-clock time: {wall_time_cyclic:.2f} ms\n")
        f.write(f"  Load imbalance λ: {lambda_cyclic:.4f}\n")
        f.write(f"  Worker times: {worker_times_cyclic}\n\n")
        
        f.write("Dynamic Chunk Scheduling\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'C':<5} {'Wall Time (ms)':<15} {'λ':<10}\n")
        f.write("-" * 70 + "\n")
        for r in dynamic_results:
            f.write(f"{r['C']:<5} {r['wall_time']:<15.2f} {r['lambda']:<10.4f}\n")
        f.write("\n")
        
        f.write("Optimal Chunk Size\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Best C (time): {best_C['C']}\n")
        f.write(f"  Best wall time: {best_C['wall_time']:.2f} ms\n")
        f.write(f"  Best C (balance): {best_balance['C']}\n")
        f.write(f"  Best λ: {best_balance['lambda']:.4f}\n\n")
        
        f.write("Verification Hash\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
    
    print(f"\nResults saved to: phase4_results_{student_id}.txt")
    print("\nPHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
