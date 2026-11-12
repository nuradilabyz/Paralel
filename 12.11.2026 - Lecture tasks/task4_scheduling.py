"""
Phase 4: Skewed Workload Scheduling & Dynamic Work-Stealing

This script performs:
1. Static block scheduling
2. Cyclic (round-robin) scheduling
3. Dynamic chunk queue scheduling
4. Load imbalance factor calculation

The experiment demonstrates the trade-offs between different scheduling
strategies for skewed workloads.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import sys
import os
import math
import time

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


def static_block_worker(task_id: int, p: int, N: int, shm_name: str) -> None:
    """
    Static block scheduling: worker t processes contiguous range.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # Calculate range
    work_per_worker = N // p
    start = task_id * work_per_worker
    end = (task_id + 1) * work_per_worker if task_id < p - 1 else N
    
    # Access shared array for timing
    timing_arr = (ctypes.c_double * p).from_buffer(shm.buf)
    
    t_start = time.perf_counter()
    
    for i in range(start, end):
        compute_work_item(i)
    
    t_end = time.perf_counter()
    timing_arr[task_id] = (t_end - t_start) * 1000  # Convert to ms
    
    del timing_arr
    shm.close()


def cyclic_worker(task_id: int, p: int, N: int, shm_name: str) -> None:
    """
    Cyclic scheduling: worker t processes indices t, t+p, t+2p, ...
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    timing_arr = (ctypes.c_double * p).from_buffer(shm.buf)
    
    t_start = time.perf_counter()
    
    for i in range(task_id, N, p):
        compute_work_item(i)
    
    t_end = time.perf_counter()
    timing_arr[task_id] = (t_end - t_start) * 1000  # Convert to ms
    
    del timing_arr
    shm.close()


def dynamic_chunk_worker(task_id: int, p: int, N: int, shm_name: str, chunk_size: int) -> None:
    """
    Dynamic chunk scheduling: workers claim chunks from a shared counter.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # First p * sizeof(double) bytes are timing array
    # Next sizeof(ctypes.c_int) bytes are shared counter
    timing_arr = (ctypes.c_double * p).from_buffer(shm.buf)
    counter = ctypes.c_int.from_buffer(shm.buf, p * 8)
    
    t_start = time.perf_counter()
    
    while True:
        # Atomically increment counter
        with counter.get_lock():
            current = counter.value
            counter.value += chunk_size
        
        if current >= N:
            break
        
        end = min(current + chunk_size, N)
        for i in range(current, end):
            compute_work_item(i)
    
    t_end = time.perf_counter()
    timing_arr[task_id] = (t_end - t_start) * 1000  # Convert to ms
    
    del timing_arr
    del counter
    shm.close()


def run_scheduling_experiment(p: int, N: int, strategy: str, chunk_size: int = 1) -> tuple:
    """
    Run a scheduling experiment.
    
    Returns: (total_wall_time, worker_times, load_imbalance)
    """
    shm = shared_memory.SharedMemory(create=True, size=p * 8 + 4)
    
    # Initialize shared memory
    timing_arr = (ctypes.c_double * p).from_buffer(shm.buf)
    for i in range(p):
        timing_arr[i] = 0.0
    
    # For dynamic, initialize counter
    counter = ctypes.c_int.from_buffer(shm.buf, p * 8)
    counter.value = 0
    
    del timing_arr
    del counter
    
    def benchmark_fn():
        pass  # Workers do the work
    
    # Create and start workers
    processes = []
    
    if strategy == "block":
        for t in range(p):
            proc = mp.Process(
                target=static_block_worker,
                args=(t, p, N, shm.name)
            )
            processes.append(proc)
            proc.start()
    
    elif strategy == "cyclic":
        for t in range(p):
            proc = mp.Process(
                target=cyclic_worker,
                args=(t, p, N, shm.name)
            )
            processes.append(proc)
            proc.start()
    
    elif strategy == "dynamic":
        for t in range(p):
            proc = mp.Process(
                target=dynamic_chunk_worker,
                args=(t, p, N, shm.name, chunk_size)
            )
            processes.append(proc)
            proc.start()
    
    # Wait for all workers to complete
    t_start = time.perf_counter()
    for proc in processes:
        proc.join()
    t_end = time.perf_counter()
    
    wall_time = (t_end - t_start) * 1000  # Convert to ms
    
    # Read timing results
    timing_arr = (ctypes.c_double * p).from_buffer(shm.buf)
    worker_times = [timing_arr[i] for i in range(p)]
    
    del timing_arr
    shm.close()
    shm.unlink()
    
    # Calculate load imbalance factor
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
    
    wall_time_block, worker_times_block, lambda_block = run_scheduling_experiment(
        Pphys, N, "block"
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
    
    wall_time_cyclic, worker_times_cyclic, lambda_cyclic = run_scheduling_experiment(
        Pphys, N, "cyclic"
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
        wall_time, worker_times, lambda_val = run_scheduling_experiment(
            Pphys, N, "dynamic", C
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
    
    # Find C* that minimizes wall time
    best_C = min(dynamic_results, key=lambda x: x['wall_time'])
    print(f"Best wall-clock time: C = {best_C['C']} ({best_C['wall_time']:.2f} ms)")
    print(f"Best load balance: C = {min(dynamic_results, key=lambda x: x['lambda'])['C']}")
    print()
    
    print("Trade-off analysis:")
    print("  C = 1: Minimum tail skew, but maximum lock contention")
    print("  C = 1024: Minimum lock contention, but high load imbalance")
    print(f"  C* = {best_C['C']}: Optimal balance for this system")
    print()
    
    # ========================================================================
    # VERIFICATION HASH
    # ========================================================================
    print("VERIFICATION HASH")
    print("-" * 70)
    
    # Use Strategy 1 and Strategy 3 (best C*) wall times
    c_star = best_C['C']
    c_star_wall = best_C['wall_time']
    
    verify_hash = emit_verify_tag(
        "PHASE4",
        student_id,
        wall_time_block,
        c_star_wall
    )
    print(verify_hash)
    
    print()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
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
        f.write(f"  Best C: {best_C['C']}\n")
        f.write(f"  Best wall time: {best_C['wall_time']:.2f} ms\n")
        f.write(f"  Best λ: {best_C['lambda']:.4f}\n\n")
        
        f.write("Verification Hash\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
    
    print(f"\nResults saved to: phase4_results_{student_id}.txt")
    print("\nPHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
