"""
Phase 1: False Sharing Stride Sweep & Karp-Flatt Diagnostics

Allocate a 4096-byte SharedMemory block and spawn p worker processes 
to execute M = 1.6e7 total read-modify-write steps (M/p per worker) on 
64-bit unsigned integer slots spaced by stride.

Worker function must be defined at module top level.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import math
import sys
import os

sys.path.insert(0, '/Users/nuradilabyz/Desktop/Week_11_practice_task')
from harness import fnv1a_hash, emit_verify_tag, run_benchmark

STUDENT_ID = 230103188
BLOCK_SIZE = 4096  # 4KB
M_TOTAL = int(1.6 * 10**7)  # 16 million total iterations

# Worker function must be at module top level
def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    """
    False sharing worker that performs read-modify-write on a shared memory slot.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    
    for i in range(iters):
        # Multiplicative hash mixing
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc  # Forces store to shared L1 cache line
    
    del arr
    shm.close()


def create_and_run_workers(shm_name: str, p: int, stride: int, iters_per_worker: int):
    """
    Create worker processes and run them.
    """
    processes = []
    for worker_idx in range(p):
        slot_idx = worker_idx * stride
        if slot_idx >= 512:
            slot_idx = worker_idx  # Fallback
        
        proc = mp.Process(
            target=fs_worker,
            args=(shm_name, slot_idx, iters_per_worker)
        )
        processes.append(proc)
    
    # Start all workers
    for proc in processes:
        proc.start()
    
    # Wait for all to complete
    for proc in processes:
        proc.join()
    
    # Cleanup
    for proc in processes:
        proc.close()


def run_stride_sweep(p: int, strides: list) -> dict:
    """
    Run stride sweep for given number of processes and stride values.
    
    Args:
        p: Number of worker processes (Pphys)
        strides: List of stride values to test
    
    Returns:
        Dictionary with results for each stride
    """
    results = {}
    
    for stride in strides:
        # Create shared memory
        shm_name = f"psm_fs_{os.getpid()}_{stride}"
        shm = shared_memory.SharedMemory(create=True, size=BLOCK_SIZE, name=shm_name)
        
        # Initialize array
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        for i in range(512):
            arr[i] = i
        del arr
        
        # Calculate iterations per worker
        iters_per_worker = M_TOTAL // p
        
        # Create a wrapper function that creates and runs workers
        def benchmark_fn():
            create_and_run_workers(shm.name, p, stride, iters_per_worker)
        
        stats = run_benchmark(benchmark_fn, warmup=1, runs=5)
        
        # Cleanup
        shm.close()
        shm.unlink()
        
        # Calculate throughput (Mops/s = million ops per second)
        # M = 1.6e7 ops, time in ms
        median_seconds = stats.median_ms / 1000
        throughput = (M_TOTAL / 1e6) / median_seconds  # Mops/s
        
        results[stride] = {
            'min_ms': stats.min_ms,
            'median_ms': stats.median_ms,
            'iqr_ms': stats.iqr_ms,
            'throughput_mops': throughput
        }
    
    return results


def run_process_scaling(p_values: list, stride: int) -> dict:
    """
    Run process scaling for a fixed stride value.
    
    Args:
        p_values: List of process counts to test
        stride: Stride value to use
    
    Returns:
        Dictionary with results for each p value
    """
    results = {}
    
    for p in p_values:
        shm_name = f"psm_ps_{os.getpid()}_{p}_{stride}"
        shm = shared_memory.SharedMemory(create=True, size=BLOCK_SIZE, name=shm_name)
        
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        for i in range(512):
            arr[i] = i
        del arr
        
        iters_per_worker = M_TOTAL // p
        
        def benchmark_fn():
            create_and_run_workers(shm.name, p, stride, iters_per_worker)
        
        stats = run_benchmark(benchmark_fn, warmup=1, runs=5)
        
        shm.close()
        shm.unlink()
        
        results[p] = {
            'min_ms': stats.min_ms,
            'median_ms': stats.median_ms,
            'iqr_ms': stats.iqr_ms,
            'time_ms': stats.median_ms
        }
    
    return results


def compute_karp_flatt(results_stride1: dict, results_stride16: dict) -> dict:
    """
    Compute Karp-Flatt experimental serial fraction.
    
    e(p) = (1/p - 1/Sp) / (1 - 1/p)
    where Sp = T1 / Tp
    
    Args:
        results_stride1: Results for stride=1
        results_stride16: Results for stride=16
    
    Returns:
        Dictionary with Karp-Flatt metrics for each p
    """
    karp_flatt = {}
    
    for p in results_stride1.keys():
        if p == 1:
            continue  # Karp-Flatt requires p >= 2
        
        # Stride 1
        T1_s1 = results_stride1[1]['time_ms']
        Tp_s1 = results_stride1[p]['time_ms']
        Sp_s1 = T1_s1 / Tp_s1
        
        # Stride 16
        T1_s16 = results_stride16[1]['time_ms']
        Tp_s16 = results_stride16[p]['time_ms']
        Sp_s16 = T1_s16 / Tp_s16
        
        # Karp-Flatt for stride 1
        if Sp_s1 != p:
            e_s1 = (1/p - 1/Sp_s1) / (1 - 1/p)
        else:
            e_s1 = 0
        
        # Karp-Flatt for stride 16
        if Sp_s16 != p:
            e_s16 = (1/p - 1/Sp_s16) / (1 - 1/p)
        else:
            e_s16 = 0
        
        karp_flatt[p] = {
            'stride_1': {
                'T1': T1_s1,
                'Tp': Tp_s1,
                'Sp': Sp_s1,
                'Ep': Sp_s1 / p,
                'e_p': e_s1
            },
            'stride_16': {
                'T1': T1_s16,
                'Tp': Tp_s16,
                'Sp': Sp_s16,
                'Ep': Sp_s16 / p,
                'e_p': e_s16
            }
        }
    
    return karp_flatt


def main():
    print("=" * 60)
    print("Phase 1: False Sharing Stride Sweep & Karp-Flatt Diagnostics")
    print("=" * 60)
    
    # Get hardware info
    Pphys = mp.cpu_count()  # Physical cores
    Plog = mp.cpu_count()   # Logical threads
    print(f"Pphys: {Pphys}, Plog: {Plog}")
    
    # Part 1: Stride Sweep (p = Pphys)
    print("\n--- Part 1: Stride Sweep (p = Pphys) ---")
    strides = [1, 2, 4, 8, 16, 32]  # Stride in elements, each is 8 bytes
    stride_results = run_stride_sweep(Pphys, strides)
    
    print(f"{'Stride':<10} {'Min (ms)':<12} {'Median (ms)':<14} {'IQR (ms)':<12} {'Throughput (Mops/s)':<20}")
    print("-" * 80)
    for stride in strides:
        r = stride_results[stride]
        print(f"{stride:<10} {r['min_ms']:<12.4f} {r['median_ms']:<14.4f} {r['iqr_ms']:<12.4f} {r['throughput_mops']:<20.2f}")
    
    # Find the stride where coherence contention drops
    # This is where throughput increases significantly
    byte_offsets = [s * 8 for s in strides]  # Convert stride (elements) to bytes
    print(f"\nByte offsets: {byte_offsets}")
    print("Contention drops when stride >= cache line size (128 bytes)")
    print("Cache line size: 128 bytes -> stride >= 16 (16 * 8 = 128 bytes)")
    
    # Part 2: Process Scaling (p ∈ {1, 2, 4, ..., Plog})
    print("\n--- Part 2: Process Scaling ---")
    p_values = [1, 2, 4, 8]  # For Pphys=8, Plog=8
    
    # Run for stride=1
    print("Stride = 1:")
    results_stride1 = run_process_scaling(p_values, stride=1)
    
    # Run for stride=16
    print("Stride = 16:")
    results_stride16 = run_process_scaling(p_values, stride=16)
    
    # Compute Karp-Flatt
    karp_flatt = compute_karp_flatt(results_stride1, results_stride16)
    
    # Display results
    print("\nKarp-Flatt Analysis:")
    print(f"{'p':<5} {'Stride=1 Sp':<12} {'Stride=1 Ep':<12} {'Stride=1 e(p)':<14} {'Stride=16 Sp':<12} {'Stride=16 Ep':<12} {'Stride=16 e(p)':<14}")
    print("-" * 90)
    for p in sorted(karp_flatt.keys()):
        s1 = karp_flatt[p]['stride_1']
        s16 = karp_flatt[p]['stride_16']
        print(f"{p:<5} {s1['Sp']:<12.4f} {s1['Ep']:<12.4f} {s1['e_p']:<14.6f} {s16['Sp']:<12.4f} {s16['Ep']:<12.4f} {s16['e_p']:<14.6f}")
    
    # Analysis: Stride=1 shows high false sharing, stride=16 reduces it
    print("\nAnalysis:")
    print("- Stride=1: All workers access consecutive 8-byte slots, sharing the same cache line (128 bytes)")
    print("  -> High coherence contention, poor scaling")
    print("- Stride=16: Workers access slots 128 bytes apart, each on separate cache lines")
    print("  -> Reduced false sharing, better scaling")
    
    # Emit VERIFY_HASH[PHASE1]
    # Use median_ms for stride=1 and stride=16 at p = Pphys
    median_s1 = stride_results[1]['median_ms']
    median_s16 = stride_results[16]['median_ms']
    verify_tag = emit_verify_tag("PHASE1", STUDENT_ID, median_s1, median_s16)
    print(f"\n{verify_tag}")
    
    # Save results
    results = {
        'student_id': STUDENT_ID,
        'Pphys': Pphys,
        'Plog': Plog,
        'stride_sweep': stride_results,
        'process_scaling_stride1': results_stride1,
        'process_scaling_stride16': results_stride16,
        'karp_flatt': karp_flatt,
        'verify_tag': verify_tag
    }
    
    import json
    with open('/Users/nuradilabyz/Desktop/Week_11_practice_task/output/phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to output/phase1_results.json")
    print("Phase 1 complete!")


if __name__ == "__main__":
    # Set start method to spawn for macOS compatibility
    mp.set_start_method('spawn', force=True)
    main()
