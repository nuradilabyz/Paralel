"""
Phase 3: Empirical Roofline & Cache-Blocked GEMM

Simplified version using numpy arrays and multiprocessing.Pool
All worker functions at module level.
"""

import multiprocessing as mp
import numpy as np
import time
import sys

sys.path.insert(0, '/Users/nuradilabyz/Desktop/Week_11_practice_task')
from harness import fnv1a_hash, emit_verify_tag

STUDENT_ID = 230103188

# Worker functions must be at module level
def fma_worker(num_ops):
    """Perform FMA operations"""
    result = 0.0
    for _ in range(num_ops // 16):
        a1, a2, a3, a4 = 1.0, 2.0, 3.0, 4.0
        b1, b2, b3, b4 = 5.0, 6.0, 7.0, 8.0
        c1, c2, c3, c4 = 9.0, 10.0, 11.0, 12.0
        
        result += a1 * b1 + c1
        result += a2 * b2 + c2
        result += a3 * b3 + c3
        result += a4 * b4 + c4
        result += a1 * b2 + c3
        result += a2 * b3 + c4
        result += a3 * b4 + c1
        result += a4 * b1 + c2
        result += a1 * b3 + c1
        result += a2 * b4 + c2
        result += a3 * b1 + c3
        result += a4 * b2 + c4
        result += a1 * b4 + c1
        result += a2 * b1 + c2
        result += a3 * b2 + c3
        result += a4 * b3 + c4
    return result


def measure_dram_bandwidth(p: int) -> tuple:
    """Measure DRAM bandwidth using parallel numpy array copies."""
    size_mb = 64
    size_bytes = size_mb * 1024 * 1024
    
    B = np.random.rand(size_bytes // 8).astype(np.float64)
    
    times = []
    for _ in range(5):
        C = np.zeros_like(B)
        
        t0 = time.perf_counter_ns()
        
        if p == 1:
            C[:] = B[:]
        else:
            chunk_size = len(B) // p
            for i in range(p):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < p - 1 else len(B)
                C[start:end] = B[start:end]
        
        t1 = time.perf_counter_ns()
        elapsed_ms = (t1 - t0) / 1e6
        times.append(elapsed_ms)
    
    times.sort()
    median_ms = times[2]
    
    total_bytes = size_bytes * 2
    bw_gbps = (total_bytes / 1e9) / (median_ms / 1000)
    
    return bw_gbps, median_ms


def measure_compute_ceiling(p: int) -> tuple:
    """Measure compute ceiling using FMA operations."""
    flops_per_worker = int(6.0 * 10**7)
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        
        if p == 1:
            fma_worker(flops_per_worker)
        else:
            pool = mp.Pool(processes=p)
            pool.map(fma_worker, [flops_per_worker // p] * p)
            pool.close()
            pool.join()
        
        t1 = time.perf_counter_ns()
        elapsed_ms = (t1 - t0) / 1e6
        times.append(elapsed_ms)
    
    times.sort()
    median_ms = times[2]
    
    total_flops = p * flops_per_worker
    gflops = (total_flops / 1e9) / (median_ms / 1000)
    
    return gflops, median_ms


def gemm_variant1_worker(args):
    """Variant 1: Naive i-j-k"""
    N, start_i, end_i = args
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)
    
    for ii in range(start_i, end_i):
        for j in range(N):
            sum_val = 0.0
            for k in range(N):
                sum_val += A[ii, k] * B[k, j]
            C[ii, j] = sum_val
    
    return C


def gemm_variant2_worker(args):
    """Variant 2: Reordered i-k-j"""
    N, start_i, end_i = args
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)
    
    for ii in range(start_i, end_i):
        for k in range(N):
            a_ik = A[ii, k]
            for j in range(N):
                C[ii, j] += a_ik * B[k, j]
    
    return C


def gemm_variant3_worker(args):
    """Variant 3: Cache-Blocked"""
    N, start_i, end_i, BS = args
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)
    
    for ii in range(start_i, end_i, BS):
        for kk in range(0, N, BS):
            for jj in range(0, N, BS):
                for i in range(ii, min(ii + BS, end_i)):
                    for k in range(kk, min(kk + BS, N)):
                        a_ik = A[i, k]
                        for j in range(jj, min(jj + BS, N)):
                            C[i, j] += a_ik * B[k, j]
    
    return C


def run_gemm_variant(variant: int, N: int, p: int, BS: int = None):
    """Run a GEMM variant and measure performance."""
    
    times = []
    for _ in range(3):  # Fewer runs to save time
        t0 = time.perf_counter_ns()
        
        pool = mp.Pool(processes=p)
        
        rows_per_worker = N // p
        args_list = []
        for i in range(p):
            start_i = i * rows_per_worker
            end_i = (i + 1) * rows_per_worker if i < p - 1 else N
            if variant == 1:
                args_list.append((N, start_i, end_i))
            elif variant == 2:
                args_list.append((N, start_i, end_i))
            elif variant == 3:
                args_list.append((N, start_i, end_i, BS))
        
        pool.map(gemm_variant1_worker if variant == 1 else (gemm_variant2_worker if variant == 2 else gemm_variant3_worker), args_list)
        pool.close()
        pool.join()
        
        t1 = time.perf_counter_ns()
        elapsed_ms = (t1 - t0) / 1e6
        times.append(elapsed_ms)
    
    times.sort()
    median_ms = times[1]  # Use middle value
    
    total_flops = 2 * (N ** 3)
    gflops = (total_flops / 1e9) / (median_ms / 1000)
    
    return median_ms, gflops


def main():
    print("=" * 60)
    print("Phase 3: Empirical Roofline & Cache-Blocked GEMM")
    print("=" * 60)
    
    Pphys = mp.cpu_count()
    Plog = mp.cpu_count()
    print(f"Pphys: {Pphys}, Plog: {Plog}")
    
    p = Pphys
    
    # Part 3A: Roofline Ceilings
    print("\n--- Part 3A: Laptop Roofline Ceilings ---")
    
    print("Measuring DRAM Bandwidth...")
    bw_gbps, bw_time = measure_dram_bandwidth(p)
    print(f"DRAM Bandwidth: {bw_gbps:.2f} GB/s (median time: {bw_time:.2f} ms)")
    
    print("\nMeasuring Compute Ceiling...")
    gflops_peak, compute_time = measure_compute_ceiling(p)
    print(f"Compute Ceiling: {gflops_peak:.2f} GFLOP/s (median time: {compute_time:.2f} ms)")
    
    I_star = gflops_peak / bw_gbps
    print(f"Ridge Point I*: {I_star:.2f} FLOP/byte")
    
    # Part 3B: GEMM Variants
    print("\n--- Part 3B: GEMM Variants (N = 128, p = Pphys) ---")
    
    N = 128
    
    print("\nVariant 1: Naive i-j-k")
    v1_time, v1_gflops = run_gemm_variant(1, N, p)
    print(f"GFLOP/s: {v1_gflops:.2f} (time: {v1_time:.2f} ms)")
    
    print("\nVariant 2: Reordered i-k-j")
    v2_time, v2_gflops = run_gemm_variant(2, N, p)
    print(f"GFLOP/s: {v2_gflops:.2f} (time: {v2_time:.2f} ms)")
    
    print("\nVariant 3: Cache-Blocked")
    BS_values = [8, 16, 32]
    v3_results = {}
    best_gflops = 0
    best_BS = None
    
    for BS in BS_values:
        v3_time, v3_gflops = run_gemm_variant(3, N, p, BS)
        v3_results[BS] = {'time_ms': v3_time, 'gflops': v3_gflops}
        print(f"BS={BS:2d}: GFLOP/s={v3_gflops:.2f} (time={v3_time:.2f} ms)")
        
        if v3_gflops > best_gflops:
            best_gflops = v3_gflops
            best_BS = BS
    
    if best_BS is None:
        best_BS = BS_values[0]
        best_gflops = v3_results[best_BS]['gflops']
    
    print(f"\nBest BS: {best_BS} with {best_gflops:.2f} GFLOP/s")
    
    # Tile footprint and operational intensity
    for BS in BS_values:
        Wtile = 3 * BS * 8
        I_tiled = BS / 12
        print(f"BS={BS:2d}: Wtile={Wtile/1024:.2f} KB, I_tiled={I_tiled:.2f} FLOP/byte")
    
    # Emit VERIFY_HASH[PHASE3]
    verify_tag = emit_verify_tag("PHASE3", STUDENT_ID, v1_gflops, v3_results[best_BS]['gflops'])
    print(f"\n{verify_tag}")
    
    # Save results
    results = {
        'student_id': STUDENT_ID,
        'Pphys': Pphys,
        'Plog': Plog,
        'dram_bandwidth_gbps': bw_gbps,
        'dram_bandwidth_time_ms': bw_time,
        'compute_ceiling_gflops': gflops_peak,
        'compute_ceiling_time_ms': compute_time,
        'ridge_point_I_star': I_star,
        'N': N,
        'gemm_variant1_gflops': v1_gflops,
        'gemm_variant1_time_ms': v1_time,
        'gemm_variant2_gflops': v2_gflops,
        'gemm_variant2_time_ms': v2_time,
        'gemm_variant3_results': v3_results,
        'best_BS': best_BS,
        'best_gflops': best_gflops,
        'verify_tag': verify_tag
    }
    
    import json
    with open('/Users/nuradilabyz/Desktop/Week_11_practice_task/output/phase3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to output/phase3_results.json")
    print("Phase 3 complete!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
