"""
Phase 3: Empirical Machine Roofline & GEMM Optimization

Simplified version for actual execution on macOS with Apple Silicon.
This version uses multiprocessing.Array instead of shared_memory for better reliability.
"""

import multiprocessing as mp
import numpy as np
import sys
import os
import math
import time

# Add parent directory to path to import harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_benchmark, emit_verify_tag


def stream_triad_benchmark(n: int = 10_000_000) -> tuple:
    """
    Measure DRAM bandwidth using STREAM Triad: a = b + scalar * c
    Uses numpy arrays for efficiency.
    Returns: (bandwidth_gb_s, median_ms)
    """
    # Create arrays
    size = n
    a = np.zeros(size, dtype=np.float64)
    b = np.ones(size, dtype=np.float64)
    c = np.full(size, 2.0, dtype=np.float64)
    
    def benchmark_fn():
        scalar = 2.0
        for _ in range(10):  # Multiple iterations for better measurement
            np.copyto(a, b + scalar * c)
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    
    # Calculate bandwidth: 3 arrays accessed (read b, read c, write a) = 3 * n * 8 bytes
    bytes_accessed = 3 * n * 8 * 10  # 10 iterations
    median_sec = stats.median_ms / 1000.0
    bandwidth_gb_s = bytes_accessed / (1024**3) / median_sec
    
    return bandwidth_gb_s, stats.median_ms


def fma_benchmark(n: int = 1_000_000) -> tuple:
    """
    Measure FMA peak performance.
    Returns: (gflops, median_ms)
    """
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)
    c = np.random.rand(n).astype(np.float64)
    d = np.zeros(n, dtype=np.float64)
    
    def benchmark_fn():
        for _ in range(100):
            # FMA: d = a * b + c
            np.copyto(d, a * b + c)
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    
    # 2 FLOPS per operation (multiply + add), n operations, 100 iterations
    flops = n * 2 * 100
    median_sec = stats.median_ms / 1000.0
    gflops = flops / 10**9 / median_sec
    
    return gflops, stats.median_ms


def gemm_naive(n: int) -> float:
    """Naive GEMM: C = A @ B + C"""
    a = np.random.rand(n, n).astype(np.float64)
    b = np.random.rand(n, n).astype(np.float64)
    c = np.random.rand(n, n).astype(np.float64)
    
    def benchmark_fn():
        c[:] = a @ b + c
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    return stats.median_ms


def gemm_blocked(n: int, bs: int) -> float:
    """Blocked GEMM with tile size bs"""
    a = np.random.rand(n, n).astype(np.float64)
    b = np.random.rand(n, n).astype(np.float64)
    c = np.random.rand(n, n).astype(np.float64)
    
    def benchmark_fn():
        for ii in range(0, n, bs):
            for kk in range(0, n, bs):
                for jj in range(0, n, bs):
                    # Process block
                    i_end = min(ii + bs, n)
                    k_end = min(kk + bs, n)
                    j_end = min(jj + bs, n)
                    for i in range(ii, i_end):
                        for k in range(kk, k_end):
                            for j in range(jj, j_end):
                                c[i, j] += a[i, k] * b[k, j]
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    return stats.median_ms


def main():
    print("=" * 70)
    print("PHASE 3: Empirical Machine Roofline & GEMM Optimization")
    print("=" * 70)
    print()
    
    # Get student ID
    if len(sys.argv) > 1:
        student_id = int(sys.argv[1])
    else:
        student_id = int(input("Enter your Student ID: ").strip())
    
    print(f"Student ID: {student_id}")
    print()
    
    Plog = mp.cpu_count()
    print(f"Logical CPUs: {Plog}")
    print()
    
    # ========================================================================
    # MEASURE DRAM BANDWIDTH
    # ========================================================================
    print("Measuring DRAM Streaming Bandwidth...")
    print("-" * 70)
    
    sizes = [1_000_000, 5_000_000, 10_000_000]
    bw_results = []
    
    for n in sizes:
        print(f"  Testing n = {n:,}...", end=" ")
        try:
            bw, median_ms = stream_triad_benchmark(n)
            bw_results.append((n, bw, median_ms))
            print(f"Done ({bw:.2f} GB/s)")
        except Exception as e:
            print(f"Failed ({e})")
    
    if bw_results:
        bw_sustained = max(bw_results, key=lambda x: x[1])[1]
        print(f"\nSustained DRAM Bandwidth: {bw_sustained:.2f} GB/s")
    else:
        bw_sustained = 50.0  # Default estimate for Apple M2
        print(f"\nUsing default bandwidth estimate: {bw_sustained:.2f} GB/s")
    print()
    
    # ========================================================================
    # MEASURE FMA PEAK
    # ========================================================================
    print("Measuring FMA Peak Performance...")
    print("-" * 70)
    
    sizes = [100_000, 500_000, 1_000_000]
    fma_results = []
    
    for n in sizes:
        print(f"  Testing n = {n:,}...", end=" ")
        try:
            gflops, median_ms = fma_benchmark(n)
            fma_results.append((n, gflops, median_ms))
            print(f"Done ({gflops:.2f} GFLOPS)")
        except Exception as e:
            print(f"Failed ({e})")
    
    if fma_results:
        p_sustained = max(fma_results, key=lambda x: x[1])[1]
        print(f"\nSustained FMA Peak: {p_sustained:.2f} GFLOPS")
    else:
        p_sustained = 10.0  # Default estimate
        print(f"\nUsing default FMA estimate: {p_sustained:.2f} GFLOPS")
    print()
    
    # ========================================================================
    # ROOFLINE MODEL
    # ========================================================================
    print("Empirical Roofline Model")
    print("-" * 70)
    print(f"Bandwidth (BW): {bw_sustained:.2f} GB/s")
    print(f"Peak FLOPS (P):  {p_sustained:.2f} GFLOPS")
    
    # Ridge point: I* = P / (BW * 8) for double precision
    if bw_sustained > 0:
        i_star = p_sustained / (bw_sustained * 8)
        print(f"Ridge Point (I*): {i_star:.4f} FLOPS/byte")
        print()
        print("Roofline Interpretation:")
        print(f"  - For arithmetic intensity < {i_star:.2f} FLOPS/byte: Memory-bound")
        print(f"  - For arithmetic intensity > {i_star:.2f} FLOPS/byte: Compute-bound")
    print()
    
    # ========================================================================
    # GEMM OPTIMIZATION
    # ========================================================================
    print("GEMM Optimization & Tile Size Sweep")
    print("-" * 70)
    
    n = 100  # Smaller matrix for faster execution
    print(f"Matrix size: {n}x{n}")
    print()
    
    tile_sizes = [1, 2, 4, 8, 16, 20, 24, 32]
    
    print("  Testing naive (no blocking)...", end=" ")
    t_naive = gemm_naive(n)
    gflops_naive = (2 * n**3) / (t_naive / 1000) / 10**9
    print(f"Done ({t_naive:.2f} ms, {gflops_naive:.2f} GFLOPS)")
    
    gemm_results = [(0, t_naive, gflops_naive, "naive")]
    
    for bs in tile_sizes:
        print(f"  Testing bs = {bs}...", end=" ")
        try:
            t_blocked = gemm_blocked(n, bs)
            gflops = (2 * n**3) / (t_blocked / 1000) / 10**9
            gemm_results.append((bs, t_blocked, gflops, "blocked"))
            print(f"Done ({t_blocked:.2f} ms, {gflops:.2f} GFLOPS)")
        except Exception as e:
            print(f"Failed ({e})")
    
    print()
    print("GEMM Results:")
    print("-" * 70)
    print(f"{'Tile':<10} {'Time (ms)':<12} {'GFLOPS':<10} {'Speedup':<10}")
    print("-" * 70)
    
    best_gflops = gflops_naive
    best_bs = 0
    
    for bs, t_ms, gflops, label in gemm_results:
        speedup = t_naive / t_ms if t_naive > 0 else 1.0
        if label == "naive":
            print(f"{'naive':<10} {t_ms:<12.2f} {gflops:<10.2f} {1.0:<10.2f}")
        else:
            print(f"{bs:<10} {t_ms:<12.2f} {gflops:<10.2f} {speedup:<10.2f}")
        if gflops > best_gflops:
            best_gflops = gflops
            best_bs = bs
    
    print()
    print(f"Optimal tile size: bs = {best_bs} ({best_gflops:.2f} GFLOPS)")
    print()
    
    # ========================================================================
    # VERIFICATION HASH
    # ========================================================================
    print("VERIFICATION HASH")
    print("-" * 70)
    
    verify_hash = emit_verify_tag(
        "PHASE3",
        student_id,
        bw_sustained,
        p_sustained
    )
    print(verify_hash)
    
    print()
    
    # Save results
    with open(f"phase3_results_{student_id}.txt", "w") as f:
        f.write("PHASE 3 RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Student ID: {student_id}\n")
        f.write(f"Logical CPUs: {Plog}\n\n")
        
        f.write("DRAM Bandwidth\n")
        f.write("-" * 70 + "\n")
        for n, bw, ms in bw_results:
            f.write(f"  n={n:>8,} -> {bw:>6.2f} GB/s ({ms:>6.2f} ms)\n")
        f.write(f"  Sustained BW: {bw_sustained:.2f} GB/s\n\n")
        
        f.write("FMA Peak\n")
        f.write("-" * 70 + "\n")
        for n, gflops, ms in fma_results:
            f.write(f"  n={n:>8,} -> {gflops:>6.2f} GFLOPS ({ms:>6.2f} ms)\n")
        f.write(f"  Sustained Peak: {p_sustained:.2f} GFLOPS\n\n")
        
        f.write("Roofline Model\n")
        f.write("-" * 70 + "\n")
        f.write(f"  BW: {bw_sustained:.2f} GB/s\n")
        f.write(f"  P:  {p_sustained:.2f} GFLOPS\n")
        if bw_sustained > 0:
            i_star = p_sustained / (bw_sustained * 8)
            f.write(f"  I*: {i_star:.4f} FLOPS/byte\n")
        f.write("\n")
        
        f.write("GEMM Results\n")
        f.write("-" * 70 + "\n")
        f.write(f"Matrix size: {n}x{n}\n")
        f.write(f"{'Tile':<10} {'Time (ms)':<12} {'GFLOPS':<10} {'Speedup':<10}\n")
        f.write("-" * 70 + "\n")
        for bs, t_ms, gflops, label in gemm_results:
            speedup = t_naive / t_ms if t_naive > 0 else 1.0
            if label == "naive":
                f.write(f"{'naive':<10} {t_ms:<12.2f} {gflops:<10.2f} {1.0:<10.2f}\n")
            else:
                f.write(f"{bs:<10} {t_ms:<12.2f} {gflops:<10.2f} {speedup:<10.2f}\n")
        f.write("\n")
        f.write(f"Optimal: bs={best_bs}, {best_gflops:.2f} GFLOPS\n\n")
        
        f.write("Verification Hash\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
    
    print(f"\nResults saved to: phase3_results_{student_id}.txt")
    print("\nPHASE 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
