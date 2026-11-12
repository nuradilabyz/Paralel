"""
Phase 3: Empirical Machine Roofline & GEMM Optimization

This script performs:
1. DRAM streaming bandwidth measurement
2. FMA peak measurement
3. Machine roofline construction
4. 3-stage GEMM optimization
5. Tile size sweep

The roofline model helps understand the performance limits of a system
based on memory bandwidth and compute capabilities.
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


def stream_triad_worker(shm_name: str, n: int) -> None:
    """
    STREAM Triad: a[i] = b[i] + scalar * c[i]
    Measures sustained memory bandwidth.
    """
    shm_a = shared_memory.SharedMemory(name=shm_name + "_a")
    shm_b = shared_memory.SharedMemory(name=shm_name + "_b")
    shm_c = shared_memory.SharedMemory(name=shm_name + "_c")
    
    a = (ctypes.c_double * n).from_buffer(shm_a.buf)
    b = (ctypes.c_double * n).from_buffer(shm_b.buf)
    c = (ctypes.c_double * n).from_buffer(shm_c.buf)
    
    scalar = 2.0
    for i in range(n):
        a[i] = b[i] + scalar * c[i]
    
    del a, b, c
    shm_a.close()
    shm_b.close()
    shm_c.close()


def measure_bandwidth(n: int = 10_000_000) -> float:
    """
    Measure sustained DRAM bandwidth using STREAM Triad.
    Returns bandwidth in GB/s.
    """
    size_bytes = n * 8 * 3  # 3 arrays of n doubles
    
    def benchmark_fn():
        shm_a = shared_memory.SharedMemory(create=True, size=n * 8)
        shm_b = shared_memory.SharedMemory(create=True, size=n * 8)
        shm_c = shared_memory.SharedMemory(create=True, size=n * 8)
        
        # Initialize arrays
        a = (ctypes.c_double * n).from_buffer(shm_a.buf)
        b = (ctypes.c_double * n).from_buffer(shm_b.buf)
        c = (ctypes.c_double * n).from_buffer(shm_c.buf)
        
        for i in range(n):
            b[i] = 1.0
            c[i] = 2.0
            a[i] = 0.0
        
        del a, b, c
        
        # Create and start worker
        proc = mp.Process(
            target=stream_triad_worker,
            args=(shm_a.name, n)
        )
        proc.start()
        proc.join()
        
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()
        shm_c.close()
        shm_c.unlink()
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    
    # Calculate bandwidth
    # STREAM Triad: 3 arrays read/write = 3 * n * 8 bytes
    bytes_accessed = 3 * n * 8
    median_sec = stats.median_ms / 1000.0
    bandwidth_gb_s = bytes_accessed / (1024**3) / median_sec
    
    return bandwidth_gb_s, stats.median_ms


def fma_worker(shm_name: str, n: int) -> None:
    """
    FMA (Fused Multiply-Add) intensive workload.
    a[i] = b[i] * c[i] + d[i]
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    data = (ctypes.c_double * (n * 4)).from_buffer(shm.buf)
    
    for i in range(n):
        # FMA: a = b * c + d
        idx_a = i * 4
        idx_b = i * 4 + 1
        idx_c = i * 4 + 2
        idx_d = i * 4 + 3
        data[idx_a] = data[idx_b] * data[idx_c] + data[idx_d]
    
    del data
    shm.close()


def measure_fma_peak(n: int = 1_000_000) -> float:
    """
    Measure FMA peak performance.
    Returns GFLOPS/s.
    """
    def benchmark_fn():
        shm = shared_memory.SharedMemory(create=True, size=n * 4 * 8)
        
        # Initialize
        data = (ctypes.c_double * (n * 4)).from_buffer(shm.buf)
        for i in range(n * 4):
            data[i] = float(i)
        
        del data
        
        proc = mp.Process(
            target=fma_worker,
            args=(shm.name, n)
        )
        proc.start()
        proc.join()
        
        shm.close()
        shm.unlink()
    
    stats = run_benchmark(benchmark_fn, warmup_runs=2, timed_runs=7)
    
    # Calculate GFLOPS: 2 FLOPS per FMA (multiply + add)
    flops = n * 2
    median_sec = stats.median_ms / 1000.0
    gflops = flops / (10**9) / median_sec
    
    return gflops, stats.median_ms


def gemm_naive(a, b, c, n: int) -> None:
    """Naive GEMM: C = A * B + C"""
    for i in range(n):
        for k in range(n):
            for j in range(n):
                c[i * n + j] += a[i * n + k] * b[k * n + j]


def gemm_blocked(a, b, c, n: int, bs: int) -> None:
    """Blocked GEMM with tile size bs"""
    for ii in range(0, n, bs):
        for kk in range(0, n, bs):
            for jj in range(0, n, bs):
                # Process block
                for i in range(ii, min(ii + bs, n)):
                    for k in range(kk, min(kk + bs, n)):
                        for j in range(jj, min(jj + bs, n)):
                            c[i * n + j] += a[i * n + k] * b[k * n + j]


def gemm_worker(shm_name: str, n: int, bs: int, use_blocked: bool) -> None:
    """GEMM worker using shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    size = n * n * 3  # A, B, C matrices
    data = (ctypes.c_double * size).from_buffer(shm.buf)
    
    a = data[0:n*n]
    b = data[n*n:2*n*n]
    c = data[2*n*n:3*n*n]
    
    if use_blocked:
        gemm_blocked(a, b, c, n, bs)
    else:
        gemm_naive(a, b, c, n)
    
    del data
    shm.close()


def run_gemm(n: int, bs: int = 0) -> float:
    """
    Run GEMM benchmark.
    bs = 0 means use naive implementation.
    Returns median execution time in ms.
    """
    use_blocked = bs > 0
    
    def benchmark_fn():
        shm = shared_memory.SharedMemory(create=True, size=n * n * 3 * 8)
        
        # Initialize
        data = (ctypes.c_double * (n * n * 3)).from_buffer(shm.buf)
        for i in range(n * n * 3):
            data[i] = float(i % 100) / 100.0
        
        del data
        
        proc = mp.Process(
            target=gemm_worker,
            args=(shm.name, n, bs, use_blocked)
        )
        proc.start()
        proc.join()
        
        shm.close()
        shm.unlink()
    
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
    
    # ========================================================================
    # MEASURE DRAM BANDWIDTH
    # ========================================================================
    print("Measuring DRAM Streaming Bandwidth...")
    print("-" * 70)
    
    # Try different sizes to find peak
    sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000]
    bw_results = []
    
    for n in sizes:
        print(f"  Testing n = {n:,}...", end=" ")
        bw, median_ms = measure_bandwidth(n)
        bw_results.append((n, bw, median_ms))
        print(f"Done ({bw:.2f} GB/s)")
    
    # Use largest size for sustained bandwidth
    bw_sustained = max(bw_results, key=lambda x: x[1])[1]
    print(f"\nSustained DRAM Bandwidth: {bw_sustained:.2f} GB/s")
    print()
    
    # ========================================================================
    # MEASURE FMA PEAK
    # ========================================================================
    print("Measuring FMA Peak Performance...")
    print("-" * 70)
    
    # Try different sizes
    fma_sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    fma_results = []
    
    for n in fma_sizes:
        print(f"  Testing n = {n:,}...", end=" ")
        try:
            gflops, median_ms = measure_fma_peak(n)
            fma_results.append((n, gflops, median_ms))
            print(f"Done ({gflops:.2f} GFLOPS)")
        except Exception as e:
            print(f"Failed ({e})")
    
    # Use best result
    if fma_results:
        p_sustained = max(fma_results, key=lambda x: x[1])[1]
        print(f"\nSustained FMA Peak: {p_sustained:.2f} GFLOPS")
    else:
        p_sustained = 0
        print("\nCould not measure FMA peak")
    print()
    
    # ========================================================================
    # ROOFLINE MODEL
    # ========================================================================
    print("Empirical Roofline Model")
    print("-" * 70)
    
    if bw_sustained > 0 and p_sustained > 0:
        print(f"Bandwidth (BW): {bw_sustained:.2f} GB/s")
        print(f"Peak FLOPS (P):  {p_sustained:.2f} GFLOPS")
        print()
        print("Ridge Point (I*):")
        # I* = P / (3 * BW) for double precision (8 bytes)
        # For STREAM Triad: 3 arrays accessed = 24 bytes per 2 FLOPS
        # But for matrix operations, typically use 8 bytes per element
        i_star = p_sustained / (bw_sustained * 1024**3 / 8)  # 8 bytes per element
        print(f"  I* = P / (BW * 8) = {p_sustained:.2f} / ({bw_sustained:.2f} * 8) = {i_star:.4f} FLOPS/byte")
        print()
        print("Roofline Interpretation:")
        print(f"  - For arithmetic intensity < {i_star:.2f} FLOPS/byte: Memory-bound")
        print(f"  - For arithmetic intensity > {i_star:.2f} FLOPS/byte: Compute-bound")
    else:
        print("Could not compute roofline (missing measurements)")
    
    print()
    
    # ========================================================================
    # GEMM OPTIMIZATION
    # ========================================================================
    print("GEMM Optimization & Tile Size Sweep")
    print("-" * 70)
    
    # Matrix size
    n = 500  # 500x500 matrices
    print(f"Matrix size: {n}x{n}")
    print()
    
    # Tile sizes to test
    tile_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
    
    print("Running GEMM with different tile sizes...")
    
    gemm_results = []
    
    # Naive first
    print("  Testing naive (no blocking)...", end=" ")
    t_naive = run_gemm(n, 0)
    gemm_results.append((0, t_naive, 0, "naive"))
    print(f"Done ({t_naive:.2f} ms)")
    
    # Blocked versions
    for bs in tile_sizes:
        print(f"  Testing bs = {bs}...", end=" ")
        try:
            t_blocked = run_gemm(n, bs)
            gemm_results.append((bs, t_blocked, bs * bs * 8, "blocked"))  # footprint in bytes
            print(f"Done ({t_blocked:.2f} ms)")
        except Exception as e:
            print(f"Failed ({e})")
    
    print()
    
    # Display results
    print("GEMM Results:")
    print("-" * 70)
    print(f"{'Tile Size':<10} {'Time (ms)':<12} {'Speedup':<10} {'Footprint (B)':<15} {'GFLOPS':<10}")
    print("-" * 70)
    
    gflops_naive = (2 * n**3) / (t_naive / 1000) / 10**9
    best_time = t_naive
    
    for bs, t_ms, footprint, label in gemm_results:
        speedup = t_naive / t_ms if t_naive > 0 else 1.0
        gflops = (2 * n**3) / (t_ms / 1000) / 10**9
        gemm_results_with_calc = (bs, t_ms, speedup, footprint, gflops, label)
        
        if label == "naive":
            print(f"{'naive':<10} {t_ms:<12.4f} {speedup:<10.4f} {'N/A':<15} {gflops:<10.2f}")
        else:
            print(f"{bs:<10} {t_ms:<12.4f} {speedup:<10.4f} {footprint:<15,} {gflops:<10.2f}")
    
    print()
    
    # Find optimal tile size
    if len(gemm_results) > 1:
        best_result = min(gemm_results[1:], key=lambda x: x[1])  # Skip naive
        print(f"Optimal tile size: bs = {best_result[0]}")
        print(f"  Time: {best_result[1]:.2f} ms")
        print(f"  Speedup over naive: {t_naive / best_result[1]:.2f}x")
        print(f"  Peak GFLOPS: {gflops_naive * (t_naive / best_result[1]):.2f}")
    
    print()
    
    # ========================================================================
    # VERIFICATION HASH
    # ========================================================================
    print("VERIFICATION HASH")
    print("-" * 70)
    
    # Use bandwidth and peak FLOPS for verification
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
        f.write(f"Student ID: {student_id}\n\n")
        
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
        if bw_sustained > 0 and p_sustained > 0:
            i_star = p_sustained / (bw_sustained * 1024**3 / 8)
            f.write(f"  BW: {bw_sustained:.2f} GB/s\n")
            f.write(f"  P:  {p_sustained:.2f} GFLOPS\n")
            f.write(f"  I*: {i_star:.4f} FLOPS/byte\n")
        f.write("\n")
        
        f.write("GEMM Results\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Tile':<10} {'Time (ms)':<12} {'Speedup':<10} {'Footprint':<15} {'GFLOPS':<10}\n")
        f.write("-" * 70 + "\n")
        for bs, t_ms, speedup, footprint, gflops, label in [(r[0], r[1], t_naive/r[1] if r[1]>0 else 1, r[2], (2*n**3)/(r[1]/1000)/1e9, r[3]) for r in gemm_results]:
            if label == "naive":
                f.write(f"{'naive':<10} {t_ms:<12.4f} {1.0:<10.4f} {'N/A':<15} {gflops:<10.2f}\n")
            else:
                f.write(f"{bs:<10} {t_ms:<12.4f} {speedup:<10.4f} {footprint:<15,} {gflops:<10.2f}\n")
        f.write("\n")
        
        f.write("Verification Hash\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
    
    print(f"\nResults saved to: phase3_results_{student_id}.txt")
    print("\nPHASE 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
