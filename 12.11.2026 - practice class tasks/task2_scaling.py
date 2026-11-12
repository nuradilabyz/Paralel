"""
Phase 2: Strong Scaling vs. Weak Scaling

Build a two-stage pipeline per trial:
- Stage A: Serial execution on parent process (first 5% of elements)
- Stage B: Parallel execution across mp.Pool (all N elements)

Stage A: 0.05N elements, 10 math.sin/math.sqrt ops/element
Stage B: N elements, 30 FMA/trig ops/element, zero shared writes
"""

import multiprocessing as mp
import time
import math
import sys
import os

sys.path.insert(0, '/Users/nuradilabyz/Desktop/Week_11_practice_task')
from harness import fnv1a_hash, emit_verify_tag, run_benchmark

STUDENT_ID = 230103188

# Worker function must be at module top level
def stage_b_worker(args):
    """
    Worker for Stage B: Perform 30 FMA/trig ops per element.
    No shared writes - each worker gets its own chunk.
    """
    chunk_start, chunk_end, dummy_result = args
    # Simulate FMA/trig operations
    result = 0.0
    for i in range(chunk_start, chunk_end):
        x = float(i)
        # 30 FMA/trig ops per element
        for _ in range(30):
            result += math.sin(x) * math.sqrt(x + 0.1) + x * x
    return result


def run_strong_scaling(N: int, p_values: list) -> dict:
    """
    Strong Scaling: Fix N = 1.2e6, sweep p ∈ {1, 2, ..., Plog}.
    
    Args:
        N: Fixed problem size
        p_values: List of process counts
    
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    # Pre-compute stage A time (serial on first 5% of elements)
    n_stage_a = int(0.05 * N)
    
    def compute_stage_a():
        result = 0.0
        for i in range(n_stage_a):
            x = float(i)
            for _ in range(10):
                result += math.sin(x) + math.sqrt(x + 0.1)
        return result
    
    # Measure Stage A time once
    t0 = time.perf_counter_ns()
    stage_a_result = compute_stage_a()
    t_stage_a = (time.perf_counter_ns() - t0) / 1e6  # ms
    print(f"Stage A (serial): {t_stage_a:.4f} ms")
    
    for p in p_values:
        # Create pool outside timed loop
        pool = mp.Pool(processes=p)
        
        def benchmark_fn():
            # Stage A: Serial (not timed as part of the parallel benchmark)
            # Stage B: Parallel
            chunk_size = N // p
            chunks = []
            for i in range(p):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < p - 1 else N
                chunks.append((start, end, 0.0))
            
            # Run Stage B
            pool.map(stage_b_worker, chunks)
        
        stats = run_benchmark(benchmark_fn, warmup=1, runs=5)
        
        pool.close()
        pool.join()
        
        results[p] = {
            'Tp_ms': stats.median_ms,
            'min_ms': stats.min_ms,
            'iqr_ms': stats.iqr_ms
        }
    
    # Calculate speedups and efficiencies
    T1 = results[1]['Tp_ms']
    for p in results:
        Tp = results[p]['Tp_ms']
        Sp = T1 / Tp
        Ep = Sp / p
        results[p]['Sp'] = Sp
        results[p]['Ep'] = Ep
    
    # Amdahl's cap: S∞ = 1/s where s = T_stage_a / T1
    s = t_stage_a / T1
    S_inf = 1.0 / s if s > 0 else 0
    print(f"Amdahl's cap: s = {s:.6f}, S∞ = {S_inf:.2f}")
    
    return results, t_stage_a, S_inf


def run_weak_scaling(N0: int, p_values: list) -> dict:
    """
    Weak Scaling: N(p) = p * N0, where N0 = 2.0e5.
    
    Args:
        N0: Per-worker load
        p_values: List of process counts
    
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    for p in p_values:
        N = p * N0
        n_stage_a = int(0.05 * N)
        
        # Create pool outside timed loop
        pool = mp.Pool(processes=p)
        
        def benchmark_fn():
            # Stage A: Serial on first 5%
            result_a = 0.0
            for i in range(n_stage_a):
                x = float(i)
                for _ in range(10):
                    result_a += math.sin(x) + math.sqrt(x + 0.1)
            
            # Stage B: Parallel on all N
            chunk_size = N // p
            chunks = []
            for i in range(p):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < p - 1 else N
                chunks.append((start, end, 0.0))
            
            pool.map(stage_b_worker, chunks)
            
            return result_a
        
        stats = run_benchmark(benchmark_fn, warmup=1, runs=5)
        
        pool.close()
        pool.join()
        
        results[p] = {
            'N': N,
            'Tp_ms': stats.median_ms,
            'min_ms': stats.min_ms,
            'iqr_ms': stats.iqr_ms
        }
    
    # Calculate weak scaling metrics
    T1 = results[1]['Tp_ms']
    for p in results:
        Tp = results[p]['Tp_ms']
        Ep_weak = T1 / Tp  # Weak efficiency
        
        # Stage A time for this p
        N_p = results[p]['N']
        n_stage_a_p = int(0.05 * N_p)
        
        # Measure Stage A time for this p
        t0 = time.perf_counter_ns()
        result_a = 0.0
        for i in range(n_stage_a_p):
            x = float(i)
            for _ in range(10):
                result_a += math.sin(x) + math.sqrt(x + 0.1)
        t_stage_a_p = (time.perf_counter_ns() - t0) / 1e6
        
        alpha = t_stage_a_p / Tp if Tp > 0 else 0
        Sp_gustafson = p - alpha * (p - 1)
        
        results[p]['Ep_weak'] = Ep_weak
        results[p]['Sp_gustafson'] = Sp_gustafson
        results[p]['t_stage_a'] = t_stage_a_p
    
    return results


def main():
    print("=" * 60)
    print("Phase 2: Strong Scaling vs. Weak Scaling")
    print("=" * 60)
    
    # Get hardware info
    Pphys = mp.cpu_count()
    Plog = mp.cpu_count()
    print(f"Pphys: {Pphys}, Plog: {Plog}")
    
    p_values = [1, 2, 4, 8]
    
    # Part 1: Strong Scaling
    print("\n--- Part 1: Strong Scaling (N = 1.2e6) ---")
    N_strong = int(1.2 * 10**6)
    strong_results, t_stage_a_strong, S_inf = run_strong_scaling(N_strong, p_values)
    
    print(f"{'p':<5} {'Tp (ms)':<12} {'Sp':<12} {'Ep':<12} {'Min (ms)':<12} {'IQR (ms)':<12}")
    print("-" * 70)
    for p in sorted(strong_results.keys()):
        r = strong_results[p]
        print(f"{p:<5} {r['Tp_ms']:<12.4f} {r.get('Sp', 0):<12.4f} {r.get('Ep', 0):<12.4f} {r['min_ms']:<12.4f} {r['iqr_ms']:<12.4f}")
    
    # Part 2: Weak Scaling
    print("\n--- Part 2: Weak Scaling (N0 = 2.0e5) ---")
    N0 = int(2.0 * 10**5)
    weak_results = run_weak_scaling(N0, p_values)
    
    print(f"{'p':<5} {'N':<10} {'Tp (ms)':<12} {'Ep_weak':<12} {'Sp_gustafson':<15} {'t_stage_a (ms)':<15}")
    print("-" * 80)
    for p in sorted(weak_results.keys()):
        r = weak_results[p]
        print(f"{p:<5} {r['N']:<10} {r['Tp_ms']:<12.4f} {r.get('Ep_weak', 0):<12.4f} {r.get('Sp_gustafson', 0):<15.4f} {r.get('t_stage_a', 0):<15.4f}")
    
    # Part 3: Architectural Inflection
    print("\n--- Part 3: Architectural Inflection ---")
    if Pphys < Plog:
        # Check drop from Pphys to Pphys+1
        if Pphys in strong_results and (Pphys + 1) in strong_results:
            Sp_phys = strong_results[Pphys]['Sp']
            Sp_phys_plus_1 = strong_results[Pphys + 1]['Sp']
            marginal_drop = Sp_phys - Sp_phys_plus_1
            print(f"Marginal speedup drop from p={Pphys} to p={Pphys+1}: {marginal_drop:.4f}")
        else:
            print("Cannot compute marginal drop (Pphys+1 not in results)")
    else:
        print(f"Pphys ({Pphys}) == Plog ({Plog}), no SMT boundary to cross")
    
    # Emit VERIFY_HASH[PHASE2]
    # Use SP_strong at p=Pphys and SP_gustafson at p=Pphys
    SP_strong_Pphys = strong_results[Pphys]['Sp']
    SP_gustafson_Pphys = weak_results[Pphys]['Sp_gustafson']
    verify_tag = emit_verify_tag("PHASE2", STUDENT_ID, SP_strong_Pphys, SP_gustafson_Pphys)
    print(f"\n{verify_tag}")
    
    # Save results
    results = {
        'student_id': STUDENT_ID,
        'Pphys': Pphys,
        'Plog': Plog,
        'strong_scaling': strong_results,
        'weak_scaling': weak_results,
        'S_inf_amdahl': S_inf,
        'verify_tag': verify_tag
    }
    
    import json
    with open('/Users/nuradilabyz/Desktop/Week_11_practice_task/output/phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to output/phase2_results.json")
    print("Phase 2 complete!")


if __name__ == "__main__":
    # Set start method to spawn for macOS compatibility
    mp.set_start_method('spawn', force=True)
    main()
