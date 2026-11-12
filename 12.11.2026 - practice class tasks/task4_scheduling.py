"""
Phase 4: Skewed Workloads & Dynamic Chunk Queue

Further simplified version.
"""

import multiprocessing as mp
import time
import math
import sys

sys.path.insert(0, '/Users/nuradilabyz/Desktop/Week_11_practice_task')
from harness import fnv1a_hash, emit_verify_tag

STUDENT_ID = 230103188
N = 8000


def compute_item(i):
    """Compute workload for item i"""
    k = 1 + int((i / 40) ** 2)
    total = 0.0
    for _ in range(k):
        x = float(i)
        total += math.sin(x) + math.cos(x) + math.sqrt(x + 0.1)
    return total


def block_static_worker(start, end):
    """Contiguous Block Static scheduling"""
    start_time = time.perf_counter_ns()
    
    for i in range(start, end):
        compute_item(i)
    
    end_time = time.perf_counter_ns()
    elapsed_ns = end_time - start_time
    return elapsed_ns / 1e6  # ms


def round_robin_worker(items):
    """Cyclic Round-Robin Static scheduling"""
    start_time = time.perf_counter_ns()
    
    for i in items:
        compute_item(i)
    
    end_time = time.perf_counter_ns()
    elapsed_ns = end_time - start_time
    return elapsed_ns / 1e6  # ms


def dynamic_queue_worker_init(index):
    """Initializer for dynamic queue worker"""
    global global_index
    global_index = index


def dynamic_queue_worker(N, C):
    """Dynamic Queue worker with chunking"""
    start_time = time.perf_counter_ns()
    
    while True:
        with global_index.get_lock():
            current = global_index.value
            if current >= N:
                break
            global_index.value = current + C
        
        end = min(current + C, N)
        for i in range(current, end):
            compute_item(i)
    
    end_time = time.perf_counter_ns()
    elapsed_ns = end_time - start_time
    return elapsed_ns / 1e6  # ms


def run_block_static(p: int):
    """Run Contiguous Block Static scheduling"""
    pool = mp.Pool(processes=p)
    
    items_per_worker = N // p
    args_list = []
    for worker_id in range(p):
        start = worker_id * items_per_worker
        end = (worker_id + 1) * items_per_worker if worker_id < p - 1 else N
        args_list.append((start, end))
    
    t0 = time.perf_counter_ns()
    worker_times = pool.starmap(block_static_worker, args_list)
    t1 = time.perf_counter_ns()
    
    wall_time_ms = (t1 - t0) / 1e6
    pool.close()
    pool.join()
    
    return wall_time_ms, worker_times


def run_round_robin(p: int):
    """Run Cyclic Round-Robin Static scheduling"""
    pool = mp.Pool(processes=p)
    
    worker_items = [[] for _ in range(p)]
    for i in range(N):
        worker_items[i % p].append(i)
    
    t0 = time.perf_counter_ns()
    worker_times = pool.map(round_robin_worker, worker_items)
    t1 = time.perf_counter_ns()
    
    wall_time_ms = (t1 - t0) / 1e6
    pool.close()
    pool.join()
    
    return wall_time_ms, worker_times


def run_dynamic_queue(p: int, C: int):
    """Run Dynamic Queue scheduling"""
    # Use shared index via inheritance
    index = mp.Value('i', 0)
    
    pool = mp.Pool(processes=p, initializer=dynamic_queue_worker_init, initargs=(index,))
    
    args_list = [(N, C) for _ in range(p)]
    
    t0 = time.perf_counter_ns()
    worker_times = pool.starmap(dynamic_queue_worker, args_list)
    t1 = time.perf_counter_ns()
    
    wall_time_ms = (t1 - t0) / 1e6
    pool.close()
    pool.join()
    
    return wall_time_ms, worker_times


def compute_load_imbalance_factor(worker_times: list) -> float:
    """Compute Load Imbalance Factor λ"""
    p = len(worker_times)
    if p == 0:
        return 0.0
    
    max_time = max(worker_times)
    avg_time = sum(worker_times) / p
    
    if avg_time == 0:
        return 0.0
    
    lambda_factor = (max_time / avg_time - 1) / (p - 1) if p > 1 else 0.0
    return lambda_factor


def main():
    print("=" * 60)
    print("Phase 4: Skewed Workloads & Dynamic Chunk Queue")
    print("=" * 60)
    
    Pphys = mp.cpu_count()
    print(f"Pphys: {Pphys}, N: {N}")
    
    p = Pphys
    C_values = [1, 8, 64, 512]
    
    # Strategy 1: Contiguous Block Static
    print("\n--- Strategy 1: Contiguous Block Static ---")
    block_wall_time, block_worker_times = run_block_static(p)
    block_lambda = compute_load_imbalance_factor(block_worker_times)
    print(f"Wall time: {block_wall_time:.2f} ms")
    print(f"Worker times: {[f'{t:.2f}' for t in block_worker_times]} ms")
    print(f"Load Imbalance Factor λ: {block_lambda:.4f}")
    
    # Strategy 2: Cyclic Round-Robin Static
    print("\n--- Strategy 2: Cyclic Round-Robin Static ---")
    rr_wall_time, rr_worker_times = run_round_robin(p)
    rr_lambda = compute_load_imbalance_factor(rr_worker_times)
    print(f"Wall time: {rr_wall_time:.2f} ms")
    print(f"Worker times: {[f'{t:.2f}' for t in rr_worker_times]} ms")
    print(f"Load Imbalance Factor λ: {rr_lambda:.4f}")
    
    # Strategy 3: Dynamic Queue with different chunk sizes
    print("\n--- Strategy 3: Dynamic Queue ---")
    dq_results = {}
    for C in C_values:
        wall_time, worker_times = run_dynamic_queue(p, C)
        lambda_factor = compute_load_imbalance_factor(worker_times)
        dq_results[C] = {
            'wall_time_ms': wall_time,
            'worker_times': worker_times,
            'lambda': lambda_factor
        }
        print(f"C={C:4d}: Wall time={wall_time:.2f} ms, λ={lambda_factor:.4f}, Worker times={[f'{t:.2f}' for t in worker_times]}")
    
    # Analysis
    print("\n--- Analysis ---")
    print("Trade-off between C=1 (synchronization contention) and C=512 (tail imbalance):")
    print("- Small C (1, 8): More synchronization overhead, better load balance")
    print("- Large C (512): Less synchronization overhead, but potential tail imbalance")
    
    # Find C with minimum wall time
    best_C = min(C_values, key=lambda c: dq_results[c]['wall_time_ms'])
    best_wall_time = dq_results[best_C]['wall_time_ms']
    print(f"\nOptimal C*: {best_C} with wall time: {best_wall_time:.2f} ms")
    
    # Emit VERIFY_HASH[PHASE4]
    verify_tag = emit_verify_tag("PHASE4", STUDENT_ID, block_wall_time, best_wall_time)
    print(f"\n{verify_tag}")
    
    # Save results
    results = {
        'student_id': STUDENT_ID,
        'Pphys': Pphys,
        'N': N,
        'block_static': {
            'wall_time_ms': block_wall_time,
            'worker_times': block_worker_times,
            'lambda': block_lambda
        },
        'round_robin': {
            'wall_time_ms': rr_wall_time,
            'worker_times': rr_worker_times,
            'lambda': rr_lambda
        },
        'dynamic_queue': dq_results,
        'best_C': best_C,
        'best_wall_time': best_wall_time,
        'verify_tag': verify_tag
    }
    
    import json
    with open('/Users/nuradilabyz/Desktop/Week_11_practice_task/output/phase4_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to output/phase4_results.json")
    print("Phase 4 complete!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
