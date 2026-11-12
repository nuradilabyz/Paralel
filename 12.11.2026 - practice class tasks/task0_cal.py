"""
Phase 0: Hardware Topology & Minimal Telemetry Harness
Task: Query mp.cpu_count(), measure average nanosecond call latency over 10^6 consecutive
time.perf_counter_ns() calls, and emit VERIFY_HASH[PHASE0] with Student ID as seed.
"""

import multiprocessing as mp
import time
import sys

# Import from harness
sys.path.insert(0, '/Users/nuradilabyz/Desktop/Week_11_practice_task')
from harness import fnv1a_hash, emit_verify_tag

STUDENT_ID = 230103188

def measure_perf_counter_latency(num_calls: int = 10**6) -> float:
    """
    Measure average latency of time.perf_counter_ns() calls.
    
    Args:
        num_calls: Number of consecutive calls to measure
    
    Returns:
        Average nanoseconds per call
    """
    # Warmup
    for _ in range(100):
        time.perf_counter_ns()
    
    # Measure
    start = time.perf_counter_ns()
    for _ in range(num_calls):
        time.perf_counter_ns()
    end = time.perf_counter_ns()
    
    total_ns = end - start
    avg_ns = total_ns / num_calls
    return avg_ns


def main():
    print("=" * 60)
    print("Phase 0: Hardware Topology & Minimal Telemetry")
    print("=" * 60)
    
    # Get hardware info
    cpu_count = mp.cpu_count()
    print(f"CPU Count (logical): {cpu_count}")
    
    # Measure perf_counter_ns latency
    avg_latency_ns = measure_perf_counter_latency()
    avg_latency_ms = avg_latency_ns / 1e6
    print(f"Average time.perf_counter_ns() latency: {avg_latency_ns:.2f} ns ({avg_latency_ms:.6f} ms)")
    
    # Emit verification tag
    # For PHASE0, we use the latency as both m1 and m2 (or we can use cpu_count and latency)
    # Based on the PDF: "emit VERIFY_HASH[PHASE0] with your Student ID as seed"
    # The function signature is: emit_verify_tag(phase, seed, m1, m2)
    # We'll use cpu_count as m1 and avg_latency_ms as m2
    verify_tag = emit_verify_tag("PHASE0", STUDENT_ID, float(cpu_count), avg_latency_ms)
    print(f"\n{verify_tag}")
    
    # Save results
    results = {
        "student_id": STUDENT_ID,
        "cpu_count": cpu_count,
        "avg_latency_ns": avg_latency_ns,
        "avg_latency_ms": avg_latency_ms,
        "verify_tag": verify_tag
    }
    
    # Write to output file
    with open('/Users/nuradilabyz/Desktop/Week_11_practice_task/output/phase0_results.txt', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to output/phase0_results.txt")
    print("Phase 0 complete!")


if __name__ == "__main__":
    main()
