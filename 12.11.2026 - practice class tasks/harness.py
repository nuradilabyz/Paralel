"""
Minimal Timing and Verification Harness
Phase 0: Hardware Topology & Minimal Telemetry Harness
"""

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Callable


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float


def fnv1a_hash(s: str) -> int:
    """FNV-1a 64-bit hash function"""
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup: int = 2, runs: int = 7) -> BenchStats:
    """
    Run a benchmark function and return statistics.
    
    Args:
        fn: Function to benchmark (no arguments, no return value)
        warmup: Number of warmup runs
        runs: Number of timed runs (must be odd for median calculation)
    
    Returns:
        BenchStats with min, median, and IQR in milliseconds
    """
    # Warmup runs
    for _ in range(warmup):
        fn()
    
    # Timed runs
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        samples.append(elapsed_ms)
    
    samples.sort()
    median = samples[runs // 2]
    q1 = samples[runs // 4]
    q3 = samples[(3 * runs) // 4]
    iqr = q3 - q1
    
    return BenchStats(min_ms=samples[0], median_ms=median, iqr_ms=iqr)


def emit_verify_tag(phase: str, seed: int, m1: float, m2: float) -> str:
    """
    Emit a verification hash tag bound to hardware and measurements.
    
    Args:
        phase: Phase identifier (e.g., "PHASE0")
        seed: Student ID or seed value
        m1: First measurement value
        m2: Second measurement value
    
    Returns:
        Verification hash string
    """
    hw = mp.cpu_count()
    sig = fnv1a_hash(f"{phase}|{seed}|{hw}|{m1:.4f}|{m2:.4f}")
    return f"VERIFY_HASH [{phase}]: {sig:016X} (P_log={hw})"


if __name__ == "__main__":
    # Test the harness
    print("Testing harness.py...")
    print(f"CPU count: {mp.cpu_count()}")
    
    # Test fnv1a_hash
    test_hash = fnv1a_hash("test|123|8|1.234|5.678")
    print(f"Test hash: {test_hash:016X}")
    
    # Test run_benchmark
    def dummy_work():
        x = sum(range(10000))
    
    stats = run_benchmark(dummy_work, warmup=1, runs=5)
    print(f"Benchmark stats: min={stats.min_ms:.4f}ms, median={stats.median_ms:.4f}ms, iqr={stats.iqr_ms:.4f}ms")
    
    # Test emit_verify_tag
    tag = emit_verify_tag("TEST", 230103188, 1.234, 5.678)
    print(f"Verify tag: {tag}")
    
    print("harness.py test complete!")
