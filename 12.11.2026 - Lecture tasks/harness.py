from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Callable


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float


def fnv1a_hash(s: str) -> int:
    """FNV-1a 64-bit hash algorithm for verification checksums."""
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup_runs: int = 3, timed_runs: int = 9) -> BenchStats:
    """
    Execute a benchmark function with warmup and timed runs.
    Returns BenchStats containing min, median, IQR, and max execution times in milliseconds.
    """
    # Warmup runs
    for _ in range(warmup_runs):
        fn()
    
    # Timed runs
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)  # Convert to milliseconds
    
    samples.sort()
    mid = timed_runs // 2
    med = samples[mid] if timed_runs % 2 == 1 else 0.5 * (samples[mid - 1] + samples[mid])
    q25 = samples[timed_runs // 4]
    q75 = samples[(3 * timed_runs) // 4]
    
    return BenchStats(samples[0], med, q75 - q25, samples[-1])


def emit_verify_tag(task_id: str, student_seed: int, primary_metric: float, secondary_metric: float) -> str:
    """
    Generate a cryptographic verification hash for a task.
    
    Args:
        task_id: Identifier for the task (e.g., "PHASE0")
        student_seed: Student ID or seed value
        primary_metric: Primary measurement value
        secondary_metric: Secondary measurement value
    
    Returns:
        VERIFY_HASH string for authentication
    """
    hw = mp.cpu_count()
    payload = (
        f"{task_id}|{student_seed}|{hw}|"
        f"{primary_metric:.4f}|{secondary_metric:.4f}"
    )
    sig = fnv1a_hash(payload)
    return f"VERIFY_HASH[{task_id}]: {sig:016X} (P_log={hw})"


def get_cpu_topology() -> dict:
    """
    Get CPU topology information.
    Returns a dictionary with hardware parameters.
    """
    import platform
    import subprocess
    
    topology = {
        'logical_cpus': mp.cpu_count(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }
    
    system = platform.system()
    
    if system == "Linux":
        try:
            result = subprocess.run(
                ['lscpu'],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.split('\n'):
                if 'Model name' in line:
                    topology['cpu_model'] = line.split(':')[1].strip()
                elif 'Socket(s)' in line:
                    topology['sockets'] = int(line.split(':')[1].strip())
                elif 'Core(s) per socket' in line:
                    topology['cores_per_socket'] = int(line.split(':')[1].strip())
                elif 'Thread(s) per core' in line:
                    topology['threads_per_core'] = int(line.split(':')[1].strip())
                elif 'L1d cache' in line:
                    topology['l1d_cache'] = line.split(':')[1].strip()
                elif 'L2 cache' in line:
                    topology['l2_cache'] = line.split(':')[1].strip()
                elif 'L3 cache' in line:
                    topology['l3_cache'] = line.split(':')[1].strip()
            
            # Get cache line size
            result = subprocess.run(
                ['getconf', 'LEVEL1_DCACHE_LINESIZE'],
                capture_output=True, text=True, check=True
            )
            topology['cache_line_size'] = int(result.stdout.strip())
        except Exception as e:
            print(f"Warning: Could not get Linux topology: {e}")
    
    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, check=True
            )
            topology['cpu_model'] = result.stdout.strip()
            
            result = subprocess.run(
                ['sysctl', 'hw.physicalcpu', 'hw.logicalcpu', 'hw.cachelinesize',
                 'hw.l1dcachesize', 'hw.l2cachesize', 'hw.l3cachesize'],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.strip().split('\n'):
                if 'hw.physicalcpu' in line:
                    topology['physical_cpus'] = int(line.split(':')[1].strip())
                elif 'hw.logicalcpu' in line:
                    topology['logical_cpus'] = int(line.split(':')[1].strip())
                elif 'hw.cachelinesize' in line:
                    topology['cache_line_size'] = int(line.split(':')[1].strip())
                elif 'hw.l1dcachesize' in line:
                    topology['l1d_cache'] = int(line.split(':')[1].strip())
                elif 'hw.l2cachesize' in line:
                    topology['l2_cache'] = int(line.split(':')[1].strip())
                elif 'hw.l3cachesize' in line:
                    topology['l3_cache'] = int(line.split(':')[1].strip())
        except Exception as e:
            print(f"Warning: Could not get macOS topology: {e}")
    
    else:  # Windows or other
        topology['cpu_model'] = platform.processor()
        topology['cache_line_size'] = 64  # Default assumption
    
    # Calculate physical cores if not already set
    if 'physical_cpus' not in topology:
        topology['physical_cpus'] = topology.get('logical_cpus', mp.cpu_count())
    
    return topology


if __name__ == "__main__":
    # Example usage
    print("CPU Topology:")
    topo = get_cpu_topology()
    for key, value in topo.items():
        print(f"  {key}: {value}")
    
    print("\nBenchmark example:")
    def example_fn():
        sum(range(100000))
    
    stats = run_benchmark(example_fn)
    print(f"  Min: {stats.min_ms:.4f} ms")
    print(f"  Median: {stats.median_ms:.4f} ms")
    print(f"  IQR: {stats.iqr_ms:.4f} ms")
    print(f"  Max: {stats.max_ms:.4f} ms")
    
    print("\nVerification hash example:")
    print(emit_verify_tag("PHASE0", 12345, stats.median_ms, stats.iqr_ms))
