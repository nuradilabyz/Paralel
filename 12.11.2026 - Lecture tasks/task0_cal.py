"""
Phase 0: Hardware Topology Audit & Harness Calibration

This script performs:
1. CPU topology interrogation
2. Timer noise-floor calibration
3. Verification hash generation for Phase 0

Run this script to gather baseline hardware information and generate
VERIFY_HASH[PHASE0] for your lab report.
"""

import multiprocessing as mp
import time
import platform
import subprocess
import sys
import os

# Add parent directory to path to import harness
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import fnv1a_hash, emit_verify_tag, get_cpu_topology


def measure_timer_noise():
    """Measure timer noise floor by calling time.perf_counter_ns() repeatedly."""
    num_samples = 2_000_000
    
    print(f"Measuring timer noise floor with {num_samples:,} samples...")
    
    samples = []
    for _ in range(num_samples):
        t0 = time.perf_counter_ns()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    
    samples.sort()
    
    # Calculate statistics
    min_ns = samples[0]
    max_ns = samples[-1]
    mid = num_samples // 2
    median_ns = samples[mid] if num_samples % 2 == 1 else 0.5 * (samples[mid - 1] + samples[mid])
    
    # Calculate mean
    total = sum(samples)
    mean_ns = total / num_samples
    
    # Calculate standard deviation
    variance = sum((x - mean_ns) ** 2 for x in samples) / num_samples
    std_ns = variance ** 0.5
    
    print(f"\nTimer Noise Floor Results:")
    print(f"  Min: {min_ns:.2f} ns")
    print(f"  Max: {max_ns:.2f} ns")
    print(f"  Median: {median_ns:.2f} ns")
    print(f"  Mean: {mean_ns:.2f} ns")
    print(f"  Std Dev: {std_ns:.2f} ns")
    
    return median_ns, std_ns


def main():
    print("=" * 70)
    print("PHASE 0: Hardware Topology Audit & Harness Calibration")
    print("=" * 70)
    print()
    
    # Get student ID from command line or prompt
    if len(sys.argv) > 1:
        student_id = sys.argv[1]
    else:
        student_id = input("Enter your Student ID: ").strip()
    
    print(f"\nStudent ID: {student_id}")
    print()
    
    # Get CPU topology
    print("Interrogating CPU topology...")
    topology = get_cpu_topology()
    
    print("\n" + "-" * 70)
    print("HARDWARE TOPOLOGY")
    print("-" * 70)
    
    # Display in table format
    hardware_params = {
        "CPU Model String": topology.get('cpu_model', 'Unknown'),
        "Physical Cores (Pphys)": topology.get('physical_cpus', topology.get('logical_cpus', mp.cpu_count())),
        "Logical Threads (Plog)": topology.get('logical_cpus', mp.cpu_count()),
        "L1 Data Cache (per core)": topology.get('l1d_cache', 'Unknown'),
        "L2 Cache Size": topology.get('l2_cache', 'Unknown'),
        "L3 Cache Size (Shared)": topology.get('l3_cache', 'Unknown'),
        "Cache Line Size (Lbytes)": topology.get('cache_line_size', 64),
        "Python Version & OS": f"{platform.python_version()} / {platform.system()} {platform.release()}",
    }
    
    for key, value in hardware_params.items():
        print(f"{key:35} {value}")
    
    print()
    
    # Measure timer noise
    median_ns, std_ns = measure_timer_noise()
    
    # Get CPU count for verification
    cpu_count = mp.cpu_count()
    
    print("\n" + "-" * 70)
    print("VERIFICATION HASH")
    print("-" * 70)
    
    # Generate VERIFY_HASH[PHASE0]
    verify_hash = emit_verify_tag(
        "PHASE0",
        int(student_id),
        median_ns,  # primary_metric: average call latency
        std_ns       # secondary_metric: noise floor std dev
    )
    print(verify_hash)
    
    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    print("\nCopy the above information to your lab report.")
    print(f"Student ID: {student_id}")
    print(f"P_log: {cpu_count}")
    print(f"Timer median: {median_ns:.2f} ns")
    print(f"Timer std dev: {std_ns:.2f} ns")
    
    # Save to file for reference
    with open(f"phase0_results_{student_id}.txt", "w") as f:
        f.write("PHASE 0 RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Student ID: {student_id}\n\n")
        f.write("HARDWARE TOPOLOGY\n")
        f.write("-" * 70 + "\n")
        for key, value in hardware_params.items():
            f.write(f"{key:35} {value}\n")
        f.write("\n")
        f.write("TIMER NOISE FLOOR\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Min:':35} {min_ns:.2f} ns\n")
        f.write(f"{'Max:':35} {max_ns:.2f} ns\n")
        f.write(f"{'Median:':35} {median_ns:.2f} ns\n")
        f.write(f"{'Mean:':35} {mean_ns:.2f} ns\n")
        f.write(f"{'Std Dev:':35} {std_ns:.2f} ns\n")
        f.write("\n")
        f.write("VERIFICATION HASH\n")
        f.write("-" * 70 + "\n")
        f.write(verify_hash + "\n")
    
    print(f"\nResults saved to: phase0_results_{student_id}.txt")


if __name__ == "__main__":
    main()
