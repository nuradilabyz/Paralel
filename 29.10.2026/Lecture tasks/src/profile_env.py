"""Collect actual host topology without substituting textbook cache sizes."""
import argparse
from datetime import datetime, timezone
import json
import os
import platform
import re
import subprocess
import sys
import time

from common import RESULTS, write_json
from salt import get_student_params


def command(*args):
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else None


def sysctl(key):
    raw = command("sysctl", "-n", key)
    if not raw:
        return None
    return int(raw) if raw and raw.isdecimal() else raw


def profile():
    import numpy as np
    info = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_model": platform.processor() or platform.machine(),
            "microarchitecture": None, "base_frequency_hz": None,
            "physical_cores": None, "logical_threads": os.cpu_count(),
            "cache_line_bytes": None, "l1_data_cache_bytes": None,
            "l2_cache_bytes": None, "l3_cache_bytes": None,
            "os": platform.system(), "kernel_version": platform.release(),
            "platform": platform.platform(), "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "perf_counter_resolution_s": time.get_clock_info("perf_counter").resolution,
            "start_method": "spawn", "thread_affinity": "OS scheduled; no core pinning",
            "benchmark_parallelism": "experiments run sequentially; workers only within one experiment",
            "hardware_performance_counters": "not collected; no cache-miss-rate claim"}
    if platform.system() == "Darwin":
        raw = command("sysctl", "-a") or ""
        relevant = "\n".join(line for line in raw.splitlines()
                             if re.match(r"hw\.(physicalcpu|logicalcpu|cache|l1|l2|l3|perflevel|nperflevels|cpufrequency|memsize)|machdep.cpu.brand_string", line))
        (RESULTS / "sysctl_raw.txt").write_text(relevant + "\n")
        info.update(cpu_model=sysctl("machdep.cpu.brand_string"),
                    microarchitecture=platform.machine(),
                    base_frequency_hz=sysctl("hw.cpufrequency"),
                    physical_cores=sysctl("hw.physicalcpu"),
                    logical_threads=sysctl("hw.logicalcpu"),
                    cache_line_bytes=sysctl("hw.cachelinesize"),
                    l1_data_cache_bytes=sysctl("hw.l1dcachesize"),
                    l2_cache_bytes=sysctl("hw.l2cachesize"),
                    l3_cache_bytes=sysctl("hw.l3cachesize"),
                    memory_bytes=sysctl("hw.memsize"))
        levels = []
        for i in range(int(sysctl("hw.nperflevels") or 0)):
            levels.append({k: sysctl(f"hw.perflevel{i}.{k}") for k in
                           ("name", "physicalcpu", "logicalcpu", "l1dcachesize", "l2cachesize", "cpusperl2")})
        info["performance_levels"] = levels
        info["cache_notes"] = "Generic hw.l1/l2 describe a reported cache domain; heterogeneous per-level sizes are authoritative. Unreported L3 is null, not zero."
        if info["cpu_model"] == "Apple M2":
            info["theoretical_memory_bandwidth_gb_s"] = 100.0
            info["theoretical_memory_bandwidth_source"] = "https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/"
            info["theoretical_memory_bandwidth_note"] = "Manufacturer advertised unified-memory maximum; not a measured bandwidth and not CPU-exclusive. Official source checked 2026-10-29."
    else:
        from pathlib import Path
        raw = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        match = re.search(r"model name\s*:\s*(.+)", raw)
        if match:
            info["cpu_model"] = match.group(1)
        topology = command("lscpu", "-J")
        if topology:
            (RESULTS / "lscpu_raw.json").write_text(topology + "\n")
            fields = {r["field"].rstrip(":"): r["data"] for r in json.loads(topology)["lscpu"]}
            info["physical_cores"] = int(fields.get("Core(s) per socket", 1)) * int(fields.get("Socket(s)", 1))
        cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
        for index in cache_root.glob("index*"):
            level, kind = (index / "level").read_text().strip(), (index / "type").read_text().strip()
            size = (index / "size").read_text().strip()
            scale = 1024 if size.endswith("K") else 1024**2 if size.endswith("M") else 1
            value = int(size.rstrip("KM")) * scale
            if level == "1" and kind == "Data": info["l1_data_cache_bytes"] = value
            if level == "2": info["l2_cache_bytes"] = value
            if level == "3": info["l3_cache_bytes"] = value
            info["cache_line_bytes"] = int((index / "coherency_line_size").read_text())
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()
    info = profile()
    params = get_student_params(args.student_id)
    write_json("host_profile.json", info)
    write_json("salted_params.json", params)
    print(json.dumps({"host": info["cpu_model"], "physical_cores": info["physical_cores"],
                      "cache_line_bytes": info["cache_line_bytes"], **params}, indent=2))


if __name__ == "__main__":
    main()
