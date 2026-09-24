"""Shared helpers for the OpenMP paradigm laboratory exercises."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sysctl(name: str) -> str:
    return command_output(["sysctl", "-n", name])


def hardware_info() -> dict[str, object]:
    logical = os.cpu_count() or 1
    physical: object = "unavailable"
    model = platform.processor() or "unavailable"
    cache_line: object = 64
    caches: dict[str, object] = {"l1_data_bytes": "unavailable", "l2_bytes": "unavailable", "l3_bytes": "unavailable"}

    if sys.platform == "darwin":
        model = sysctl("machdep.cpu.brand_string")
        physical = int(sysctl("hw.physicalcpu"))
        logical = int(sysctl("hw.logicalcpu"))
        cache_line = int(sysctl("hw.cachelinesize"))
        caches = {
            "l1_data_bytes": int(sysctl("hw.l1dcachesize")),
            "l2_bytes": int(sysctl("hw.l2cachesize")),
            "l3_bytes": sysctl("hw.l3cachesize"),
        }
    else:
        try:
            import psutil

            physical = psutil.cpu_count(logical=False) or "unavailable"
        except ImportError:
            pass
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break

    return {
        "cpu_model": model,
        "physical_cores": physical,
        "logical_threads": logical,
        "cache_line_bytes": cache_line,
        **caches,
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def save_hardware_info() -> dict[str, object]:
    ensure_output_dirs()
    info = hardware_info()
    (DATA_DIR / "system_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

