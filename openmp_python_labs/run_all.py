"""Run all required Python labs and generate the CSV data and plots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT, save_hardware_info


LABS = (
    ROOT / "lab1" / "lab1_fork_join.py",
    ROOT / "lab2" / "lab2_pi_reduction.py",
    ROOT / "lab3" / "lab3_mandelbrot_scheduling.py",
    ROOT / "lab4" / "lab4_false_sharing.py",
    ROOT / "lab5" / "lab5_parallel_merge_sort.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    save_hardware_info()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    for lab in LABS:
        print(f"Running {lab.parent.name}...", flush=True)
        subprocess.run([sys.executable, str(lab), "--profile", args.profile], cwd=ROOT, env=environment, check=True)
    subprocess.run([sys.executable, str(ROOT / "make_plots.py")], cwd=ROOT, env=environment, check=True)
    print("All required labs completed. Lab 6 bonus was intentionally omitted.")


if __name__ == "__main__":
    main()
