#!/usr/bin/env python3
"""Reproduce the experiments sequentially on the current physical host."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, logfile, *arguments):
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    print(f"\nRunning {script}", flush=True)
    process = subprocess.Popen([sys.executable, "-u", str(ROOT / script), *arguments],
                               cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    with (ROOT / "results" / logfile).open("w") as stream:
        for line in process.stdout:
            sys.stdout.write(line)
            stream.write(line)
    if process.wait() != 0:
        raise SystemExit(f"{script} failed; inspect results/{logfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", default="230103188")
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    run("src/profile_env.py", "profile_console.txt", "--student-id", args.student_id)
    run("src/stride_bench.py", "stride_console.txt")
    run("src/dense_kernel.py", "dense_console.txt")
    run("src/false_sharing.py", "false_sharing_console.txt")
    run("src/scaling_study.py", "scaling_console.txt")
    run("src/verify_numerics.py", "numerical_console.txt")
    run("src/build_summary.py", "summary_console.txt")
    print("\nExperiments and local audit complete. Check the provided verifier status in README.md.")


if __name__ == "__main__":
    main()
