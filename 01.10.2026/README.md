# OpenMP Python Lab Practicum - 01.10.2026

Abyz Nuradil - student ID **230103188**.

Completed implementation of all three challenges in `OpenMP_Python_Lab_Practicum.pdf`:

1. Monte Carlo pi: 120,000,000 samples at 1, 2, 4, 8 and maximum available threads, deduplicated and filtered to valid counts.
2. Mandelbrot: row-parallel and column-parallel 2500 x 2500 renders with 1,000 iterations; identical pixels and a 300-DPI image.
3. Heat diffusion: 1500 x 1500 grids for 300 steps, both float64 and float32, measured at every thread count.

## Run on macOS or Linux

```bash
cd 01.10.2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_practicum.py
python openmp_practicum.py --environment "Student laptop"
```

On Windows, create the environment with `py -m venv .venv`, activate it with `.venv\Scripts\Activate.ps1` in PowerShell, then run the same `python` commands.

Plug the laptop into wall power and close CPU-heavy applications as instructed in the practicum. No local C/C++ compiler is required. The script records the actual Numba threading backend; its OpenMP/TBB availability depends on the installed platform build.

## Files

| File | Purpose |
| --- | --- |
| `openmp_practicum.py` | All kernels, benchmarking, correctness checks, and automatic result export |
| `test_practicum.py` | Independent numerical checks for all three challenges |
| `benchmark_report.md` | Completed scorecard and answers to every technical question |
| `completed_practicum.pdf` | Original task sheet with student details and scorecard filled, followed by answer pages |
| `benchmark_results.json` | Hardware, software versions, every timing repetition, estimates, and validation results |
| `benchmark_results.csv` | Measured tables for spreadsheet import |
| `benchmark_console.txt` | Console output from the full benchmark run |
| `mandelbrot_output.png` | Exported fractal image |
| `requirements.txt` | Runtime dependencies |

The committed results identify an automated Linux workspace and its actual CPU. They are not measurements from the student's MacBook. Re-run the script on the laptop to obtain laptop results; the Markdown report, CSV, JSON, and image update automatically. The committed PDF is a snapshot of the committed workspace results.

All full benchmarks use the assigned sizes and report the median of three repetitions after JIT warmup. Both heat buffers are reset before each timing. `--quick --output-dir /tmp/openmp-smoke` performs a small smoke run in a separate directory without replacing the full results. `--repeats 1` requests a single measurement.

Dynamic scheduling is discussed in the report. The assigned row/column experiment uses Numba's default static scheduling; it does not claim to turn on OpenMP `schedule(dynamic)` merely by swapping axes.
