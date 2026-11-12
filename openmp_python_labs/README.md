# OpenMP Paradigms Python Laboratory Submission

This repository implements Labs 1 through 5 from the supplied practice manual in Python. Lab 6 is the optional bonus challenge and is intentionally omitted.

## Contents

- `lab1/` fork join teams, nondeterministic ordering, oversubscription, and CPU saturation
- `lab2/` numerical integration using a race, a critical section, and private accumulator reduction
- `lab3/` static and dynamic Mandelbrot work scheduling with configurable chunks
- `lab4/` false sharing, cache line padding, and thread local accumulation
- `lab5/` bounded task parallel merge sort with cutoff tuning and work span analysis
- `data/` raw CSV benchmark results and the ten Lab 1 console runs
- `plots/` figures generated only from the raw CSV files
- `report/` formal DOCX and PDF laboratory report

## Environment setup

Python 3.10 or newer is required. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the measured quick profile

The quick profile executes every configuration and all required cutoff values with reduced problem sizes so it can finish on a laptop. It still records actual measurements, not fabricated values.

```bash
PYTHONPATH=. .venv/bin/python run_all.py --profile quick
```

## Run the manual scale profile

The full profile selects the large values from the manual, including 100 million integration steps and counter increments, 1920 by 1080 Mandelbrot rendering, and five million merge sort integers. It can take a long time and may trigger the Lab 5 resource guard for tiny cutoffs because K equals 1 creates millions of tasks.

```bash
PYTHONPATH=. .venv/bin/python run_all.py --profile full
```

To deliberately run a five-million-element Lab 5 sweep with more than 50,000 live leaf tasks, invoke the lab directly and set an explicit cap after confirming that enough memory is available:

```bash
PYTHONPATH=. .venv/bin/python lab5/lab5_parallel_merge_sort.py --profile full --max-live-tasks 6000000
```

The raw data always records the actual input size, trial count, status, and timing. Re-run `make_plots.py` after changing any benchmark parameter.

## Reproducibility notes

- Timings use `time.perf_counter()`.
- Multiprocessing uses the portable `spawn` start method.
- Random inputs use a fixed seed.
- Lab 4 detects the machine cache line size and pads counters accordingly. The supplied Apple M2 reports 128 byte lines, so the measured stride is 16 signed 64 bit integers rather than the manual's generic 64 byte stride of 8.
- The Linux `perf` task is marked unavailable on macOS. On Linux, collect it with the event names stated in the manual.
- Pure Python threads are subject to the GIL. CPU bound experiments therefore use worker processes or NumPy kernels that release the GIL, while Lab 1 keeps native threads for team lifecycle identification.
