# Practice tasks - 29.10.2026

Completed answers and actual benchmark results: **[RESULTS.md](RESULTS.md)**.

- [A: fixed serial tail](exp_a_serial.py)
- [B: per-increment lock contention](exp_b_contention.py)
- [C: privatized counters](exp_c_fixed.py)
- [Original handout](Practice%20Lab%20Handout.pdf)
- [Numerical and metrics audit](results/verification.json)

From this directory, with Python 3.8+ (standard library only):

```bash
python3 run_all.py
```

This reruns all experiments sequentially, verifies outputs, and rebuilds RESULTS.md. Individual experiments accept `--trials`; the committed report and verifier require the documented five-trial run. Results depend on hardware and OS scheduling. No installation or third-party Python packages are required.
