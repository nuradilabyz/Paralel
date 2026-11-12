"""Write the lab answers from actual measured results; no target is fabricated."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'


def load(name):
    return json.loads((RESULTS/name).read_text())


def main():
    host = load('host_profile.json')
    data = {letter: load(f'experiment_{letter}.json') for letter in ('a', 'b', 'c')}
    rows = {letter: {r['p']: r for r in d['summary']} for letter, d in data.items()}
    verification = load('verification.json')
    a, b, c = (rows[key] for key in ('a', 'b', 'c'))
    ratio = a[4]['karp_flatt']/a[2]['karp_flatt']
    a_flat = 0.80 <= ratio <= 1.25
    b_slow = b[4]['median_s'] > b[1]['median_s']
    b_grows = b[4]['karp_flatt'] > b[2]['karp_flatt']
    c_faster = c[4]['median_s'] < c[1]['median_s']
    c_eff = c[4]['efficiency_pct'] > 70
    target = lambda condition: 'MET' if condition else 'NOT MET'
    main_table = ['| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Diagnosis |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    diagnoses = {'a': 'Approximately flat effective serialization; intentional sleep plus process overhead',
                 'b': 'Per-increment mutex contention; observe actual speedup and e below',
                 'c': 'Private counters remove the per-increment shared lock; startup remains timed'}
    for letter, label in [('a','A: serial tail'), ('b','B: contention'), ('c','C: privatization')]:
        r = rows[letter]
        main_table.append(f"| {label} | {r[1]['median_s']:.6f} | {r[2]['median_s']:.6f} | {r[4]['median_s']:.6f} | {r[2]['karp_flatt']:.6f} | {r[4]['karp_flatt']:.6f} | {diagnoses[letter]} |")
    details = []
    for letter in ('a', 'b', 'c'):
        details += [f'### Experiment {letter.upper()}: speedup, efficiency and spread', '',
                    '| P | Median (s) | Min (s) | Max (s) | Speedup | Efficiency (%) | e(P) |',
                    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for p in (1,2,4):
            r=rows[letter][p]
            e='undefined' if p==1 else f"{r['karp_flatt']:.6f}"
            details.append(f"| {p} | {r['median_s']:.6f} | {r['min_s']:.6f} | {r['max_s']:.6f} | {r['speedup']:.4f} | {r['efficiency_pct']:.2f} | {e} |")
        details += ['', f"[Raw trials](results/experiment_{letter}.json) / [metrics TSV](results/experiment_{letter}.tsv) / [console](results/experiment_{letter}_console.txt)", '']
    groups = '; '.join(f"{g['physicalcpu']} {g['name']} cores" for g in host.get('core_groups', []))
    parts = '\n'.join(main_table)
    text = f"""# Empirical Scalability Diagnostics: Karp-Flatt

**Student:** Nuradil Abyz, **ID:** 230103188. **Date:** 29 October 2026. **Branch:** `practice-230103188`. Submission location: `29.10.2026/practice tasks` in `nuradilabyz/Paralel`.

## Part 0 - host and measurement method

Physical host: **{host['cpu_model']}**, {host['physical_cores']} physical cores, {host['logical_cores']} logical execution cores; {groups}. Platform: {host['platform']}. Python: {host['python'].splitlines()[0]}. [Recorded host profile](results/host_profile.json).

The three programs use Python's standard library and the `spawn` start method. Each P in {{1,2,4}} has five genuine trials; table values are medians. Worker-count order rotates between trials. Experiments run sequentially. Timers include pool/process creation, computation, IPC, final reduction where applicable, and process cleanup; A also includes the specified sleep. B allocates its counter before timing, as in the handout. No CPU pinning or thermal controls were imposed, so background activity, startup and mixed core types can affect timing. No example runtimes were substituted for measurements. Code and analysis were prepared with AI assistance; all timings were collected on this host.

## Metric calculation and requested synthesis table

For each experiment independently, `S(P)=T1/TP`, `E(P)=100*S(P)/P`, and

`e(P)=(1/S(P)-1/P)/(1-1/P)=(P*TP/T1-1)/(P-1)` for P>=2.

Work is fixed across P within each experiment. e(1) is undefined. The two Karp-Flatt expressions are algebraically equivalent; calculations below use unrounded medians.

{parts}

## Part 1 - Experiment A: the stubborn serial tail

[`exp_a_serial.py`](exp_a_serial.py) computes the handout's 16 chunks of `sum(sin(i)*cos(i), i=1..1,999,999)`, with `sleep(0.40)` before creating the pool. Total useful work is unchanged when P changes. Every chunk is checked against the independent trigonometric sum identity `0.5*sin(m)*sin(m+1)/sin(1)`, where m=1,999,999, with absolute tolerance 1e-8.

The observed speedups are S2={a[2]['speedup']:.4f} and S4={a[4]['speedup']:.4f}. e(4)/e(2)={ratio:.4f}; the rubric's [0.80,1.25] flatness criterion is **{target(a_flat)}**. The deliberate pause alone is {0.4/a[1]['median_s']*100:.2f}% of the measured P=1 runtime. Effective Karp-Flatt includes process startup and IPC as well as that pause, so it need not equal this fraction or fall exactly in the handout's illustrative 0.10-0.14 interval. These values are compatible with a roughly fixed serial contribution, not proof that sleep is the only overhead.

## Part 2 - Experiment B: the contention traffic jam

[`exp_b_contention.py`](exp_b_contention.py) performs exactly **2,000,000 total increments**, partitioned among P processes. The shared `Value('i',0)` and explicit `Lock` are preserved from the handout: Value has its own internal lock, and the explicit mutex protects the complete read-modify-write. Every trial verifies the final count exactly. No artificial sleep is used.

T4/T1={b[4]['median_s']/b[1]['median_s']:.4f}; S4={b[4]['speedup']:.4f}. The negative-speedup condition T4>T1 is **{target(b_slow)}**. e(2)={b[2]['karp_flatt']:.6f}, e(4)={b[4]['karp_flatt']:.6f}; e4/e2={b[4]['karp_flatt']/b[2]['karp_flatt']:.4f} (**{target(b_grows)}** for e4>e2). The controlled implementation repeatedly acquires locks around a single shared counter. A monotonic rise in e is not guaranteed: if lock-limited runtimes plateau while P grows, the metric's denominator also grows, and e can decline despite obvious negative speedup. We did not measure cache misses, bus traffic, kernel sleep states or lock spinning, so these are mechanisms consistent with the experiment rather than separately measured facts.

## Part 3 - Experiment C: the privatization fix

[`exp_c_fixed.py`](exp_c_fixed.py) performs exactly **20,000,000 total increments** in private Python integer variables. Each worker returns one partial result through Pool.map; the parent sums the results once. Every partial and the final 20,000,000 count are checked. Pool IPC and startup remain, but no shared lock is acquired per increment. A Python integer is not guaranteed to remain in a hardware register.

S4={c[4]['speedup']:.4f}; E4={c[4]['efficiency_pct']:.2f}%. T4<T1: **{target(c_faster)}**. E4>70%: **{target(c_eff)}**. The illustrative e<=0.03 target is **{target(c[2]['karp_flatt']<=0.03 and c[4]['karp_flatt']<=0.03)}** for both P=2 and P=4. Finite tasks and timed process startup can prevent ideal scaling even after the per-increment mutex is removed. B and C use different total workloads (2M versus 20M), so their raw seconds are not a fixed-work speedup comparison.

## Part 4 - verification, interpretation and provenance

Local numerical and metric verification: **{verification['status']}**. [`verify_results.py`](verify_results.py) checks all 45 recorded trials, independently recomputes medians, speedups, efficiencies and Karp-Flatt, verifies A's analytic sums and B/C's exact counts, and records SHA-256 digests of sources and measured data in [verification.json](results/verification.json). Numerical correctness and hardware-dependent performance targets are reported separately.

Karp-Flatt is an effective overhead measure, not a unique hardware diagnosis. A flat curve can be compatible with fixed serialization; a rising curve can reflect locks, scheduling, bandwidth, cache effects or other P-dependent costs. The controlled A/B/C implementations support the interpretations above without claiming that e alone proves a particular physical mechanism. The example table in the [handout](Practice%20Lab%20Handout.pdf) also contains arithmetic inconsistencies: its B times (0.65,1.10,2.40 s) imply e2=2.384615 and e4=4.589744 under its printed formula. Our results use the formula and actual times rather than those example metric values.

The source PDF is preserved unchanged. The handout asks for RESULTS.md at a repository root; this submission places it at the root of the requested `practice tasks` folder. Genuine staged commits record A, B and C/results, and the completed solution is available on both `main` and `practice-230103188` after push. No instructor sign-off is implied.

## Complete raw timing tables

{chr(10).join(details)}
"""
    (ROOT/'RESULTS.md').write_text(text)
    readme = """# Practice tasks - 29.10.2026

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
"""
    (ROOT/'README.md').write_text(readme)
    print('Wrote RESULTS.md and README.md from verified real measurements.')


if __name__ == '__main__':
    main()
