"""Shared measurement and metric reporting; Python standard library only."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
STUDENT_ID = '230103188'


def save_json(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def split_work(total, p):
    if total < 1 or p < 1:
        raise ValueError('Positive work and worker counts required')
    q, r = divmod(total, p)
    return [q + (i < r) for i in range(p)]


def metric(t1, tp, p):
    speedup = t1 / tp
    return {'speedup': speedup, 'efficiency_pct': 100 * speedup / p,
            'karp_flatt': ((p * tp / t1) - 1) / (p - 1) if p > 1 else None}


def benchmark(name, run, configuration, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    counts = [1, 2, 4]
    runs = []
    document = {'experiment': name, 'student_id': STUDENT_ID,
                'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                'configuration': configuration, 'description': description,
                'start_method': 'spawn', 'trials_per_p': args.trials,
                'worker_counts': counts, 'timer': 'time.perf_counter_ns',
                'timer_resolution_s': time.get_clock_info('perf_counter').resolution,
                'timing_scope': 'End-to-end: includes process/pool creation, work, IPC, reduction and joining. A additionally includes 0.40s sleep; counter allocation for B precedes timing.',
                'trial_order': 'Rotate [1,2,4] between trials; no overlapping experiments.',
                'runs': runs}
    for trial in range(args.trials):
        shift = trial % len(counts)
        for p in counts[shift:] + counts[:shift]:
            record = run(p)
            if not math.isfinite(record['elapsed_s']) or record['elapsed_s'] <= 0:
                raise AssertionError('Invalid measured duration')
            record.update(p=p, trial=trial + 1)
            runs.append(record)
            save_json(f'{name}.json', document)
            print(f'{name} trial={trial+1} P={p} T={record["elapsed_s"]:.6f}s correct={record["correct"]}', flush=True)
    medians = {p: statistics.median(r['elapsed_s'] for r in runs if r['p'] == p) for p in counts}
    rows = []
    for p in counts:
        samples = [r['elapsed_s'] for r in runs if r['p'] == p]
        rows.append({'p': p, 'median_s': medians[p], 'min_s': min(samples), 'max_s': max(samples),
                     **metric(medians[1], medians[p], p)})
    document['summary'] = rows
    document['all_numerical_checks_passed'] = all(r['correct'] for r in runs)
    save_json(f'{name}.json', document)
    with (RESULTS / f'{name}.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)
