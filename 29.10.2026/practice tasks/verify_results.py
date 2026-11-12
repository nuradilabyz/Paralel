"""Independently check saved reductions, raw timing summaries and Karp-Flatt."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def near(a, b, message):
    check(math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-14), message)


def verify():
    reports = {}
    for letter in ('a', 'b', 'c'):
        name = f'experiment_{letter}'
        data = json.loads((RESULTS / f'{name}.json').read_text())
        config, runs = data['configuration'], data['runs']
        check(data['worker_counts'] == [1, 2, 4] and data['trials_per_p'] == 5, 'Expected five trials at P=1,2,4')
        check(len(runs) == 15 and {(r['p'], r['trial']) for r in runs} ==
              {(p, trial) for p in (1, 2, 4) for trial in range(1, 6)}, 'Incomplete or duplicated trial coverage')
        for run in runs:
            check(run['elapsed_ns'] > 0 and math.isfinite(run['elapsed_s']), 'Invalid duration')
            near(run['elapsed_s'], run['elapsed_ns']/1e9, 'Duration conversion mismatch')
            check(run['correct'] is True, 'Recorded numerical failure')
            if letter == 'a':
                check(config == {'total_chunks': 16, 'chunk_size': 2_000_000, 'serial_sleep_s': 0.40}, 'A defaults changed')
                m = config['chunk_size'] - 1
                reference = 0.5 * math.sin(m) * math.sin(m+1) / math.sin(1)
                check(len(run['partial_results']) == 16, 'A missing chunk')
                check(all(math.isclose(x, reference, rel_tol=0, abs_tol=1e-8) for x in run['partial_results']), 'A analytic finite-sum mismatch')
                near(run['checksum'], math.fsum(run['partial_results']), 'A reduction checksum')
                near(run['analytic_chunk_reference'], reference, 'A reference mismatch')
                check(run['actual_sleep_s'] >= 0.40 and run['elapsed_s'] > run['actual_sleep_s'], 'A sleep missing')
            else:
                expected = 2_000_000 if letter == 'b' else 20_000_000
                check(config['total_increments'] == expected, 'Counter problem size changed')
                check(run['actual_total'] == run['expected_total'] == expected, 'Incorrect final counter')
                check(len(run['work_per_worker']) == run['p'] and sum(run['work_per_worker']) == expected, 'Work partition mismatch')
                if letter == 'c':
                    check(run['partial_results'] == run['work_per_worker'], 'Private partial results mismatch')
        times = {p: statistics.median(r['elapsed_s'] for r in runs if r['p'] == p) for p in (1, 2, 4)}
        with (RESULTS / f'{name}.tsv').open(newline='') as stream:
            tsv = list(csv.DictReader(stream, delimiter='\t'))
        check(len(tsv) == len(data['summary']) == 3, 'Summary row coverage')
        check({int(r['p']) for r in tsv} == {1, 2, 4}, 'TSV worker coverage')
        check({r['p'] for r in data['summary']} == {1, 2, 4}, 'JSON summary worker coverage')
        for summary in (data['summary'], tsv):
            for row in summary:
                p = int(row['p'])
                samples = [r['elapsed_s'] for r in runs if r['p'] == p]
                s = times[1]/times[p]
                near(row['median_s'], times[p], 'Median mismatch')
                near(row['min_s'], min(samples), 'Minimum mismatch')
                near(row['max_s'], max(samples), 'Maximum mismatch')
                near(row['speedup'], s, 'Speedup mismatch')
                near(row['efficiency_pct'], 100*s/p, 'Efficiency mismatch')
                if p == 1:
                    check(row['karp_flatt'] in (None, ''), 'e(1) must be undefined')
                else:
                    near(row['karp_flatt'], (p*times[p]/times[1]-1)/(p-1), 'Karp-Flatt formula mismatch')
        reports[name] = {'numerical_status': 'PASS', 'raw_trials_verified': len(runs),
                         'median_and_metric_check': 'PASS'}
        print(f'PASS: {name}: all 15 trials and numerical results verified')
    manifest = []
    paths = list(ROOT.glob('*.py')) + [ROOT/'Practice Lab Handout.pdf']
    paths += [RESULTS/'host_profile.json']
    paths += [RESULTS/f'experiment_{letter}.{extension}' for letter in ('a', 'b', 'c') for extension in ('json', 'tsv')]
    for path in sorted(paths):
        manifest.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    output = {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'checks': reports, 'files_sha256': manifest,
              'scope': 'Checks saved numerical outputs and timing arithmetic; does not certify authorship or require hardware-dependent target speedups.'}
    (RESULTS/'verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print('PASS: complete local numerical and metrics audit')


if __name__ == '__main__':
    (RESULTS/'verification.json').unlink(missing_ok=True)
    verify()
