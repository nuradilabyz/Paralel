"""Experiment A: fixed compute work with the specified 0.40s serial pause."""
import math
import multiprocessing as mp
import time
from common import benchmark


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    ctx = mp.get_context('spawn')
    start = time.perf_counter_ns()
    sleep_start = time.perf_counter_ns()
    time.sleep(0.40)
    sleep_s = (time.perf_counter_ns() - sleep_start) / 1e9
    with ctx.Pool(processes=workers) as pool:
        partials = pool.map(heavy_math, [chunk_size] * total_chunks)
    elapsed_ns = time.perf_counter_ns() - start
    # Independent trigonometric finite-sum identity, not a second copy of the loop.
    m = chunk_size - 1
    expected_chunk = 0.5 * math.sin(m) * math.sin(m + 1) / math.sin(1)
    correct = len(partials) == total_chunks and all(math.isclose(x, expected_chunk, rel_tol=0, abs_tol=1e-8) for x in partials)
    if not correct:
        raise AssertionError('Trigonometric reduction differs from analytic finite sum')
    return {'elapsed_ns': elapsed_ns, 'elapsed_s': elapsed_ns / 1e9,
            'actual_sleep_s': sleep_s, 'partial_results': partials,
            'checksum': math.fsum(partials), 'analytic_chunk_reference': expected_chunk,
            'correct': correct}


if __name__ == '__main__':
    mp.freeze_support()
    benchmark('experiment_a', run_experiment,
              {'total_chunks': 16, 'chunk_size': 2_000_000, 'serial_sleep_s': 0.40},
              'Fixed 16 chunks of sin(i)*cos(i); 0.40s sleep before pool creation.')
