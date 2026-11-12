"""Experiment C: private accumulation, with one final parent reduction."""
import multiprocessing as mp
import time
from common import benchmark, split_work


def private_worker(iterations):
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=20_000_000):
    ctx = mp.get_context('spawn')
    tasks = split_work(total_increments, p)
    start = time.perf_counter_ns()
    with ctx.Pool(processes=p) as pool:
        partials = pool.map(private_worker, tasks)
    total = sum(partials)
    elapsed_ns = time.perf_counter_ns() - start
    if total != total_increments or partials != tasks:
        raise AssertionError('Private accumulation/reduction returned incorrect counts')
    return {'elapsed_ns': elapsed_ns, 'elapsed_s': elapsed_ns / 1e9,
            'work_per_worker': tasks, 'partial_results': partials,
            'actual_total': total, 'expected_total': total_increments, 'correct': True}


if __name__ == '__main__':
    mp.freeze_support()
    benchmark('experiment_c', run_privatized,
              {'total_increments': 20_000_000, 'synchronization': 'Pool.map IPC and one final parent sum; no shared counter'},
              'Fixed 20,000,000 private increments with a single final reduction.')
