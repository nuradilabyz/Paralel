"""Experiment B: the handout's shared Value and per-increment mutex."""
import multiprocessing as mp
import time
from common import benchmark, split_work


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    if total_increments > 2**31 - 1:
        raise ValueError('Value(i) capacity exceeded')
    ctx = mp.get_context('spawn')
    # Preserve both synchronization layers in the handout: Value's own RLock
    # and the explicit Lock around the read-modify-write operation.
    counter = ctx.Value('i', 0)
    lock = ctx.Lock()
    work = split_work(total_increments, p)
    processes = [ctx.Process(target=contended_worker, args=(counter, lock, n)) for n in work]
    start = time.perf_counter_ns()
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
            if process.exitcode != 0:
                raise RuntimeError(f'Worker failed or timed out: {process.exitcode}')
        elapsed_ns = time.perf_counter_ns() - start
        total = counter.value
        if total != total_increments:
            raise AssertionError(f'Lost increments: {total} != {total_increments}')
        return {'elapsed_ns': elapsed_ns, 'elapsed_s': elapsed_ns / 1e9,
                'work_per_worker': work, 'actual_total': total,
                'expected_total': total_increments, 'correct': True}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == '__main__':
    mp.freeze_support()
    benchmark('experiment_b', run_contended,
              {'total_increments': 2_000_000, 'counter_type': 'signed int32',
               'synchronization': 'Value default RLock plus explicit per-increment Lock, as in handout'},
              'Fixed 2,000,000 shared increments with a mutex on every update.')
