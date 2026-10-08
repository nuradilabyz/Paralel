"""Uniform, interleaved and warp-aligned arithmetic; 10 warmed-up trials."""
import json
import time
import numpy as np
from numba import cuda, float32

@cuda.jit
def uniform_kernel(d_in, d_out):
    idx = cuda.grid(1)
    if idx < d_in.size:
        value = d_in[idx]
        for _ in range(1000):
            value = value * float32(1.0001) + float32(0.0001)
        d_out[idx] = value

@cuda.jit
def interleaved_kernel(d_in, d_out):
    idx = cuda.grid(1)
    if idx < d_in.size:
        value = d_in[idx]
        if idx % 2 == 0:
            for _ in range(1000):
                value = value * float32(1.0001) + float32(0.0001)
        else:
            for _ in range(1000):
                value = (value - float32(0.0001)) / float32(1.0001)
        d_out[idx] = value

@cuda.jit
def warp_aligned_kernel(d_in, d_out):
    idx = cuda.grid(1)
    if idx < d_in.size:
        value = d_in[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(1000):
                value = value * float32(1.0001) + float32(0.0001)
        else:
            for _ in range(1000):
                value = (value - float32(0.0001)) / float32(1.0001)
        d_out[idx] = value

def benchmark():
    N = 2**20
    # Separate output keeps all warm-ups and trials on identical input.
    d_in = cuda.to_device(np.ones(N, dtype=np.float32))
    d_out = cuda.device_array(N, dtype=np.float32)
    result = {}
    plus = minus = np.float32(1)
    for _ in range(1000):
        plus = plus*np.float32(1.0001) + np.float32(0.0001)
        minus = (minus-np.float32(0.0001))/np.float32(1.0001)
    idx = np.arange(N)
    for name, kernel in [('A_uniform', uniform_kernel), ('B_interleaved', interleaved_kernel), ('C_warp_aligned', warp_aligned_kernel)]:
        kernel[(N+255)//256, 256](d_in, d_out)
        cuda.synchronize()
        samples = []
        for _ in range(10):
            cuda.synchronize()
            start = time.perf_counter()
            kernel[(N+255)//256, 256](d_in, d_out)
            cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        out = d_out.copy_to_host()
        expected = np.full(N, plus, dtype=np.float32) if name.startswith('A') else np.where((idx if name.startswith('B') else idx//32) % 2 == 0, plus, minus)
        # Fused multiply-add may differ slightly from NumPy's separate operations.
        assert np.allclose(out, expected, rtol=2e-4, atol=1e-4), name
        result[name] = {'mean_ms': float(np.mean(samples)), 'trials_ms': samples, 'max_delta': float(np.max(np.abs(out-expected)))}
        print(f'{name}: {np.mean(samples):.6f} ms (mean of 10 trials)')
    return result

if __name__ == '__main__':
    from pathlib import Path
    Path('benchmark_results.json').write_text(json.dumps(benchmark(), indent=2)+'\n')
