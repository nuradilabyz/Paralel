"""Three-point smoothing with replicated boundary values."""
import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        left = d_in[0] if idx == 0 else d_in[idx - 1]
        right = d_in[N - 1] if idx == N - 1 else d_in[idx + 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right

def run_stencil(h_in):
    arr = np.ascontiguousarray(h_in, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError('Expected a 1D array')
    if not arr.size:
        return arr.copy()
    d_in = cuda.to_device(arr)
    d_out = cuda.device_array_like(arr)
    stencil_1d[(arr.size + 255) // 256, 256](d_in, d_out, arr.size)
    return d_out.copy_to_host()

def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

if __name__ == '__main__':
    arr = np.sin(np.linspace(0, 10, 100007)).astype(np.float32)
    out, ref = run_stencil(arr), cpu_stencil(arr)
    assert np.allclose(out, ref, atol=1e-4)
    print(f'TASK 2 PASSED: MAX DELTA = {np.max(np.abs(out-ref))}')
