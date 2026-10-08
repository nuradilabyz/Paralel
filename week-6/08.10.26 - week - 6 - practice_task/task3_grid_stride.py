"""Scale arbitrary-size vectors using exactly 64 x 256 threads."""
import numpy as np
from numba import cuda

@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor

def run_grid_stride(h_arr, factor):
    arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError('Expected a 1D array')
    if not arr.size:
        return arr.copy()
    d_arr = cuda.to_device(arr)
    threads_per_block = 256
    blocks_per_grid = 64
    grid_stride_scale_kernel[blocks_per_grid, threads_per_block](d_arr, np.float32(factor), arr.size)
    return d_arr.copy_to_host()

if __name__ == '__main__':
    N, factor = 2**24, 4.25
    result = run_grid_stride(np.ones(N, dtype=np.float32), factor)
    assert np.allclose(result, factor)
    print(f'TASK 3 PASSED: {N} elements equal {factor}; 64 x 256 = 16384 threads')
