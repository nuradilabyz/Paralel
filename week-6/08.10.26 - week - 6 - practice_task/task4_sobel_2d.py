"""Sobel-X correlation with zero output on all four image borders."""
import numpy as np
from numba import cuda

@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            d_out[row, col] = (
                -1.0 * d_in[row-1, col-1] + 1.0 * d_in[row-1, col+1]
                -2.0 * d_in[row, col-1] + 2.0 * d_in[row, col+1]
                -1.0 * d_in[row+1, col-1] + 1.0 * d_in[row+1, col+1])
        else:
            d_out[row, col] = 0.0

def run_sobel(h_img):
    img = np.ascontiguousarray(h_img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError('Expected a 2D image')
    if not img.size:
        return img.copy()
    rows, cols = img.shape
    d_in = cuda.to_device(img)
    d_out = cuda.device_array_like(img)
    threads_2d = (16, 16)
    grid = ((cols + 15) // 16, (rows + 15) // 16)
    sobel_x_kernel[grid, threads_2d](d_in, d_out, rows, cols)
    return d_out.copy_to_host()

def cpu_sobel(img):
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (-img[:-2, :-2] + img[:-2, 2:]
        - 2*img[1:-1, :-2] + 2*img[1:-1, 2:]
        - img[2:, :-2] + img[2:, 2:])
    return out

if __name__ == '__main__':
    img = np.random.default_rng(230103188).random((2048, 2048), dtype=np.float32)
    out, ref = run_sobel(img), cpu_sobel(img)
    assert np.allclose(out, ref, atol=1e-4)
    assert np.all(out[[0, -1], :] == 0) and np.all(out[:, [0, -1]] == 0)
    print(f'TASK 4 PASSED: shape={img.shape}; MAX DELTA = {np.max(np.abs(out-ref))}')
