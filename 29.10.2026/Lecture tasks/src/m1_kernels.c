/* Explicit memory access kernels called by Python through ctypes.
 * Build without fast-math or FP contraction to retain k-order summation.
 */
#include <stddef.h>

double stride_sum(const double *data, size_t elements, size_t stride) {
    double total = 0.0;
    for (size_t i = 0; i < elements; i += stride) {
        total += data[i];
    }
    return total;
}

/* A and B are full N-by-N arrays; C is the M-by-M leading-block product. */
void dense_ijk(const double *restrict a, const double *restrict b,
               double *restrict c, size_t n, size_t m) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double total = 0.0;
            for (size_t k = 0; k < m; ++k) {
                total += a[i*n + k] * b[k*n + j];
            }
            c[i*m + j] = total;
        }
    }
}

void dense_ikj(const double *restrict a, const double *restrict b,
               double *restrict c, size_t n, size_t m) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < m; ++k) {
            const double aik = a[i*n + k];
            for (size_t j = 0; j < m; ++j) {
                c[i*m + j] += aik * b[k*n + j];
            }
        }
    }
}
