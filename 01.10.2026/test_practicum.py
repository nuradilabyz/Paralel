"""Independent numerical checks; run: python -m unittest -v test_practicum.py."""

import math
import unittest

import numba
import numpy as np

from openmp_practicum import (
    heat_step,
    initial_heat_grid,
    monte_carlo_pi,
    render_mandelbrot_cols,
    render_mandelbrot_rows,
    time_heat,
)


class PracticumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))

    def test_monte_carlo_pi_is_statistically_plausible(self):
        # Thread-local random generators make exact seeded equality unsuitable.
        self.assertLess(abs(monte_carlo_pi(100_000) - math.pi), 0.05)

    def test_both_mandelbrot_axes_match_independent_complex_reference(self):
        h, w, iterations = 24, 30, 60
        expected = np.zeros((h, w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                point = complex(-2.0 + c / w * 2.5, -1.2 + r / h * 2.4)
                z = 0j
                count = 0
                while abs(z) <= 2.0 and count < iterations:
                    z = z * z + point
                    count += 1
                expected[r, c] = count
        self.assertEqual(expected[12, 24], iterations)  # c=0 lies inside the set.
        self.assertEqual(expected[0, 0], 1)  # (-2, -1.2) escapes immediately.
        np.testing.assert_array_equal(render_mandelbrot_rows(h, w, iterations), expected)
        np.testing.assert_array_equal(render_mandelbrot_cols(h, w, iterations), expected)

    def test_one_heat_step_has_known_temperatures(self):
        for dtype in (np.float64, np.float32):
            with self.subTest(precision=np.dtype(dtype).name):
                u, target = initial_heat_grid(5, dtype)
                heat_step(u, target, dtype(0.2))
                expected = np.array([
                    [100, 100, 100, 100, 100],
                    [100, 40, 20, 20, 0],
                    [100, 20, 0, 0, 0],
                    [100, 20, 0, 0, 0],
                    [100, 0, 0, 0, 0],
                ], dtype=dtype)
                np.testing.assert_allclose(target, expected, rtol=0, atol=1e-5)

    def test_heat_boundaries_precision_and_thread_invariance(self):
        grids = {}
        for dtype in (np.float64, np.float32):
            numba.set_num_threads(1)
            _, single = time_heat(40, 60, dtype)
            numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
            _, parallel = time_heat(40, 60, dtype)
            np.testing.assert_array_equal(single, parallel)
            self.assertTrue(np.isfinite(parallel).all())
            self.assertGreaterEqual(parallel.min(), 0)
            self.assertLessEqual(parallel.max(), 100)
            np.testing.assert_array_equal(parallel[0, :], 100)
            np.testing.assert_array_equal(parallel[:, 0], 100)
            np.testing.assert_array_equal(parallel[-1, 1:], 0)
            np.testing.assert_array_equal(parallel[1:, -1], 0)
            grids[np.dtype(dtype).name] = parallel
        np.testing.assert_allclose(grids['float64'], grids['float32'], rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
