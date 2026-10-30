import unittest
import numpy as np
from sklearn.linear_model import Ridge
from next_loop.difference_ridge import adjacent_pairs, fit_difference_ridge, predict_difference_ridge


class DifferenceRidgeTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(421)
        x = np.column_stack([rng.normal(size=(18, 4)), np.ones(18)])
        y = 420 + x[:, 0] * 20 + rng.normal(size=18) * 5
        return x, y, np.linspace(.4, 3, 18), np.repeat([0, 1, 2], 6), np.tile(np.arange(6, 12), 3)

    def test_pairs_never_cross_blocks_or_missing_offsets(self):
        left, right = adjacent_pairs(np.array([0, 0, 0, 1, 1]), np.array([6, 7, 9, 8, 9]))
        np.testing.assert_array_equal(left, [0, 3])
        np.testing.assert_array_equal(right, [1, 4])
        with self.assertRaises(ValueError):adjacent_pairs(np.array([0, 0]), np.array([6, 6]))

    def test_zero_strength_matches_weighted_ridge_including_intercept(self):
        x, y, w, b, o = self.data()
        custom = fit_difference_ridge(x, y, w, b, o, 100, 0)
        reference = Ridge(alpha=100, solver='cholesky').fit(x, y, sample_weight=w)
        np.testing.assert_allclose(predict_difference_ridge(custom, x), reference.predict(x), rtol=0, atol=1e-10)
        test = x[::-1].copy();test[:, 1] += .2
        np.testing.assert_allclose(predict_difference_ridge(custom, test), reference.predict(test), rtol=0, atol=1e-10)

    def test_positive_strength_solves_declared_normal_equations(self):
        x, y, w, b, o = self.data();alpha = 10;strength = 4
        model = fit_difference_ridge(x, y, w, b, o, alpha, strength)
        beta = model['ridge'].coef_;c = model['y_mean'] - model['x_mean'] @ beta
        left, right = model['pair_left'], model['pair_right'];v = model['pair_weights']
        residual = c + x @ beta - y;dx = x[right] - x[left];dy = y[right] - y[left]
        self.assertAlmostEqual(float(np.sum(w * residual)), 0, places=8)
        gradient = x.T @ (w * residual) + strength * dx.T @ (v * (dx @ beta - dy)) + alpha * beta
        np.testing.assert_allclose(gradient, np.zeros(x.shape[1]), rtol=0, atol=1e-8)
        self.assertAlmostEqual(float(v.mean()), 1)

    def test_pairing_uses_identifiers_even_if_rows_are_shuffled(self):
        x, y, w, b, o = self.data();order = np.random.default_rng(9).permutation(len(y))
        a = fit_difference_ridge(x, y, w, b, o, 10, 1)
        z = fit_difference_ridge(x[order], y[order], w[order], b[order], o[order], 10, 1)
        np.testing.assert_allclose(predict_difference_ridge(a, x), predict_difference_ridge(z, x), rtol=0, atol=1e-10)

    def test_rejects_nonfinite_and_unaligned_training_inputs(self):
        x, y, w, b, o = self.data()
        with self.assertRaises(ValueError):fit_difference_ridge(x, y[:-1], w, b, o, 10, 1)
        bad = w.copy();bad[0] = 0
        with self.assertRaises(ValueError):fit_difference_ridge(x, y, bad, b, o, 10, 1)
        bad = x.copy();bad[0, 0] = np.nan
        with self.assertRaises(ValueError):fit_difference_ridge(bad, y, w, b, o, 10, 1)


if __name__ == '__main__':unittest.main()
