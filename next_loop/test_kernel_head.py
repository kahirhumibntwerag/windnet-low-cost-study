import unittest
import numpy as np
from sklearn.linear_model import Ridge
from next_loop.kernel_head import prepare_kernel, fit_centered_kernel, predict_centered_kernel


class KernelTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(471);x = rng.normal(size=(25, 6));x[:, -1] = 1
        y = 400+20*x[:, 0]+rng.normal(size=25)*4;w = np.linspace(.3, 3, len(x))
        return x, y, w, rng.normal(size=(9, 6))

    def test_linear_matches_weighted_primal_ridge_with_intercept(self):
        x, y, w, test = self.data();state, k = prepare_kernel(x, w, 'linear')
        for alpha in [.1, 1, 10]:
            fit = fit_centered_kernel(state, k, y, w, alpha)
            primal = Ridge(alpha=alpha*x.shape[1], solver='cholesky').fit(x, y, sample_weight=w)
            np.testing.assert_allclose(predict_centered_kernel(fit, state, test), primal.predict(test), rtol=0, atol=1e-9)

    def test_weighted_centering_is_symmetric_and_zero_mean(self):
        x, _, w, _ = self.data()
        for kind, factor in [('linear', None), ('rbf', 1)]:
            state, k = prepare_kernel(x, w, kind, factor)
            np.testing.assert_allclose(k, k.T, rtol=0, atol=1e-13)
            np.testing.assert_allclose(k@state['a'], 0, rtol=0, atol=1e-13)
            self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-10)

    def test_inference_is_batch_and_order_invariant(self):
        x, y, w, test = self.data();state, k = prepare_kernel(x, w, 'rbf', 4);fit = fit_centered_kernel(state, k, y, w, 1)
        full = predict_centered_kernel(fit, state, test)
        chunks = np.concatenate([predict_centered_kernel(fit, state, z) for z in [test[:3], test[3:]]])
        np.testing.assert_allclose(full, chunks, rtol=0, atol=1e-10)
        np.testing.assert_allclose(full, predict_centered_kernel(fit, state, test[::-1])[::-1], rtol=0, atol=1e-10)

    def test_gamma_uses_only_positive_fitting_offdiagonal_distances(self):
        state, k = prepare_kernel(np.array([[0.], [0.], [2.], [5.]]), np.ones(4), 'rbf', .25)
        self.assertEqual(state['distance_median'], 9)
        gamma = state['gamma'];fit = fit_centered_kernel(state, k, np.array([400., 400, 450, 500]), np.ones(4), 1)
        predict_centered_kernel(fit, state, np.array([[1000.], [-1000.]]))
        self.assertEqual(state['gamma'], gamma)

    def test_target_shift_changes_only_prediction_intercept(self):
        x, y, w, test = self.data();state, k = prepare_kernel(x, w, 'rbf', 1)
        a = fit_centered_kernel(state, k, y, w, 1);b = fit_centered_kernel(state, k, y+80, w, 1)
        np.testing.assert_allclose(predict_centered_kernel(b, state, test)-predict_centered_kernel(a, state, test), 80, rtol=0, atol=1e-10)

    def test_invalid_inputs_and_zero_distance_fail_without_fallback(self):
        with self.assertRaises(ValueError):prepare_kernel(np.ones((4, 3)), np.ones(4), 'rbf', 1)
        x, y, w, _ = self.data();bad = w.copy();bad[0] = 0
        with self.assertRaises(ValueError):prepare_kernel(x, bad, 'linear')
        state, k = prepare_kernel(x, w, 'linear')
        with self.assertRaises(ValueError):fit_centered_kernel(state, k, y[:-1], w, 1)
        with self.assertRaises(ValueError):predict_centered_kernel({}, state, np.array([[np.nan]*6]))

    def test_registered_panel_counts(self):
        from next_loop.m17 import panel
        rows = panel();self.assertEqual(len(rows), 144)
        self.assertEqual(sum(r['kernel']=='linear' for r in rows), 36)
        self.assertEqual(sum(r['kernel']=='rbf' for r in rows), 108)
        self.assertEqual(len({(r['family'], r['kernel'], r['gamma_factor'], r['ridge_alpha']) for r in rows}), 36)


if __name__ == '__main__':unittest.main()
