import unittest
import numpy as np
from sklearn.linear_model import Ridge
from next_loop.m19_kernel import prepare_kernel, fit_centered_kernel, predict_centered_kernel


def explicit(x, c):
    z = np.column_stack((x/np.sqrt(x.shape[1]), np.full(len(x), np.sqrt(c))))
    return np.einsum('ni,nj,nk->nijk', z, z, z).reshape(len(x), -1)


class CubicChecks(unittest.TestCase):
    def test_explicit_features_centering_and_batch_invariance(self):
        rng = np.random.default_rng(82)
        x = rng.normal(size=(25, 4)); y = rng.normal(size=25)*20+400
        weights = np.linspace(.2, 3, len(x)); z = rng.normal(size=(11, 4))
        for c in (.5, 1, 2):
            phi, psi = explicit(x, c), explicit(z, c)
            np.testing.assert_allclose(phi@phi.T, (x@x.T/x.shape[1]+c)**3, atol=1e-12)
            state, k = prepare_kernel(x, weights, 'cubic', c)
            np.testing.assert_allclose(k@state['a'], 0, atol=1e-12)
            for alpha in (.1, 1, 10):
                fit = fit_centered_kernel(state, k, y, weights, alpha)
                primal = Ridge(alpha=alpha, solver='cholesky').fit(phi, y, sample_weight=weights)
                actual = predict_centered_kernel(fit, state, z)
                np.testing.assert_allclose(actual, primal.predict(psi), atol=1e-9, rtol=0)
                np.testing.assert_allclose(actual, np.concatenate([predict_centered_kernel(fit, state, z[:4]), predict_centered_kernel(fit, state, z[4:])]), atol=1e-10)

    def test_registered_panel(self):
        from next_loop.m19 import panel
        rows = panel()
        self.assertEqual(len(rows), 108)
        self.assertTrue(all(r['kernel'] == 'cubic' for r in rows))
        self.assertEqual(len({(r['family'], r['kernel_factor'], r['ridge_alpha']) for r in rows}), 27)


if __name__ == '__main__': unittest.main()
