import unittest
import numpy as np
from sklearn.linear_model import Ridge
from next_loop.m18_kernel import prepare_kernel, fit_centered_kernel, predict_centered_kernel


def explicit(x, c):
    d = x.shape[1]
    # Ordered pairs give the exact degree-two polynomial feature map.
    return np.concatenate([np.einsum('ni,nj->nij', x, x).reshape(len(x), -1)/d,
                           np.sqrt(2*c/d)*x, np.full((len(x), 1), c)], axis=1)


class PolynomialChecks(unittest.TestCase):
    def test_kernel_primal_weighted_ridge_and_centering(self):
        rng = np.random.default_rng(81)
        x = rng.normal(size=(25, 5)); y = rng.normal(size=25)*20+400
        weights = np.linspace(.2, 3, len(x)); z = rng.normal(size=(11, 5))
        for c in (.5, 1, 2):
            phi, psi = explicit(x, c), explicit(z, c)
            np.testing.assert_allclose(phi@phi.T, (x@x.T/x.shape[1]+c)**2, atol=1e-12)
            state, k = prepare_kernel(x, weights, 'poly', c)
            np.testing.assert_allclose(k@state['a'], 0, atol=1e-12)
            for alpha in (.1, 1, 10):
                fit = fit_centered_kernel(state, k, y, weights, alpha)
                primal = Ridge(alpha=alpha, solver='cholesky').fit(phi, y, sample_weight=weights)
                actual = predict_centered_kernel(fit, state, z)
                np.testing.assert_allclose(actual, primal.predict(psi), atol=1e-9, rtol=0)
                np.testing.assert_allclose(actual, np.concatenate([predict_centered_kernel(fit, state, z[:4]), predict_centered_kernel(fit, state, z[4:])]), atol=1e-10)

    def test_registered_panel(self):
        from next_loop.m18 import panel
        rows = panel()
        self.assertEqual(len(rows), 144)
        self.assertEqual(sum(r['kernel'] == 'linear' for r in rows), 36)
        self.assertEqual(sum(r['kernel'] == 'poly' for r in rows), 108)
        self.assertEqual(len({(r['family'], r['kernel'], r['kernel_factor'], r['ridge_alpha']) for r in rows}), 36)


if __name__ == '__main__': unittest.main()
