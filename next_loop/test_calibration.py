import unittest
import numpy as np
from next_loop.calibration import fit_map, predict_map

class CalibrationTests(unittest.TestCase):
    def test_known_affine_distribution_and_clipping(self):
        x = np.linspace(0, 100, 1001)
        m = fit_map(x, 2*x+10)
        np.testing.assert_allclose(predict_map(m, [25, 50, 75]), [60,110,160])
        np.testing.assert_allclose(predict_map(m, [-100,200]), [15,205])

    def test_ties_and_no_target_argument_at_inference(self):
        m = fit_map(np.zeros(20), np.arange(20))
        self.assertEqual(len(m['x']), 1)
        p = predict_map(m, [-5, 0, 7])
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all(p == p[0]))

if __name__ == '__main__':
    unittest.main()
