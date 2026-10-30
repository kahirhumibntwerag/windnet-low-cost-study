import unittest,numpy as np
from next_loop.target_weights import target_weights
class TargetWeightTests(unittest.TestCase):
    def test_declared_fast_threshold(self):
        y=np.array([499.,500.,600.]);np.testing.assert_array_equal(target_weights(y,'fast1.5'),[1,1.5,1.5]);np.testing.assert_array_equal(target_weights(y,'fast2'),[1,2,2])
    def test_density_emphasizes_rare_regions_and_is_bounded(self):
        y=np.r_[np.full(40,400.),800.];w=target_weights(y,'density');self.assertGreater(w[-1],w[0]);self.assertTrue(np.all((w>=.5)&(w<=3)))
        np.testing.assert_allclose(target_weights(np.full(20,400.),'density'),np.ones(20))
    def test_unit_weights(self):np.testing.assert_array_equal(target_weights([300,500,700],'unit'),[1,1,1])
if __name__=='__main__':unittest.main()
