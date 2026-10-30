import unittest,numpy as np
from next_loop.regional_features import regions,temporal

class RegionTests(unittest.TestCase):
    def test_uniform_images_and_brightness_invariance_of_local_ratios(self):
        x=np.full((2,256,256),50,dtype=np.float32);x[1]*=2
        z=regions(x,(4,3));self.assertEqual(z.shape,(2,50))
        np.testing.assert_allclose(z[0,:-2],z[1,:-2]);self.assertNotEqual(z[0,-2],z[1,-2])
        self.assertTrue(np.isfinite(z).all())
    def test_limb_exclusion_and_empty_cells(self):
        x=np.full((1,256,256),50,dtype=np.float32);base=regions(x,(6,6));x[:,0,0]=10000
        np.testing.assert_allclose(regions(x,(6,6)),base)
        self.assertTrue(np.isfinite(regions(np.zeros_like(x),(6,6))).all())
    def test_temporal_order_and_nonlinear_encoding(self):
        x=np.arange(8,dtype=float).reshape(1,4,2);linear=temporal(x,'trend','linear');squared=temporal(x,'trend','squares')
        np.testing.assert_allclose(linear,[[6,7,3,4,6,6]])
        np.testing.assert_allclose(squared[:,linear.shape[1]:],linear**2)
if __name__=='__main__':unittest.main()
