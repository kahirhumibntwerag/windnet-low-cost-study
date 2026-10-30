import unittest
import numpy as np
from next_loop.digital_topology import component_stats,topology,feature_families

class TopologyTests(unittest.TestCase):
    def test_ring_two_islands_and_diagonal_connectivity(self):
        ring=np.zeros((12,12),bool);ring[3:9,3:9]=True;ring[4:8,4:8]=False
        got=component_stats(ring,100)
        np.testing.assert_allclose(got[:2],[np.log(2),np.log(2)])
        self.assertAlmostEqual(got[2],.20)
        islands=np.zeros((12,12),bool);islands[3,3]=True;islands[7,7]=True
        np.testing.assert_allclose(component_stats(islands,100),[np.log(3),0,.01,.02])
        islands[4,4]=True
        self.assertAlmostEqual(component_stats(islands,100)[0],np.log(3))
        empty=component_stats(np.zeros((12,12),bool),100)
        np.testing.assert_array_equal(empty,0)

    def test_brightness_invariance_of_topology(self):
        images=np.full((2,256,256),100.,dtype=np.float32)
        images[:,70:90,70:90]=10;images[:,140:180,140:180]=45
        np.testing.assert_allclose(topology(images),topology(images*3),rtol=0,atol=0)

    def test_threshold_order_and_feature_dimensions(self):
        image=np.full((1,256,256),100.,dtype=np.float32);image[:,90:95,90:95]=25
        t=topology(image)
        self.assertEqual(t.shape,(1,8,4))
        np.testing.assert_array_equal(t[0,0],0)
        self.assertAlmostEqual(t[0,1,0],np.log(2))
        features=feature_families(image)
        self.assertEqual({k:v.shape for k,v in features.items()},{'betti':(1,18),'morphology':(1,34),'regional_morphology':(1,178)})
        np.testing.assert_allclose(features['betti'][0,:16],t[0,:,:2].ravel())
        np.testing.assert_allclose(features['morphology'][0,:32],t[0].ravel())
        np.testing.assert_allclose(features['betti'][0,-2:],features['regional_morphology'][0,144:146])

if __name__=='__main__':unittest.main()
