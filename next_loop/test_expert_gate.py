import unittest
import numpy as np
from next_loop.expert_gate import panel,gate_weights,predict

class ExpertGateTests(unittest.TestCase):
    def test_registered_panel_and_extreme_sigmoid_stability(self):
        rows=panel();self.assertEqual(len(rows),104)
        self.assertEqual(sum(r['recipe']['kind']=='static' for r in rows),8)
        for row in rows:
            g=gate_weights(np.array([-1e300,400.,500.,1e300]),row['recipe'])
            self.assertTrue(np.isfinite(g).all() and (g>=0).all() and (g<=1).all())
        g=gate_weights(np.array([-1e300,500.,1e300]),{'kind':'conditional','threshold':500,'temperature':25,'max_share':.75})
        np.testing.assert_allclose(g,[0,.375,.75])

    def test_convex_routing_and_identical_expert_invariance(self):
        experts={'A':np.array([300.,400.]),'B':np.array([600.,500.]),'incumbent':np.array([450.,450.])}
        np.testing.assert_allclose(predict(experts,{'kind':'static','event_share':.25,'incumbent_mix':.5}),[412.5,437.5])
        same={k:np.array([310.,440.,530.]) for k in experts}
        for row in panel():np.testing.assert_allclose(predict(same,row['recipe']),same['A'])

    def test_label_fields_and_misaligned_shapes_rejected(self):
        experts={k:np.array([400.,500.]) for k in ['A','B','incumbent']};recipe=panel()[0]['recipe']
        with self.assertRaises(AssertionError):predict({**experts,'y':np.zeros(2)},recipe)
        with self.assertRaises(AssertionError):predict({**experts,'B':np.zeros(3)},recipe)

if __name__=='__main__':unittest.main()
