"""Synthetic-only checks for continuation gates and label-free prediction."""
import unittest
import numpy as np
from event_loop.science import gate,aggregate,train_aligned
class EventChecks(unittest.TestCase):
    def test_all_four_required(self):
        a={'rmse':80.,'r':.5,'chi2':30.,'hse_ts':.2};b={'rmse':79.,'r':.51,'chi2':29.,'hse_ts':.21}
        x={'rmse':77.,'r':.53,'chi2':28.,'hse_ts':.24};self.assertTrue(gate(x,[a,b])['all_four'])
        for k,v in [('rmse',80),('r',.50),('chi2',30),('hse_ts',.21)]:
            self.assertFalse(gate({**x,k:v},[a,b])['all_four'])
    def test_paper_aggregation(self):
        ms=[{'n':14,'rmse':v,'r':r,'chi2':20.,'hse_ts':.25,'mae':40.} for v,r in [(60,.4),(80,.8)]]
        x=aggregate(ms);self.assertEqual(x['rmse'],70);self.assertAlmostEqual(x['r'],np.tanh((np.arctanh(.4)+np.arctanh(.8))/2))
    def test_aligned_fit_never_reads_validation_labels(self):
        rng=np.random.default_rng(2);seq=rng.normal(420,30,(8,4));tr={'energy':rng.normal(0,.1,(8,4,832)),'y':seq[:,-1],'sigma':np.ones(8)*20,'future_y':seq,'future_sigma':np.ones((8,4))*20}
        features={'energy':np.zeros((2,4,832))};p,state=train_aligned(tr,features,19,1,'mse',.1)
        self.assertEqual(p.shape,(2,));self.assertTrue(np.isfinite(p).all());self.assertEqual(state['epochs'],1)
if __name__=='__main__':unittest.main()
