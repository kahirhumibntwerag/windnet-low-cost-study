"""Scientific invariant checks using synthetic data only; never load study holdouts."""
import unittest
import numpy as np
import torch
from matched.science import KaffeFeatures,metrics,samples,block_moments,moment_score,neural_fit

class ProtocolTests(unittest.TestCase):
    def data(self):
        return {'energy':np.stack([np.full((20,832),f,dtype=float) for f in range(1,6)]),'spatial':np.ones((5,20,4)),'output':np.stack([np.column_stack([np.arange(20)+1000*f if j==11 else np.full(20,10.) for j in range(15)]) for f in range(1,6)]),'fold':np.arange(1,6),'name':np.array(['f'+str(f) for f in range(1,6)])}
    def test_rows_and_causal_history(self):
        tr,va,_=samples(self.data(),[1,2,3],[4]);self.assertEqual(len(tr['y']),39);self.assertEqual(len(va['y']),14)
        self.assertEqual(tr['y'][0],1006);np.testing.assert_equal(tr['history_y'][0],[1000,1001,1002,1003]);self.assertEqual(va['y'][-1],4019)
        self.assertEqual(tr['offset'].max(),18);self.assertEqual(va['offset'].max(),19);self.assertFalse(set(tr['row'])&set(va['row']))
    def test_validation_cannot_fit_normalization(self):
        d=self.data();_,_,a=samples(d,[1,2,3],[4]);d['energy'][3:]=1e9;_,_,b=samples(d,[1,2,3],[4])
        np.testing.assert_array_equal(a['emin'],b['emin']);np.testing.assert_array_equal(a['escale'],b['escale'])
    def test_original_variance_metric(self):
        y=np.arange(14,dtype=float)*0+350;p=y+10;s=np.arange(14,dtype=float)*0+5;m=metrics(y,p,s)
        self.assertEqual(m['rmse'],10);self.assertEqual(m['chi2'],4);self.assertEqual(m['hse_ts'],None)
    def test_tensorflow_same_padding(self):
        x=torch.ones((1,1,224,224));p=KaffeFeatures.same(x,7,2)
        self.assertEqual(p.shape[-2:],(229,229));self.assertEqual(p[0,0,2,2],1);self.assertEqual(p[0,0,-1,-1],0)
    def test_block_moments_preserve_metrics(self):
        _,a,_=samples(self.data(),[1,2,3],[4]);p=a['y']*.9+300.;x=metrics(a['y'],p,a['sigma']);y=moment_score(block_moments(a,p))
        for k in ['rmse','r','chi2']:self.assertAlmostEqual(x[k],y[k],places=7)
    def test_fixed_epoch_prediction_does_not_read_test_labels(self):
        tr,va,_=samples(self.data(),[1,2,3],[4]);features={'energy':va['energy'],'spatial':va['spatial']}
        # There is deliberately no y/sigma/history_y key in the test feature dict.
        p,state,_=neural_fit(tr,features,19,fixed_epochs=1)
        self.assertEqual(len(p),14);self.assertTrue(np.isfinite(p).all());self.assertEqual(state['epochs'],1)
if __name__=='__main__':unittest.main()
