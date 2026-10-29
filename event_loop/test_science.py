"""Synthetic-only checks for continuation gates and label-free prediction."""
import unittest
import numpy as np
from event_loop.science import gate,aggregate,train_aligned
class EventChecks(unittest.TestCase):
    def test_final_four_metric_bootstrap_identical_models(self):
        from event_loop.final import all_four_bootstrap
        y=np.tile([300,300,350,450,550,550,500,450,400,350,300,300,300,300],4).astype(float);va={'block':np.repeat(np.arange(4),14),'y':y,'sigma':np.ones(len(y))*20};folds=[{'test':va,'predictions':{'same':y,'released_windnet':y}} for _ in range(5)]
        result=all_four_bootstrap(folds,'same',draws=50)
        for m in result['metrics'].values():self.assertEqual(m['delta_95'],[0.,0.]);self.assertEqual(m['defined_draws'],50)
    def test_bootstrap_cache_preserves_exact_draws_and_intervals(self):
        from event_loop.mixtures import bootstrap
        from unittest.mock import patch
        rng=np.random.default_rng(4);datasets={};pp={};cc={}
        for f in range(1,5):
            y=np.tile([300,300,350,450,550,550,500,450,400,350,300,300,300,300],4).astype(float);va={'block':np.repeat(np.arange(4),14),'y':y,'sigma':np.ones(len(y))*20};datasets[f]=({},va);pp[f]=y+rng.normal(0,5,len(y));cc[f]=y+rng.normal(0,30,len(y))
        with patch('event_loop.mixtures.split',side_effect=lambda f:({},datasets[f][1],{})):
            expected=bootstrap(pp,cc)
        with patch('event_loop.mixtures.split',side_effect=AssertionError('Cached bootstrap must not reload splits')):
            actual=bootstrap(pp,cc,datasets=datasets)
        self.assertEqual(actual,expected)
    def test_final_location_prediction_has_no_validation_labels(self):
        from event_loop.final import fit_location
        from event_loop.location import ridge_path
        from pathlib import Path
        import tempfile
        rng=np.random.default_rng(3);x=rng.normal(size=(20,4,3));v=rng.normal(size=(5,4,3));y=rng.normal(400,20,20);sigma=rng.uniform(10,30,20)
        recipe={'family':'location_ridge','representation':'ratio','view':'flatten','weight_lambda':1,'alpha':10000}
        with tempfile.TemporaryDirectory() as td:
            actual=fit_location({'ratio':x,'y':y,'sigma':sigma},{'ratio':v},recipe,{},Path(td))
        expected=ridge_path(x.reshape(20,-1),v.reshape(5,-1),y,sigma,1,[10000])[10000]
        np.testing.assert_allclose(actual,expected,atol=1e-9)
    def test_stability_rejects_a_single_fold_driven_win(self):
        from event_loop.precision import stability
        base={'n':14,'rmse':80.,'r':.5,'chi2':30.,'hse_ts':.2,'mae':40.};good={**base,'rmse':75.,'r':.6,'chi2':28.,'hse_ts':.3};controls=[{'fold_metrics':[base.copy() for _ in range(4)]}]*2
        self.assertTrue(stability({'fold_metrics':[good.copy() for _ in range(4)]},controls)['pre_bootstrap_passed'])
        mixed=[{**base,'rmse':81.,'r':.49,'chi2':31.,'hse_ts':.19} for _ in range(4)];mixed[0]=good
        self.assertFalse(stability({'fold_metrics':mixed},controls)['pre_bootstrap_passed'])
    def test_location_ridge_matches_weighted_primal(self):
        from event_loop.location import ridge_path,representation
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(19);x=rng.normal(size=(20,8));v=rng.normal(size=(5,8));y=rng.normal(size=20);sigma=rng.uniform(.1,2,20)
        for weight in [0,1]:
            actual=ridge_path(x,v,y,sigma,weight,[100])[100];s=StandardScaler().fit(x);s.scale_=np.maximum(s.scale_,.01);w=1+weight*(np.median(sigma)/np.maximum(sigma,np.quantile(sigma,.05)))**2;w/=w.mean()
            expected=Ridge(alpha=100).fit(s.transform(x),y,sample_weight=w).predict(s.transform(v));np.testing.assert_allclose(actual,expected,atol=1e-9)
        local=np.ones((2,832,16),dtype=np.float32);glob=np.ones((2,832),dtype=np.float32)
        for kind in ['log','ratio']:self.assertEqual(representation(local,glob,kind).shape,(2,14144))
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
