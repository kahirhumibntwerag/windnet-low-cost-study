import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from next_loop.f3_panel import fixed_panel,fit_regional,calibrate,metric_gate
from next_loop.regional_features import temporal

class F3Tests(unittest.TestCase):
    def test_one_parameter_ablations_and_no_outer_calibration(self):
        panel={p['model']:p['recipe'] for p in fixed_panel()};self.assertEqual(len(panel),10)
        for name,key,value in [('regional_target_unit','target_weight','unit'),('regional_precision_zero','precision_lambda',0),('regional_global_1x1','grid',[1,1])]:
            expected=copy.deepcopy(panel['PRIMARY_F3']);expected['regional'][key]=value;self.assertEqual(panel[name],expected)
        no=panel['without_outer_calibration'];self.assertEqual((no['gain'],no['bias'],no['prior']['gain'],no['prior']['bias']),(1,0,1,0))
        self.assertEqual(no['prior']['prior'],'M6')
        np.testing.assert_allclose(calibrate(np.array([450,550]),500,1,0),[450,550])

    def test_regional_matches_explicit_M13_fitting_formula(self):
        rng=np.random.default_rng(43);train={'regions':rng.normal(size=(80,4,7)),'y':rng.uniform(300,650,80),'sigma':rng.uniform(3,40,80)};features={'regions':rng.normal(size=(17,4,7))}
        recipe=fixed_panel()[0]['recipe']['regional'];pred,bundle=fit_regional(train,features,recipe)
        x=temporal(train['regions'],'trend','linear');v=temporal(features['regions'],'trend','linear');s=StandardScaler().fit(x);s.scale_=np.maximum(s.scale_,.01)
        floor=np.quantile(train['sigma'],.05);med=np.median(train['sigma']);w=1+4*(med/np.maximum(train['sigma'],floor))**2;w/=w.mean();w*=np.where(train['y']>=500,2.,1.);w/=w.mean()
        expected=Ridge(alpha=100,solver='cholesky').fit(s.transform(x),train['y'],sample_weight=w).predict(s.transform(v))
        np.testing.assert_allclose(pred,expected,rtol=1e-12,atol=1e-12)
        with self.assertRaises(AssertionError):fit_regional(train,{**features,'y':np.zeros(17)},recipe)

    def test_undefined_metrics_and_tradeoff_fail_joint_gate(self):
        base={'mean_rmse':80.,'fisher_r':.55,'mean_chi2':25.,'mean_hse_ts':.25}
        good={'mean_rmse':79.,'fisher_r':.56,'mean_chi2':24.,'mean_hse_ts':.26}
        self.assertTrue(metric_gate(good,base)['all_four'])
        for value in [None,float('nan'),.24]:self.assertFalse(metric_gate({**good,'mean_hse_ts':value},base)['all_four'])

    def test_freeze_rejects_source_mutation_before_data_access(self):
        from next_loop import f3
        import json,hashlib
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);f={'source_hashes':{'x':'old'}};raw=json.dumps(f);(p/'f3_freeze.json').write_text(raw)
            digest=hashlib.sha256(raw.encode()).hexdigest()
            with patch.object(f3,'DATA',p),patch.object(f3,'source_hashes',return_value={'x':'new'}):
                with self.assertRaisesRegex(AssertionError,'Sources changed'):f3.check_freeze(digest)

if __name__=='__main__':unittest.main()
