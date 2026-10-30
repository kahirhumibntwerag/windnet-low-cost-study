"""Synthetic evaluator checks; no study labels or forecasts are read."""
import unittest
import numpy as np
from external_eval.scoring import evaluate, replica_moments, summarize, bootstrap_units

class ScoreTests(unittest.TestCase):
    def test_moments_equal_direct_scores(self):
        y=np.array([320.,330,380,470,520,480,400,360,330,350,390,460,520,450])
        p=np.tile(y+np.arange(14)*.2,(5,1));s=np.full(14,20.);s[0]=0
        m=replica_moments(y,p,s);v=summarize(m[None])
        self.assertAlmostEqual(v[0],np.sqrt(np.mean((p[0]-y)**2)),places=10)
        self.assertAlmostEqual(v[1],np.corrcoef(p[0],y)[0,1],places=10)
        self.assertAlmostEqual(v[2],np.mean(((p[0,1:]-y[1:])/s[1:])**2),places=10)

    def test_common_mask_drops_whole_gap_block_and_no_replica_inflation(self):
        n=70;dates=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-01')+n)
        y=400+100*np.sin(np.arange(n));s=np.ones(n)*25;p=np.tile(y+2,(5,1))
        mask=np.ones(n,dtype=bool);mask[19]=False
        r=evaluate(dates,y,s,{'original_windnet':p,'M13':p},mask)
        self.assertEqual(r['coverage']['retained_unique_target_days'],56)
        self.assertEqual(r['overall']['M13']['unique_target_days'],56)
        self.assertFalse(r['strict_all_four_numerical_win'])
        self.assertEqual(r['paired_uncertainty']['metrics']['mean_rmse']['M13_minus_original_delta_95'],[0,0])

    def test_bootstrap_does_not_pair_across_missing_calendar_units(self):
        dates=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-01')+70)
        kept=[np.arange(0,14),np.arange(28,42),np.arange(42,56),np.arange(56,70)]
        units=bootstrap_units(dates,kept)
        self.assertEqual([v.tolist() for v in units[0]],[[0],[1,2],[3]])

if __name__=='__main__':unittest.main()
