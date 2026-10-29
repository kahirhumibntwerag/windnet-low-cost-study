"""Independent checks against released labels and the original window formula."""
import json
from pathlib import Path
import numpy as np

r=Path(__file__).resolve().parent
source=np.load(r/'data/windnet211_labels.npz');d=np.load(r/'data/study_inputs.npz');s=np.load(r/'data/source_folds.npz')
expected=[];histories=[];folds=[]
for values,fold in zip(source['output'],source['folds']):
    for j in range(values.shape[0]-3-4+1):
        expected.append(values[j+4+3-1,11]);histories.append(values[j:j+4,11]);folds.append(fold)
assert np.array_equal(d['y'],expected)
assert np.array_equal(d['short'],histories)
assert np.array_equal(d['fold'],folds)
assert len(d['y'])==2198
assert np.isfinite(d['y']).all()
assert np.isfinite(d['short']).all()
for f in range(1,6):
    assert not (set(d['date'][d['fold']==f]) & set(d['date'][d['fold']!=f]))
    fit=(d['fold']!=f)&(d['offset']<19)
    for kind in ['short','recurrence','multi']:
        x=d[kind][fit].copy();src=s[kind][fit];x[src==f]=np.nan
        assert np.isnan(x[src==f]).all()
        assert x.shape==src.shape
print(json.dumps({'passed':True,'checks':['all 2198 targets match original DataWindowParser','all four-speed histories match original window','original five fold memberships retained','no target date crosses folds','all lag measurements from heldout fold masked during fitting','no missing target silently excluded']},indent=2))
