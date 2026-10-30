"""Compare training-only control reproduction and immutable source identities."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/next_loop'
old=json.loads((ROOT/'results/event_loop/event_controls.json').read_text())
new=json.loads((R/'event_controls.json').read_text())
assert [(x['fit_row_sha256'],x['validation_row_sha256']) for x in old['audits']]==[(x['fit_row_sha256'],x['validation_row_sha256']) for x in new['audits']]
protocol=json.loads((ROOT/'next_loop/protocol.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['frozen_sources'].items())
diff=[]
for a,b in zip(new['rows'],old['rows']):
    for key in ['rmse','r','chi2','hse_ts']:
        tolerance=1e-5 if key in ['r','hse_ts'] else .001
        diff.append({'model':a['model'],'metric':key,'archived_training':b['metrics'][key],'current_training':a['metrics'][key],'delta':a['metrics'][key]-b['metrics'][key],'tolerance':tolerance,'passes':abs(a['metrics'][key]-b['metrics'][key])<=tolerance})
d={'kind':'technical_reproduction_observation','identical_training_row_hashes':True,'frozen_scientific_sources_unchanged':True,'archived_environment':{k:v for k,v in old['identity'].items() if not isinstance(v,dict)},'current_environment':{k:v for k,v in new['identity'].items() if not isinstance(v,dict)},'comparisons':diff,'passed':all(x['passes'] for x in diff),'cause':'Not isolated. Matching recorded versions and seeds do not prove cross-hardware numerical reproducibility.','scientific_hypothesis_verdict':'Not assessed. Registered run cannot adopt a candidate after control reproduction failure.','original_final_evaluations':0,'literature_after_observation':['https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md']}
(R/'technical_observation.json').write_text(json.dumps(d,indent=2))
(ROOT/'next_loop/execution.json').write_text(json.dumps({'status':'running','observation':'Reproduction gate flagged: training rows and source hashes match, recorded versions match, but controls differ beyond tolerance. No scientific hypothesis decision or new original test scoring.'},indent=2))
print(json.dumps({'control_reproduction_passed':d['passed'],'row_hashes_match':True,'frozen_sources_unchanged':True,'failed_comparisons':sum(not x['passes'] for x in diff)}))
