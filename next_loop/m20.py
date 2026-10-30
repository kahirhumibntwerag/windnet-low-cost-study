"""Fixed M18 tail transformations on the original development bank."""
import hashlib, json, tarfile, time
import numpy as np
from matched.runner import DATA, EXPORT, ROOT, save_json, sha
from next_loop.m12 import load_blocks, samples
from next_loop.m19 import controls_m19
from next_loop.regional_stats import metrics, aggregate, gate, stability, bootstrap


def transform(p, a, b):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(a) or not np.isfinite(b):
        raise ValueError('Nonfinite transformation input')
    return p + a*np.maximum(p-450, 0) + b*np.maximum(400-p, 0)


def panel():
    return [{'a': a, 'b': b, 'upper_hinge': 450, 'lower_hinge': 400}
            for a in [.1, .2, .4] for b in [-.1, 0, .1]]


def run():
    t = time.time(); blocks = load_blocks(); datasets = {f:samples(blocks, f) for f in range(1,5)}
    control_rows, cp, audits = controls_m19(datasets)
    protocol = json.loads((ROOT/'next_loop/m20_protocol.json').read_text())
    assert not (DATA/'sealed').exists() and not (DATA/'m20.started.json').exists()
    assert all(sha((ROOT/p).read_bytes()) == h for p,h in protocol['frozen_sources'].items())
    members = json.loads((DATA/'members.json').read_text())
    assert len(members) == 157 and all(not m['retained'] and not m['parsed_for_development'] for m in members if m['fold']==5)
    original = json.loads((DATA/'m18_freeze.json').read_text())
    hashes = {str(p.relative_to(DATA)):sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))}
    assert len(hashes) == 108 and hashes == original['development_input_hashes']
    assert sha((DATA/'members.json').read_bytes()) == original['members_sha256']
    freeze = {'stage':'M20','panel':panel(),'protocol_sha256':sha((ROOT/'next_loop/m20_protocol.json').read_bytes()),
              'source_hashes':protocol['frozen_sources'],'row_audits':audits,'controls':control_rows,
              'development_input_hashes':hashes,'members_sha256':sha((DATA/'members.json').read_bytes()),
              'm18_archive_sha256':sha((DATA/'m18_training_artifacts.tar.gz').read_bytes()),
              'm18_results_sha256':sha((DATA/'m18_results.json').read_bytes()),
              'fold5_available':False,'original_final_evaluations':0}
    with (DATA/'m20.started.json').open('x') as f: json.dump(freeze,f,indent=2,allow_nan=False)
    (EXPORT/'m20_freeze.json').write_bytes((DATA/'m20.started.json').read_bytes())
    print('M20 five controls, source and complete nine-condition panel frozen',flush=True)
    rows=[]; predictions={}
    for recipe in panel():
        a,b=recipe['a'],recipe['b']; name=f'M20_a{a}_b{b}'
        pp={f:transform(cp['fixed_M18'][f],a,b) for f in range(1,5)};predictions[name]=pp
        fm=[metrics(datasets[f][1]['y'],pp[f],datasets[f][1]['sigma']) for f in range(1,5)]
        row={'model':name,'recipe':recipe,'fold_metrics':fm,'metrics':aggregate(fm)}
        row['gate']=gate(row['metrics'],[c['metrics'] for c in control_rows]);row['stability']=stability(row,control_rows);rows.append(row)
    selected=None;checks=[]
    for row in sorted((r for r in rows if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']),key=lambda r:r['metrics']['rmse']):
        intervals={n:bootstrap(predictions[row['model']],p,datasets=datasets) for n,p in cp.items()}
        passed=all(v['hse_ts']['delta_95'] is not None and v['hse_ts']['delta_95'][0]>0 for v in intervals.values())
        checks.append({'model':row['model'],'bootstrap':intervals,'passed':passed})
        if passed:selected=row;break
    assert all(sha((ROOT/p).read_bytes()) == h for p,h in freeze['source_hashes'].items())
    np.savez_compressed(DATA/'m20_predictions.npz',**{n+f'_cv{f}':p for n,pp in predictions.items() for f,p in pp.items()})
    result={'stage':'M20','scope':'Original Folds1-4 training-side validation only','rows':rows,'controls':control_rows,
            'selected':selected,'gate_passed':selected is not None,'bootstrap_checks':checks,'row_audits':audits,
            'fold5_available':False,'original_final_evaluations':0,'freeze_sha256':sha((DATA/'m20.started.json').read_bytes()),
            'seconds':time.time()-t,'limitations':'Reused training validation bank, adaptive hypotheses, conditional intervals not selection-adjusted; no external or original test.'}
    save_json(EXPORT/'m20_results.json',result)
    archive=EXPORT/'m20_training_artifacts.tar.gz'
    with tarfile.open(archive,'w:gz') as tf: tf.add(DATA/'m20_predictions.npz',arcname='m20_predictions.npz')
    save_json(EXPORT/'m20_artifacts.json',{'filename':archive.name,'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'conditions':9})
    print(json.dumps({'M20_complete':True,'conditions':len(rows),'numerical_passes':sum(r['gate']['all_four'] for r in rows),
                      'qualified':selected is not None,'selected':None if selected is None else selected['model']}),flush=True)


if __name__=='__main__': run()
