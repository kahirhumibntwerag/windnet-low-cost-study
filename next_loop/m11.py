"""Registered six-condition calibration experiment; no original final data."""
import json, time
import numpy as np
from matched.runner import DATA, EXPORT, ROOT, save_json, sha
from matched.science import metrics, rowhash
from event_loop.science import split, aggregate, gate, save_predictions
from event_loop.location import representation, temporal, ridge_path
from event_loop.precision import stability
from event_loop.mixtures import bootstrap
from next_loop.calibration import fit_map, predict_map

def run(paired_restart=False):
    start = time.time()
    assert not (DATA / 'sealed').exists()
    members = json.loads((DATA / 'members.json').read_text())
    assert all(not x['retained'] and not x['parsed_for_development'] for x in members if x['fold'] == 5)
    controls = json.loads((EXPORT / 'event_controls.json').read_text())['rows']
    prior = np.load(DATA / 'bank_prior_M6.npz')
    datasets = {f: split(f)[:2] for f in range(1, 5)}
    conditions = {f'crossfit_strength_{s}': {} for s in [0, .25, .5, .75, 1]}
    conditions['in_sample_map_ablation'] = {}
    audits, mapping_records = [], []
    for held, (tr, va) in datasets.items():
        d = np.load(DATA / f'location_cv{held}.npz')
        frames = representation(d['local'], d['glob'], 'ratio')
        x, v = [temporal(np.stack([frames[int(b), int(o)-6:int(o)-2] for b, o in zip(a['block'], a['offset'])]), 'flatten') for a in [tr, va]]
        pred = ridge_path(x, v, tr['y'], tr['sigma'], 4, [10000])[10000]
        oof = np.full(len(x), np.nan)
        inner_audit = []
        for inner in sorted(set(tr['fold'])):
            validation = tr['fold'] == inner
            fit = ~validation
            assert not set(tr['block'][fit]) & set(tr['block'][validation])
            oof[validation] = ridge_path(x[fit], x[validation], tr['y'][fit], tr['sigma'][fit], 4, [10000])[10000]
            inner_audit.append({'held_fitting_fold': int(inner), 'fit_rows': int(fit.sum()), 'calibration_rows': int(validation.sum()), 'disjoint_blocks': True})
        assert np.isfinite(oof).all()
        mapping = fit_map(oof, tr['y'])
        corrected = predict_map(mapping, pred)
        for s in [0, .25, .5, .75, 1]:
            location = (1-s)*pred+s*corrected
            mu = float(tr['y'].mean())
            conditions[f'crossfit_strength_{s}'][held] = mu+1.2*(.625*location+.375*prior[f'cv{held}']-mu)-5
        in_sample = ridge_path(x, x, tr['y'], tr['sigma'], 4, [10000])[10000]
        in_map = fit_map(in_sample, tr['y'])
        mu = float(tr['y'].mean())
        conditions['in_sample_map_ablation'][held] = mu+1.2*(.625*predict_map(in_map, pred)+.375*prior[f'cv{held}']-mu)-5
        audits.append({'held_training_fold': held, 'fit_row_sha256': rowhash(tr), 'validation_row_sha256': rowhash(va), 'inner_folds': inner_audit, 'encoder_centering': 'outer fitting images only; reused across inner regressions', 'inner_validation_labels_in_inner_fit': False})
        mapping_records.append({'held_training_fold': held, 'crossfit_map': mapping, 'in_sample_map': in_map})
        print(json.dumps({'M11_outer_fold_complete': held, 'seconds': time.time()-start}), flush=True)
    def score(name, pp):
        fm = [metrics(datasets[f][1]['y'], pp[f], datasets[f][1]['sigma']) for f in range(1, 5)]
        return {'model': name, 'fold_metrics': fm, 'metrics': aggregate(fm)}
    fixed = score('fixed_M10', conditions['crossfit_strength_0'])
    # Reproduction checks against training-side exports only.
    reference = json.loads((ROOT / 'next_loop/training_reference.json').read_text())
    for left, right in zip(audits, reference['row_audits']):
        assert left['fit_row_sha256'] == right['fit_row_sha256']
        assert left['validation_row_sha256'] == right['validation_row_sha256']
    reproduction = []
    for actual, expected in [(fixed, reference['selected']), *zip(controls, reference['controls'])]:
        for key in ['rmse', 'r', 'chi2', 'hse_ts']:
            tolerance = 1e-5 if key in ['r', 'hse_ts'] else .001
            passes = abs(actual['metrics'][key]-expected['metrics'][key]) <= tolerance
            reproduction.append({'model':actual['model'],'metric':key,'current':actual['metrics'][key],'archived':expected['metrics'][key],'tolerance':tolerance,'passes':passes})
            if not paired_restart:
                assert passes, (actual['model'], key, actual['metrics'][key], expected['metrics'][key])
    controls = [*controls, fixed]
    rows = []
    for name, pp in conditions.items():
        row = score(name, pp)
        row['recipe'] = {'calibration': name, 'location': 'fixed M10 ratio/flatten lambda4 alpha10000', 'location_weight': .625, 'gain': 1.2, 'bias': -5}
        row['gate'] = gate(row['metrics'], [c['metrics'] for c in controls])
        row['stability'] = stability(row, controls)
        row['eligible_family'] = name.startswith('crossfit') and name != 'crossfit_strength_0'
        rows.append(row)
        save_predictions('M11_'+name, pp)
    cps = {c['model']: {f: np.load(DATA / (c['model']+'.npz'))[f'cv{f}'] for f in range(1,5)} for c in controls[:2]}
    cps['fixed_M10'] = conditions['crossfit_strength_0']
    checks, selected = [], None
    for row in sorted((r for r in rows if r['eligible_family'] and r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']), key=lambda r:r['metrics']['rmse']):
        intervals = {name: bootstrap(conditions[row['model']], pp, datasets=datasets) for name, pp in cps.items()}
        passed = all(c['hse_ts']['delta_95'][0] > 0 for c in intervals.values())
        checks.append({'model': row['model'], 'bootstrap': intervals, 'passed': passed})
        if passed:
            selected = row
            break
    identity = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for pkg in ['matched', 'event_loop', 'next_loop'] for p in sorted((ROOT/pkg).glob('*')) if p.suffix in ['.py', '.json']}
    stage = 'M11R' if paired_restart else 'M11'
    result = {'stage': stage, 'scope': 'Training-bank Folds1-4 only; within-platform paired comparison', 'rows': rows, 'controls': controls, 'row_audits': audits, 'mapping_records': mapping_records, 'selected': selected, 'gate_passed': selected is not None, 'bootstrap_checks': checks, 'source_hashes': identity, 'archived_control_reproduction':reproduction, 'paired_restart':paired_restart, 'seconds': time.time()-start, 'fold5_available': False, 'original_final_evaluations': 0, 'limitations': 'Adaptive reused training bank; conditional intervals not selection adjusted; inner regression is cross-fitted but fixed encoder centering uses all outer fitting images. Quantile clipping may suppress extremes. No test-generalization claim. Current-platform controls do not numerically reproduce archived controls; original M11 failure is retained.'}
    save_json(EXPORT / (stage.lower()+'_results.json'), result)
    print(json.dumps({'M11_complete': True, 'selected': selected, 'metrics': [{'model': r['model'], 'metrics': r['metrics'], 'gate': r['gate']} for r in rows], 'seconds':time.time()-start}), flush=True)

if __name__ == '__main__':
    run()
