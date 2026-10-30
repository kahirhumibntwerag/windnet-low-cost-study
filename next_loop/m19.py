"""Registered training-only weighted linear/polynomial interaction head comparison."""
import json, platform, tarfile, time
import joblib, numpy as np, scipy, sklearn
from sklearn.preprocessing import StandardScaler
from matched.runner import DATA, EXPORT, ROOT, save_json, sha
from next_loop.m12 import load_blocks, samples
from next_loop.m14 import controls
from next_loop.regional_features import regions, temporal
from next_loop.digital_topology import feature_families
from next_loop.target_weights import target_weights
from next_loop.regional_stats import metrics, aggregate, gate, stability, bootstrap
from next_loop.m19_kernel import prepare_kernel, fit_centered_kernel, predict_centered_kernel


def controls_m19(datasets):
    import io
    rows, cp, audits = controls(datasets)
    archive = DATA/'m18_training_artifacts.tar.gz'
    assert sha(archive.read_bytes()) == '2062eed10b01159d6c0d5b5a2d61c45778b259ec8e069e82a14293f3a4ecaaeb'
    ref_path = DATA/'m18_results.json'
    assert sha(ref_path.read_bytes()) == '349abe46bd111bdec4dccd80daac1f743c0b27eec3a4bc50cf118ed16059d1ba'
    reference = json.loads(ref_path.read_text())['selected']
    assert reference['model'] == 'M18_regional_morphology_poly_g1_a1_share0.5'
    with tarfile.open(archive) as tf: raw = tf.extractfile('m18_predictions.npz').read()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        pp = {f: z[reference['model']+f'_cv{f}'].copy() for f in range(1,5)}
    fm = [metrics(datasets[f][1]['y'], pp[f], datasets[f][1]['sigma']) for f in range(1,5)]
    for i in range(4):
        for k in ['rmse','r','chi2','hse_ts']: assert abs(fm[i][k]-reference['fold_metrics'][i][k]) < 1e-9
    row = {'model':'fixed_M18','metrics':aggregate(fm),'fold_metrics':fm}
    for k in ['rmse','r','chi2','hse_ts']: assert abs(row['metrics'][k]-reference['metrics'][k]) < 1e-9
    rows.append(row); cp['fixed_M18'] = pp
    return rows, cp, audits


def panel():
    return [{'family': family, 'kernel': kernel, 'kernel_factor': factor, 'ridge_alpha': alpha,
             'new_head_share': share, 'view': 'trend', 'encoding': 'linear', 'precision_lambda': 4,
             'target_weight': 'fast2', 'gain': 1.2, 'bias': -5, 'prior': 'fixed M18'}
            for family in ['regional', 'morphology', 'regional_morphology']
            for kernel in ['cubic'] for factor in [.5, 1, 2]
            for alpha in [.1, 1, 10] for share in [.125, .25, .5, 1]]


def freeze(protocol, audits, control_rows):
    assert not (DATA/'sealed').exists() and not (DATA/'m19.started.json').exists()
    assert all(sha((ROOT/name).read_bytes()) == digest for name, digest in protocol['frozen_sources'].items())
    members = json.loads((DATA/'members.json').read_text())
    expected = {m['member']: m for m in json.loads((ROOT/'matched/source_manifest.json').read_text())['members']}
    assert len(members) == 157
    for m in members:
        assert m['sha256'] == expected[m['member']]['release_member_sha256']
        if m['fold'] == 5:assert not m['retained'] and not m['parsed_for_development']
    record = {'stage': 'M19', 'scope': 'Original Folds1-4 training-only kernel comparison',
              'protocol_sha256': sha((ROOT/'next_loop/m19_protocol.json').read_bytes()), 'source_hashes': protocol['frozen_sources'], 'panel': panel(),
              'members_sha256': sha((DATA/'members.json').read_bytes()),
              'development_input_hashes': {str(p.relative_to(DATA)): sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))},
              'control_artifact_hashes': {name: sha((DATA/name).read_bytes()) for name in ['m11_training_artifacts.tar.gz', 'm13_training_artifacts.tar.gz', 'm18_training_artifacts.tar.gz']},
              'row_audits': audits, 'controls': control_rows, 'original_final_evaluations': 0, 'fold5_available': False,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    assert len(record['development_input_hashes']) == 108 and len(record['panel']) == 108
    with (DATA/'m19.started.json').open('x') as handle:json.dump(record, handle, indent=2, allow_nan=False)
    (EXPORT/'m19_freeze.json').write_bytes((DATA/'m19.started.json').read_bytes())
    save_json(EXPORT/'m19_preflight.json', {'passed': True, 'freeze_sha256': sha((DATA/'m19.started.json').read_bytes()), 'five_controls_match_1e_9': True, 'sources_match': True, 'row_audits': audits, 'controls': control_rows, 'original_final_evaluations': 0, 'fold5_available': False})
    return record


def run():
    started = time.time();blocks = load_blocks();datasets = {f: samples(blocks, f) for f in range(1, 5)}
    control_rows, cp, audits = controls_m19(datasets);protocol = json.loads((ROOT/'next_loop/m19_protocol.json').read_text())
    frozen = freeze(protocol, audits, control_rows)
    print('M19 five controls, original rows, data and full source/panel freeze verified before fitting', flush=True)
    features = {'regional': [], 'morphology': [], 'regional_morphology': []}
    for i, b in enumerate(blocks):
        ff = feature_families(b['images']);features['regional'].append(regions(b['images'], (6, 6)))
        for family in ['morphology', 'regional_morphology']:features[family].append(ff[family])
        if (i+1) % 30 == 0:print(json.dumps({'M19_feature_blocks': i+1, 'seconds': time.time()-started}), flush=True)
    features = {family: np.stack(frames) for family, frames in features.items()}
    prepared, kernel_states, grams, models, raw_predictions = {}, {}, {}, {}, {}
    all_predictions, recipes, kernel_audits = {}, {}, []

    def prepare(family, f):
        key = family+f'_cv{f}'
        if key not in prepared:
            tr, va = datasets[f];frames = features[family]
            x, v = [temporal(np.stack([frames[int(b), int(o)-6:int(o)-2] for b, o in zip(a['block'], a['offset'])]), 'trend', 'linear') for a in (tr, va)]
            scaler = StandardScaler().fit(x);scaler.scale_ = np.maximum(scaler.scale_, .01);x, v = scaler.transform(x), scaler.transform(v)
            sig = tr['sigma'][tr['sigma'] > 0];floor = float(np.quantile(sig, .05));median = float(np.median(sig))
            weights = 1+4*(median/np.maximum(tr['sigma'], floor))**2;weights /= weights.mean()
            weights *= target_weights(tr['y'], 'fast2');weights /= weights.mean()
            prepared[key] = {'x': x, 'v': v, 'weights': weights, 'scaler': scaler, 'target_mean': float(tr['y'].mean()), 'sigma_floor': floor, 'sigma_median': median}
        return key, prepared[key]

    def fit(family, kernel, factor, alpha, f):
        base, data = prepare(family, f);state_key = base+f'_{kernel}_g{factor}';model_key = state_key+f'_a{alpha}'
        if state_key not in kernel_states:
            state, k = prepare_kernel(data['x'], data['weights'], kernel, factor)
            residual = float(np.max(np.abs(k@state['a'])))
            assert residual <= 1e-9
            kernel_states[state_key] = {**state, 'feature_state': base};grams[state_key] = k
            kernel_audits.append({'state': state_key, 'fold': f, 'family': family, 'kernel': kernel, 'features': data['x'].shape[1], 'fitting_rows': len(data['x']), 'kernel_factor': factor, 'gamma': state.get('gamma'), 'distance_median': state.get('distance_median'), 'max_weighted_center_residual': residual, 'fit_only': True})
        if model_key not in models:
            state = kernel_states[state_key];fit = fit_centered_kernel(state, grams[state_key], datasets[f][0]['y'], data['weights'], alpha)
            models[model_key] = {**fit, 'kernel_state': state_key, 'gain': 1.2, 'bias': -5, 'target_mean': data['target_mean']}
            mu = data['target_mean'];raw_predictions[model_key] = mu+1.2*(predict_centered_kernel(fit, state, data['v'])-mu)-5
        return raw_predictions[model_key]

    for recipe in frozen['panel']:
        family, kernel, factor, alpha, share = [recipe[k] for k in ['family', 'kernel', 'kernel_factor', 'ridge_alpha', 'new_head_share']]
        name = f'M19_{family}_{kernel}_g{factor}_a{alpha}_share{share}'
        all_predictions[name] = {f: share*fit(family, kernel, factor, alpha, f)+(1-share)*cp['fixed_M18'][f] for f in range(1, 5)}
        recipes[name] = recipe
    assert len(models) == 108 and len(all_predictions) == 108 and len(kernel_states) == 36
    rows = []
    for name, pp in all_predictions.items():
        fm = [metrics(datasets[f][1]['y'], pp[f], datasets[f][1]['sigma']) for f in range(1, 5)]
        row = {'model': name, 'recipe': recipes[name], 'fold_metrics': fm, 'metrics': aggregate(fm)}
        row['gate'] = gate(row['metrics'], [c['metrics'] for c in control_rows]);row['stability'] = stability(row, control_rows);rows.append(row)
    selected, checks = None, []
    for row in sorted((r for r in rows if r['gate']['all_four'] and r['stability']['pre_bootstrap_passed']), key=lambda r: r['metrics']['rmse']):
        intervals = {name: bootstrap(all_predictions[row['model']], pp, datasets=datasets) for name, pp in cp.items()}
        passed = all(v['hse_ts']['delta_95'] is not None and v['hse_ts']['delta_95'][0] > 0 for v in intervals.values());checks.append({'model': row['model'], 'bootstrap': intervals, 'passed': passed})
        if passed:selected = row;break
    assert not (DATA/'sealed').exists() and all(sha((ROOT/name).read_bytes()) == digest for name, digest in frozen['source_hashes'].items())
    np.savez_compressed(DATA/'m19_predictions.npz', **{name+f'_cv{f}': p for name, pp in all_predictions.items() for f, p in pp.items()})
    feature_states = {key: {k: d[k] for k in ['scaler', 'target_mean', 'sigma_floor', 'sigma_median']} for key, d in prepared.items()}
    joblib.dump({'models': models, 'kernel_states': kernel_states, 'feature_states': feature_states}, DATA/'m19_models.joblib', compress=3)
    result = {'stage': 'M19', 'scope': 'Original Folds1-4 training-bank validation only', 'rows': rows, 'controls': control_rows, 'selected': selected, 'gate_passed': selected is not None, 'bootstrap_checks': checks, 'row_audits': audits, 'kernel_audits': kernel_audits, 'raw_recipes': 27, 'fold_fits': len(models), 'kernel_states': len(kernel_states), 'original_final_evaluations': 0, 'fold5_available': False, 'freeze_sha256': sha((DATA/'m19.started.json').read_bytes()), 'source_hashes': frozen['source_hashes'], 'versions': frozen['runtime'], 'seconds': time.time()-started, 'limitations': 'Adaptively reused development bank; conditional intervals not selection adjusted. Deterministic kernel regression is not Gaussian-process uncertainty estimation. No historical feedback or additional predictor data.'}
    save_json(EXPORT/'m19_results.json', result)
    target = EXPORT/'m19_training_artifacts.tar.gz'
    with tarfile.open(target, 'w:gz') as archive:
        for name in ['m19_models.joblib', 'm19_predictions.npz']:archive.add(DATA/name, arcname=name)
    save_json(EXPORT/'m19_artifacts.json', {'filename': target.name, 'sha256': sha(target.read_bytes()), 'bytes': target.stat().st_size, 'models': 108, 'candidate_recipes': 27, 'conditions': 108, 'scope': 'Fitted heads, fitting kernel features/centering statistics and training validation forecasts; fixed M18 and M13 parent models required; no original holdout data'})
    print(json.dumps({'M19_complete': True, 'conditions': len(rows), 'fold_fits': len(models), 'numerical_passes': sum(r['gate']['all_four'] for r in rows), 'qualified': selected is not None, 'selected': None if selected is None else selected['model'], 'seconds': time.time()-started}), flush=True)


if __name__ == '__main__':run()
