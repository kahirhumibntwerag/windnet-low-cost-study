"""Registered training-only adjacent-window difference ridge ablation."""
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
from next_loop.difference_ridge import adjacent_pairs, fit_difference_ridge, predict_difference_ridge


def panel():
    return [{'family': family, 'ridge_alpha': alpha, 'difference_strength': strength,
             'new_head_share': share, 'view': 'trend', 'encoding': 'linear',
             'precision_lambda': 4, 'target_weight': 'fast2', 'gain': 1.2, 'bias': -5, 'prior': 'fixed M10'}
            for family in ['regional', 'morphology', 'regional_morphology']
            for alpha in [10, 100, 1000] for strength in [0, .25, 1, 4]
            for share in [.125, .25, .5, 1]]


def freeze(protocol, audits, control_rows, pair_audits):
    assert not (DATA/'sealed').exists() and not (DATA/'m16.started.json').exists()
    assert all(sha((ROOT/name).read_bytes()) == digest for name, digest in protocol['frozen_sources'].items())
    members = json.loads((DATA/'members.json').read_text())
    expected = {m['member']: m for m in json.loads((ROOT/'matched/source_manifest.json').read_text())['members']}
    assert len(members) == 157
    for m in members:
        assert m['sha256'] == expected[m['member']]['release_member_sha256']
        if m['fold'] == 5:assert not m['retained'] and not m['parsed_for_development']
    record = {'stage': 'M16', 'scope': 'Original Folds1-4 training-only difference ridge',
              'protocol_sha256': sha((ROOT/'next_loop/m16_protocol.json').read_bytes()),
              'source_hashes': protocol['frozen_sources'], 'panel': panel(),
              'members_sha256': sha((DATA/'members.json').read_bytes()),
              'development_input_hashes': {str(p.relative_to(DATA)): sha(p.read_bytes()) for p in sorted((DATA/'development').glob('*.npz'))},
              'control_artifact_hashes': {name: sha((DATA/name).read_bytes()) for name in ['m11_training_artifacts.tar.gz', 'm13_training_artifacts.tar.gz']},
              'row_audits': audits, 'controls': control_rows, 'pair_audits': pair_audits,
              'original_final_evaluations': 0, 'fold5_available': False,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    assert len(record['development_input_hashes']) == 108 and len(record['panel']) == 144
    with (DATA/'m16.started.json').open('x') as handle:json.dump(record, handle, indent=2, allow_nan=False)
    (EXPORT/'m16_freeze.json').write_bytes((DATA/'m16.started.json').read_bytes())
    save_json(EXPORT/'m16_preflight.json', {'passed': True, 'freeze_sha256': sha((DATA/'m16.started.json').read_bytes()), 'four_controls_match_1e_9': True, 'sources_match': True, 'row_audits': audits, 'pair_audits': pair_audits, 'controls': control_rows, 'original_final_evaluations': 0, 'fold5_available': False})
    return record


def run():
    started = time.time();blocks = load_blocks();datasets = {f: samples(blocks, f) for f in range(1, 5)}
    control_rows, cp, audits = controls(datasets);pair_audits = []
    for f, (tr, va) in datasets.items():
        left, right = adjacent_pairs(tr['block'], tr['offset'])
        assert len(left) == len(np.unique(tr['block'])) * 12
        for b in np.unique(tr['block']):assert np.array_equal(tr['offset'][tr['block'] == b], np.arange(6, 19))
        assert np.all(tr['block'][left] == tr['block'][right]) and np.all(tr['offset'][right] - tr['offset'][left] == 1)
        assert not set(tr['row']) & set(va['row'])
        pair_audits.append({'held_training_fold': f, 'fit_rows': len(tr['y']), 'fitting_blocks': len(np.unique(tr['block'])), 'adjacent_pairs': len(left), 'pairs_per_fitting_block': 12, 'cross_block_pairs': 0, 'validation_pairs': 0, 'max_target_offset': int(tr['offset'].max())})
    protocol = json.loads((ROOT/'next_loop/m16_protocol.json').read_text());frozen = freeze(protocol, audits, control_rows, pair_audits)
    print('M16 controls, original rows, fitting pair boundaries and full source/panel freeze verified before fitting', flush=True)
    features = {'regional': np.stack([regions(b['images'], (6, 6)) for b in blocks])}
    prepared = {};models = {};all_predictions = {};recipes = {};fit_count = 0

    def prepare(family, f):
        key = (family, f)
        if key in prepared:return prepared[key]
        tr, va = datasets[f];frames = features[family]
        x, v = [temporal(np.stack([frames[int(b), int(o)-6:int(o)-2] for b, o in zip(a['block'], a['offset'])]), 'trend', 'linear') for a in (tr, va)]
        scaler = StandardScaler().fit(x);scaler.scale_ = np.maximum(scaler.scale_, .01);x, v = scaler.transform(x), scaler.transform(v)
        sig = tr['sigma'][tr['sigma'] > 0];floor = float(np.quantile(sig, .05));med = float(np.median(sig))
        weights = 1 + 4 * (med / np.maximum(tr['sigma'], floor))**2;weights /= weights.mean()
        weights *= target_weights(tr['y'], 'fast2');weights /= weights.mean()
        prepared[key] = (x, v, scaler, weights, floor, med)
        return prepared[key]

    def fit(family, alpha, strength, f):
        nonlocal fit_count
        name = f'M16_{family}_trend_w4_targetfast2_a{alpha}_d{strength}'
        key = name + f'_cv{f}'
        if key not in models:
            x, v, scaler, weights, floor, med = prepare(family, f);tr = datasets[f][0]
            fitted = fit_difference_ridge(x, tr['y'], weights, tr['block'], tr['offset'], alpha, strength)
            models[key] = {'difference_model': fitted, 'scaler': scaler, 'target_mean': float(tr['y'].mean()), 'family': family, 'view': 'trend', 'encoding': 'linear', 'gain': 1.2, 'bias': -5, 'sigma_floor': floor, 'sigma_median': med}
            fit_count += 1
        model = models[key];v = prepare(family, f)[1];mu = model['target_mean']
        return mu + 1.2 * (predict_difference_ridge(model['difference_model'], v) - mu) - 5

    # Reproduce the incumbent before computing ANY positive-penalty candidate.
    zero_checks = []
    for f in range(1, 5):
        pred = .25 * fit('regional', 100, 0, f) + .75 * cp['fixed_M10'][f]
        diff = float(np.max(np.abs(pred - cp['fixed_M13'][f])));assert diff <= 1e-7, ('zero-strength mismatch', f, diff)
        zero_checks.append({'held_training_fold': f, 'max_abs_prediction_difference': diff, 'passed': True})
    save_json(EXPORT/'m16_zero_control.json', {'passed': True, 'tolerance': 1e-7, 'checks': zero_checks, 'positive_strength_fits_before_check': 0, 'cached_zero_fits': fit_count})
    print('M16 zero-strength prediction check passed on every original training validation row; now testing positive penalties', flush=True)
    extra = {family: [] for family in ['morphology', 'regional_morphology']}
    for i, b in enumerate(blocks):
        ff = feature_families(b['images'])
        for family in extra:extra[family].append(ff[family])
        if (i+1) % 30 == 0:print(json.dumps({'M16_feature_blocks': i+1, 'seconds': time.time()-started}), flush=True)
    features.update({family: np.stack(values) for family, values in extra.items()})
    for recipe in frozen['panel']:
        family, alpha, strength, share = [recipe[k] for k in ['family', 'ridge_alpha', 'difference_strength', 'new_head_share']]
        name = f'M16_{family}_trend_w4_targetfast2_a{alpha}_d{strength}_share{share}'
        pp = {f: share * fit(family, alpha, strength, f) + (1-share) * cp['fixed_M10'][f] for f in range(1, 5)}
        all_predictions[name] = pp;recipes[name] = recipe
    assert fit_count == 144 and len(models) == 144 and len(all_predictions) == 144
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
    np.savez_compressed(DATA/'m16_predictions.npz', **{name+f'_cv{f}': p for name, pp in all_predictions.items() for f, p in pp.items()});joblib.dump(models, DATA/'m16_models.joblib', compress=3)
    result = {'stage': 'M16', 'scope': 'Original Folds1-4 training-bank validation only', 'rows': rows, 'controls': control_rows, 'selected': selected, 'gate_passed': selected is not None, 'bootstrap_checks': checks, 'row_audits': audits, 'pair_audits': pair_audits, 'zero_control': zero_checks, 'raw_recipes': 36, 'fold_fits': fit_count, 'original_final_evaluations': 0, 'fold5_available': False, 'freeze_sha256': sha((DATA/'m16.started.json').read_bytes()), 'source_hashes': frozen['source_hashes'], 'versions': frozen['runtime'], 'seconds': time.time()-started, 'limitations': 'Reused development bank; conditional intervals not selection adjusted. Adjacent original-window differences are not physical per-day derivatives. No runtime neighboring forecasts or historical F3 feedback.'}
    save_json(EXPORT/'m16_results.json', result)
    target = EXPORT/'m16_training_artifacts.tar.gz'
    with tarfile.open(target, 'w:gz') as archive:
        for name in ['m16_models.joblib', 'm16_predictions.npz']:archive.add(DATA/name, arcname=name)
    save_json(EXPORT/'m16_artifacts.json', {'filename': target.name, 'sha256': sha(target.read_bytes()), 'bytes': target.stat().st_size, 'models': 144, 'candidate_recipes': 36, 'conditions': 144, 'scope': 'Training-side fitted heads and validation predictions only; fixed M10 parent models required for inference; no original holdout data'})
    print(json.dumps({'M16_complete': True, 'conditions': len(rows), 'fold_fits': fit_count, 'numerical_passes': sum(r['gate']['all_four'] for r in rows), 'qualified': selected is not None, 'selected': None if selected is None else selected['model'], 'seconds': time.time()-started}), flush=True)


if __name__ == '__main__':run()
