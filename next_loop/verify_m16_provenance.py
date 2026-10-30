"""Check frozen provenance and the saved zero-penalty control, without holdouts."""
import hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'results/next_loop'
    read = lambda name: json.loads((out/name).read_text(encoding='utf8'))
    result, frozen, pre, zero = [read('m16_'+s+'.json') for s in ['results', 'freeze', 'preflight', 'zero_control']]
    protocol_path = ROOT/'next_loop/m16_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf8'))
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    assert digest(protocol_path.read_bytes()) == frozen['protocol_sha256']
    assert result['freeze_sha256'] == pre['freeze_sha256'] == digest((out/'m16_freeze.json').read_bytes())
    assert protocol['frozen_sources'] == frozen['source_hashes'] == result['source_hashes']
    for name, expected in frozen['source_hashes'].items():assert digest((ROOT/name).read_bytes()) == expected, name
    assert len(frozen['development_input_hashes']) == 108 and len(frozen['panel']) == 144
    assert pre['passed'] and pre['four_controls_match_1e_9'] and pre['sources_match']
    assert pre['row_audits'] == frozen['row_audits'] == result['row_audits']
    assert pre['controls'] == frozen['controls'] == result['controls']
    assert result['raw_recipes'] == 36 and result['fold_fits'] == 144
    for record in [pre, frozen, result]:assert record['original_final_evaluations'] == 0 and not record['fold5_available']
    assert result['pair_audits'] == pre['pair_audits'] == frozen['pair_audits']
    for row in result['pair_audits']:
        assert row['fit_rows'] == row['fitting_blocks']*13
        assert row['adjacent_pairs'] == row['fitting_blocks']*12 and row['pairs_per_fitting_block'] == 12
        assert row['max_target_offset'] == 18 and row['cross_block_pairs'] == row['validation_pairs'] == 0
    assert zero['passed'] and zero['positive_strength_fits_before_check'] == 0 and zero['cached_zero_fits'] == 4
    assert zero['checks'] == result['zero_control']
    with tarfile.open(ROOT/'artifacts/next_loop/m13_training_artifacts.tar.gz') as old_archive, tarfile.open(ROOT/'artifacts/next_loop/m16_training_artifacts.tar.gz') as new_archive:
        assert sorted(new_archive.getnames()) == ['m16_models.joblib', 'm16_predictions.npz']
        assert digest((ROOT/'artifacts/next_loop/m13_training_artifacts.tar.gz').read_bytes()) == frozen['control_artifact_hashes']['m13_training_artifacts.tar.gz']
        with np.load(io.BytesIO(old_archive.extractfile('m13_predictions.npz').read()), allow_pickle=False) as old, np.load(io.BytesIO(new_archive.extractfile('m16_predictions.npz').read()), allow_pickle=False) as new:
            parent = json.loads((ROOT/'next_loop/m14_reference.json').read_text(encoding='utf8'))['selected']['model']
            differences = []
            for f in range(1, 5):
                difference = float(np.max(np.abs(new[f'M16_regional_trend_w4_targetfast2_a100_d0_share0.25_cv{f}'] - old[parent+f'_cv{f}'])))
                assert difference <= 1e-7 and difference == zero['checks'][f-1]['max_abs_prediction_difference']
                differences.append(difference)
    report = {'passed': True, 'frozen_sources': len(frozen['source_hashes']), 'conditions': 144, 'raw_recipes': 36, 'fold_fits': 144, 'original_development_blocks': 108, 'four_controls_match': True, 'pair_boundaries_verified_from_audits': True, 'zero_control_verified_from_saved_training_predictions': True, 'zero_control_max_abs_differences': differences, 'original_final_evaluations': 0, 'fold5_available': False, 'scope': 'Source/freeze/control provenance and saved zero-penalty identity; no independent full model refit or raw-label metric rerun'}
    (out/'m16_provenance_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':main()
