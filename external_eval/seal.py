"""Fail-closed complete freeze. This module never opens external arrays."""
import datetime as dt
import base64
import hashlib,json,subprocess,sys
from pathlib import Path
from external_eval.worker import save

EXPECTED_MODEL_LOCK='566e401d2e52feb077eddf4b22afecc4306c8f8b471fd59f8aa708117f89de02'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def sources():
    files=[Path('external_eval')/n for n in ['__init__.py','acquire_pre2019.py','author_inference.py','preflight_author.py','m13_inference.py','preflight_m13.py','preprocessing.py','data_source.py','scoring.py','test_scoring.py','worker.py','run.py','seal.py']]
    files += [Path('matched')/n for n in ['__init__.py','runner.py','science.py','hse_original.py','final_study.py','source_manifest.json','reference_sources.json','protocol.json']]
    files += [Path('event_loop')/n for n in ['__init__.py','runner.py','science.py','location.py','final.py','mixtures.py']]
    files += [Path('next_loop')/n for n in ['__init__.py','regional_features.py','target_weights.py','f3_panel.py']]
    files += [Path('external_eval/calibration/aia_V8_20171210_050627_response_table.txt'),Path('data/external_models/author_normalization.json')]
    return {str(p).replace('\\','/'):digest(p) for p in sorted(files)}

def check_models():
    assert digest(Path('external_eval/model_lock.json'))==EXPECTED_MODEL_LOCK
    lock=json.loads(Path('external_eval/model_lock.json').read_text())
    root=Path('data/external_models')
    for spec in lock['original_windnet']['checkpoints']:
        assert digest(root/f"author_fold{spec['fold']}"/spec['file'])==spec['sha256']
    for spec in lock['m13']['artifacts']:
        assert digest(root/'m13/models'/spec['file'])==spec['sha256']
    assert digest(root/'kaffe/googlenet.npy')=='d93955ecfb66c9cdbe0c038f7f47fbf971e46dee30930970e22fc4a1b8fa184f'
    assert digest(root/'kaffe/googlenet.py')=='c40b34bd4910e175ce121a41083ab5336bd6689167eb7206f63831706335c834'

def main():
    path=Path('external_eval/evaluator_freeze.json')
    assert not path.exists(),'Refuse to overwrite freeze'
    assert not Path('data/external_sealed').exists(),'External data existed before complete freeze'
    a=json.loads(Path('exports/author_restoration_preflight.json').read_text())
    m=json.loads(Path('exports/m13_inference_preflight.json').read_text())
    assert a['passed'] and m['passed'] and not a['external_data_accessed'] and not m['external_arrays_accessed']
    subprocess.check_call([sys.executable,'-m','unittest','external_eval.test_scoring','-v'])
    check_models()
    value={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'model_lock_sha256':EXPECTED_MODEL_LOCK,
           'source_hashes':sources(),'preflight_hashes':{n:digest(Path('exports')/n) for n in ['author_restoration_preflight.json','m13_inference_preflight.json','runtime_versions.json']},
           'registered_design_sha256':digest(Path('external_eval/registered_design.json')),
           'normalization':'Original image mean and target scaling from corresponding pre2019 fitting folds; original embedding constants in saved graph. M13 saved normalizations only.',
           'image_eligibility':'First available211A synoptic image nearest00UTC in same filenameUTCday; no later quality-based replacement. QUALITY0, finite1024square, level1.5, positive finite exposure/radius; published SDOMLv1 SUM256 and author ProcessImg_common. Technical unreadability or unsupported inputs excluded; transport exhaustion aborts.',
           'target_eligibility':'OMNI2 daily bulk speed word25, daily sigma word32,9999fill invalid. Finite positive speed for common block eligibility; finite positive sigma only for common chi2 mask. No new minimum hourly sample count.',
           'event_blocks':'Fixed14target-day blocks anchored2020-01-01; final short block retained. Drop whole blocks with any unavailable input/target. Never join across gaps.',
           'bootstrap_detail':'Fixed calendar28-day units anchored2020-01-01, stratified by14-day-block start year;2000 paired resamples,seed20261030,all replicas share draw. Missing calendar units never compacted; final partial units retained.',
           'annual_detail':'RMSE/r/chi2 on exact calendar-year target dates from retained common blocks. Annual HSE uses intersections of original14day blocks with each calendar year, retaining partial boundary segments.',
           'external_arrays_accessed':False,'refits_or_external_tuning':0,'external_passes_permitted':1,
           'outputs':'All ten replica forecasts completed and hashed before labels opened. Aggregate and fixed-year metrics only; no individual images/labels/forecasts/residuals exported.'}
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    save('evaluator_freeze.json',value)
    bundle_files=list(value['source_hashes'])+['external_eval/model_lock.json','external_eval/registered_design.json','external_eval/evaluator_freeze.json']
    bundle={'format':'base64 exact source bytes, preserving line endings','includes_external_arrays':False,
            'files':{name:base64.b64encode(Path(name).read_bytes()).decode('ascii') for name in bundle_files}}
    save('frozen_source_bundle.json',bundle)
    save('evaluator_freeze_identity.json',{'sha256':digest(path),'external_arrays_accessed':False,'models_frozen':2,'replicas':10})
    print(json.dumps({'complete_evaluator_frozen':True,'sha256':digest(path),'external_arrays_accessed':False}),flush=True)

def verify():
    value=json.loads(Path('external_eval/evaluator_freeze.json').read_text())
    assert value['source_hashes']==sources(),'Scientific source or normalization changed after freeze'
    assert value['registered_design_sha256']==digest(Path('external_eval/registered_design.json'))
    check_models()
    return value

if __name__=='__main__':main()
