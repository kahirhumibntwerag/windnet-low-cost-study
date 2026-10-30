"""Local authenticated worker client; credentials stay in ignored .secrets."""
import argparse,hashlib,json
from pathlib import Path
import requests

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['health','state','upload','run','artifact']);ap.add_argument('value',nargs='?');a=ap.parse_args()
    config=json.loads(Path('.secrets/external_pod.json').read_text(encoding='utf8'))
    url='https://'+config['pod_id']+'-8888.proxy.runpod.net'
    session=requests.Session();session.headers['Authorization']='Bearer '+config['token']
    if a.action in ['health','state']:
        r=session.get(url+'/'+a.action,timeout=40);r.raise_for_status();print(json.dumps(r.json()));return
    if a.action=='run':
        r=session.post(url+'/run',json={'stage':a.value},timeout=40);r.raise_for_status();print(json.dumps(r.json()));return
    if a.action=='artifact':
        assert a.value.endswith(('.json','.txt','.csv')) and '/' not in a.value and '\\' not in a.value
        r=session.get(url+'/artifact/'+a.value,timeout=60);r.raise_for_status()
        target=Path('external_eval')/a.value;target.write_bytes(r.content)
        print(json.dumps({'file':str(target),'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}));return
    names=['matched/'+n for n in ['__init__.py','runner.py','science.py','hse_original.py','final_study.py','source_manifest.json','reference_sources.json','protocol.json']]
    names+=['event_loop/'+n for n in ['__init__.py','runner.py','science.py','location.py','final.py','mixtures.py']]
    names+=['next_loop/'+n for n in ['__init__.py','regional_features.py','target_weights.py','f3_panel.py']]
    names+=[str(p).replace('\\','/') for p in Path('external_eval').glob('*.py') if p.name not in ['transport.py','client.py','prepare_request.py']]
    names+=['external_eval/'+n for n in ['model_lock.json','author_checkpoint_hashes.json','registry.json','registered_design.json']]
    names+=['external_eval/calibration/aia_V8_20171210_050627_response_table.txt']
    records=[]
    for name in names:
        raw=Path(name).read_bytes();r=session.post(url+'/upload/'+name,data=raw,timeout=40);r.raise_for_status()
        digest=hashlib.sha256(raw).hexdigest();assert r.json()['sha256']==digest
        records.append({'file':name,'sha256':digest})
    Path('external_eval/worker_source_upload.json').write_text(json.dumps({'files':records},indent=2)+'\n',encoding='utf8')
    print(json.dumps({'uploaded_files':len(records),'all_hash_verified':True}))

if __name__=='__main__':main()
