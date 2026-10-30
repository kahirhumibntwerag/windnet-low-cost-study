"""M12 transport; upload only code and the declared training-side reference."""
import argparse,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('op',choices=['health','state','upload','run','download']);ap.add_argument('names',nargs='*');a=ap.parse_args()
    c=json.loads((ROOT/'.secrets/m12_pod.json').read_text());h={'Authorization':'Bearer '+c['token']}
    if a.op in ['health','state']:
        r=requests.get(c['url']+'/'+a.op,headers=h,timeout=45);r.raise_for_status();d=r.json()
        if 'log' in d:d['log']=d['log'][-1800:]
        print(json.dumps(d))
    elif a.op=='upload':
        r=requests.get(c['url']+'/state',headers=h,timeout=45);r.raise_for_status();assert not r.json()['running']
        paths=[p for pkg in ['matched','event_loop'] for p in (ROOT/pkg).glob('*.py')]
        paths+=[ROOT/'matched/source_manifest.json']
        paths+=[ROOT/'next_loop'/n for n in ['__init__.py','m12.py','m12_entry.py','m12_protocol.json','m12_reference.json','regional_features.py','regional_stats.py']]
        paths+=[ROOT/'next_loop'/n for n in ['m13.py','m13_protocol.json','target_weights.py'] if (ROOT/'next_loop'/n).exists()]
        for p in paths:
            r=requests.post(c['url']+'/upload/'+p.parent.name+'/'+p.name,headers=h,data=p.read_bytes(),timeout=90);r.raise_for_status()
        print('Uploaded declared M12 code, frozen sources and exact training-only control reference')
    elif a.op=='run':
        r=requests.post(c['url']+'/run',headers=h,json={'stage':'m12'},timeout=45);r.raise_for_status();print(r.text)
    else:
        for name in a.names:
            local_name='m12_acquisition.json' if name=='event_acquisition.json' else name
            dest=ROOT/('artifacts/next_loop' if name.endswith('.gz') else 'results/next_loop')/local_name;dest.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(c['url']+'/file/'+name,headers=h,timeout=180,stream=True) as r:
                r.raise_for_status()
                with dest.open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
            print(name,dest.stat().st_size)
if __name__=='__main__':main()
