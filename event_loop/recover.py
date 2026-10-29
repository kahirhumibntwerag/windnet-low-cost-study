"""Recover only the preregistered training-side run after temporary disk loss."""
import json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]

def main():
    c=json.loads((ROOT/'.secrets/event_pod.json').read_text());headers={'Authorization':'Bearer '+c['token']};last=''
    stages=[{'stage':'acquire'},{'stage':'prepare'},{'stage':'baseline'},{'stage':'gradient','mode':'location_prepare'},{'stage':'gradient','mode':'precision'}]
    for cfg in stages:
        r=requests.post(c['url']+'/run',headers=headers,json=cfg,timeout=45);r.raise_for_status();print(json.dumps({'recovery_stage_started':cfg}),flush=True)
        while True:
            time.sleep(5);r=requests.get(c['url']+'/state',headers=headers,timeout=45);r.raise_for_status();state=r.json();line=state['log'].splitlines()[-1] if state['log'].splitlines() else ''
            if line!=last:print(line[-1000:],flush=True);last=line
            if not state['running']:
                assert state['returncode']==0,state['log'];break
    exports={'event_acquisition.json':'m10_recovery_acquisition.json','event_embedding_audit.json':'m10_recovery_embedding_audit.json','event_controls.json':'event_controls_rerun.json','m9_embedding_audit.json':'m10_recovery_location_audit.json','m10_results.json':'m10_results.json'}
    out=ROOT/'results/event_loop';out.mkdir(exist_ok=True)
    for source,dest in exports.items():
        r=requests.get(c['url']+'/file/'+source,headers=headers,timeout=180);r.raise_for_status();(out/dest).write_bytes(r.content);print('Recovered '+dest,flush=True)
    print('RECOVERY_COMPLETE: training-only M10 result saved; final data remains unavailable',flush=True)
if __name__=='__main__':main()
