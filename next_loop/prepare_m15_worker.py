"""Prepare a private M15 worker request; no scientific choices are changed."""
import argparse,base64,json,secrets
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--volume',required=True);args=ap.parse_args()
token=secrets.token_urlsafe(32)
request={'name':'windnet-M15-training-gate','image':'python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159',
         'cloud':'SECURE','cpu':{'id':'cpu3c','vcpuCount':2},'dataCenterIds':['EU-RO-1'],'disk':10,'ports':['8888/http'],
         'mounts':{'network':[{'volumeId':args.volume,'path':'/workspace'}]},
         'env':{'EVENT_TOKEN':token,'M15_BOOTSTRAP':base64.b64encode((root/'next_loop/m15_transport.py').read_bytes()).decode()},
         'entrypoint':['python','-u','-c'],'cmd':['import os,base64;exec(compile(base64.b64decode(os.environ["M15_BOOTSTRAP"]),"m15_transport.py","exec"))']}
(root/'.secrets/m15_request.json').write_text(json.dumps(request));(root/'.secrets/m15_pod.json').write_text(json.dumps({'token':token,'volume_id':args.volume}))
print('Prepared private M15 worker request; secrets not printed')
