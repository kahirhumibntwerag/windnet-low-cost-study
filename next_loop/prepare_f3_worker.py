"""Prepare private worker authentication without changing scientific choices."""
import base64,json,secrets
from pathlib import Path
root=Path(__file__).resolve().parents[1]
token=secrets.token_urlsafe(32)
body={'name':'windnet-F3-locked-historical',
      'image':'runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853',
      'gpu':{'id':'NVIDIA A40','count':1,'minCudaVersion':'12.8','minRamPerGpu':32},
      'cloud':'SECURE','dataCenterIds':['CA-MTL-1'],'disk':30,'ports':['8888/http'],
      'mounts':{'network':[{'volumeId':'rrxqngwa8s','path':'/workspace'}]},
      'env':{'EVENT_TOKEN':token,'F3_BOOTSTRAP':base64.b64encode((root/'next_loop/f3_transport.py').read_bytes()).decode()},
      'entrypoint':['python','-u','-c'],
      'cmd':['import os,base64;exec(compile(base64.b64decode(os.environ["F3_BOOTSTRAP"]),"f3_transport.py","exec"))']}
(root/'.secrets/f3_pod.json').write_text(json.dumps({'token':token}))
(root/'.secrets/f3_request.json').write_text(json.dumps(body))
print('Prepared private F3 request and authentication; no data uploaded')
