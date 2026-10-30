"""Private Runpod request; token never printed or committed."""
import base64,json,secrets
from pathlib import Path

root=Path(__file__).resolve().parents[1]
token=secrets.token_urlsafe(32)
request={'name':'windnet-external2020-2025-frozen-comparison',
         'image':'runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853',
         'cloud':'SECURE','cpu':{'id':'cpu3c','vcpuCount':8},'dataCenterIds':['EU-RO-1'],'disk':40,'ports':['8888/http'],
         'env':{'EXTERNAL_TOKEN':token,'EXTERNAL_BOOTSTRAP':base64.b64encode((root/'external_eval/transport.py').read_bytes()).decode()},
         'entrypoint':['python','-u','-c'],'cmd':['import os,base64;exec(compile(base64.b64decode(os.environ["EXTERNAL_BOOTSTRAP"]),"transport.py","exec"))']}
(root/'.secrets/external_request.json').write_text(json.dumps(request),encoding='utf8')
(root/'.secrets/external_pod.json').write_text(json.dumps({'token':token}),encoding='utf8')
print('Prepared private external comparison worker request')
