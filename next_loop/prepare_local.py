"""Prepare a secret worker request and minimal training-only reference."""
import base64, hashlib, json, secrets
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'next_loop'
result=json.loads((ROOT/'results/event_loop/m10_results.json').read_text())
reference={k:result[k] for k in ['selected','controls','row_audits']}
(P/'training_reference.json').write_text(json.dumps(reference,indent=2))
registry={'campaign_id':'1b7e08e2-70f1-4945-b613-5e785d3f76bb','hypothesis_id':'db413194-1606-4095-87d7-23b608d5e626','experiment_id':'571f6a04-4649-4e79-a83a-edb9ad102840'}
(P/'registry.json').write_text(json.dumps(registry,indent=2))
protocol={'stage':'M11','title':'Cross-fitted empirical distribution correction','scope':'Original Folds1-4 training-bank validation only','input_information':'Original 211 Angstrom four images, H4D3, unchanged forecasts and fitting rows','hypothesis_sources':['https://doi.org/10.1029/2024SW004125','https://arxiv.org/abs/2609.26683'],'last_training_observation':'M10 qualified on training-bank metrics: RMSE78.00280, Fisher r0.573537, chi-square26.56181, HSE0.248904. No final metrics used to choose M11.','candidate_strengths':[0,.25,.5,.75,1],'quantile_knots':[.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975],'ablation':'Full correction map from in-sample fitting predictions; ineligible for adoption','fixed_location':{'representation':'ratio','temporal_view':'flatten','precision_lambda':4,'ridge_alpha':10000,'location_weight':.625,'gain':1.2,'bias':-5},'fit_calibration':'Three leave-fitting-fold-out regressions for each outer split. Scaler/precision/target-anchor inner-fit-only. Encoder mean is outer-fit-only and reused across inner folds; not fully nested preprocessing. Map predictions to fitting target quantiles; linear interpolation, endpoint clipping.','success':'All four margins against fixed M10, port and M6: RMSE<=.99*minimum, r>=maximum+.005, chi2<=.995*minimum, HSE>=maximum+.02; >=3/4 leave-one-fold-out directional passes; every foldRMSE<=1.05*minimum control; positive HSE conditional bootstrap95 lower bound against all three. LowestRMSE eligible cross-fitted strength.','test_access':'No original final evaluation in M11. Fold5 hash/discard without parsing. No final outputs, raw targets, test predictions, or paper supplied to worker.','budget':{'cumulative_limit_usd':5,'prior_matched_estimate_usd':1.2338398583333332,'max_worker_lifetime_seconds':3600},'registry':registry,'frozen_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for pkg in ['matched','event_loop'] for p in (ROOT/pkg).glob('*') if p.suffix=='.py'}}
(P/'protocol.json').write_text(json.dumps(protocol,indent=2))
transport=(ROOT/'event_loop/transport.py').read_text().replace('/workspace/event-windnet','/workspace/next-windnet').replace("['matched','event_loop']", "['matched','event_loop','next_loop']")
start=transport.index("            cfg=json.loads(b)")
end=transport.index("            return self.reply({'started':cfg['stage'],'pid':PROCESS.pid})")
transport=transport[:start]+'''            cfg=json.loads(b)
            if cfg != {'stage':'m11'}:return self.reply({'error':'Only preregistered M11 is permitted'},400)
            with (ROOT/'worker.log').open('w') as log:
                env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
                PROCESS=subprocess.Popen([sys.executable,'-u','-m','next_loop.worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
'''+transport[end:]
transport=transport.replace("print('EVENT_WORKER listening 8888',flush=True)","import threading\nthreading.Timer(3600,lambda: os._exit(0)).start()\nprint('Training-only M11 worker listening 8888; max lifetime3600s',flush=True)")
(P/'transport.py').write_text(transport)
token=secrets.token_urlsafe(32)
request={'name':'windnet-M11-training-only','imageName':'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04','gpuTypeIds':['NVIDIA A40'],'gpuCount':1,'cloudType':'SECURE','containerDiskInGb':30,'volumeInGb':0,'ports':['8888/http'],'env':{'EVENT_TOKEN':token,'M11_BOOTSTRAP':base64.b64encode(transport.encode()).decode()},'dockerEntrypoint':['python','-u','-c'],'dockerStartCmd':['import os,base64;exec(compile(base64.b64decode(os.environ["M11_BOOTSTRAP"]),"m11_transport.py","exec"))']}
(ROOT/'.secrets/next_request.json').write_text(json.dumps(request))
(ROOT/'.secrets/next_pod.json').write_text(json.dumps({'token':token}))
print('Prepared minimal training reference, registered protocol and private pod request; secrets not printed')
