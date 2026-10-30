"""Prepare M12 without changing the registered M11/M11R sources or credentials."""
import ast,base64,hashlib,json,secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'next_loop'
def extract(path,names):
    s=path.read_text();tree=ast.parse(s)
    return '\n\n'.join(ast.get_source_segment(s,x) for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names)
stats='import numpy as np\nfrom matched.runner import sha\nfrom matched.hse_original import BatchwiseHSE\n\n'
stats+=extract(ROOT/'matched/science.py',['metrics','rowhash','block_moments','moment_score'])+'\n\n'
stats+=extract(ROOT/'event_loop/science.py',['aggregate','gate'])+'\n\n'
stats+=extract(ROOT/'event_loop/precision.py',['stability'])+'\n\n'
stats+=extract(ROOT/'event_loop/mixtures.py',['bootstrap'])+'\n'
stats+='''
# Undefined metrics fail adoption, rather than crashing an ablation export.
_original_aggregate=aggregate
_original_gate=gate
_original_stability=stability
def aggregate(ms):
    if all(all(x.get(k) is not None for k in ['rmse','r','chi2','hse_ts']) for x in ms):return _original_aggregate(ms)
    result={'n':sum(x['n'] for x in ms)}
    for key in ['rmse','r','chi2','hse_ts','mae']:
        v=[x.get(key) for x in ms]
        result[key]=None if any(x is None for x in v) else float(np.tanh(np.mean(np.arctanh(np.clip(v,-.999999,.999999))))) if key=='r' else float(np.mean(v))
    return result
def gate(m,controls):
    if any(x.get(k) is None for x in [m,*controls] for k in ['rmse','r','chi2','hse_ts']):return {'rmse':False,'r':False,'chi2':False,'hse_ts':False,'all_four':False,'undefined':True}
    return _original_gate(m,controls)
def stability(row,controls):
    if any(x.get(k) is None for r in [row,*controls] for x in r['fold_metrics'] for k in ['rmse','r','chi2','hse_ts']):return {'pre_bootstrap_passed':False,'undefined':True,'directional_passes':0}
    return _original_stability(row,controls)
'''
(P/'regional_stats.py').write_text(stats)
d=json.loads((ROOT/'results/next_loop/m11r_results.json').read_text())
(P/'m12_reference.json').write_text(json.dumps({k:d[k] for k in ['controls','row_audits']},indent=2))
protocol={'stage':'M12','experiment_id':'97b0bd0d-8bc1-46f7-882c-7d9859ae4bcd','hypothesis_id':'9e3aee34-65cc-497f-ab21-303e91f9ed0b','campaign_id':'1b7e08e2-70f1-4945-b613-5e785d3f76bb','scope':'Training-side Folds1-4 only; no original final evaluation','grids':[[1,1],[4,3],[6,6]],'views':['flatten','trend'],'encodings':['linear','linear_plus_elementwise_squares'],'precision_lambda':[0,4],'ridge_alpha':[10,100,1000],'new_head_share':[1,.125,.25,.5],'heads':72,'conditions':288,'disk_center':[127.5,127.5],'disk_radius_pixels':112.5,'relative_dark_thresholds':[.3,.5,.7],'per_cell_features':'fractions below each threshold plus mean normalized intensity clipped[0,3]','global_features':'log1p(nonnegative disk median), diskIQR/max(diskmedian,1e-6)','fixed_calibration':{'gain':1.2,'bias':-5,'anchor':'fitting-target mean'},'controls':'Frozen M11R port/M6/M10 training-bank forecasts. Recompute metrics from original reacquired labels and require agreement1e-9 and exact original row hashes. No cross-hardware LSTM refitting.','success':'Same registered all-four margins against all3controls, >=3/4 directional leave-one-fold-out checks, foldRMSE<=1.05*stronger control, positive HSE pairedblockbootstrap95 lower bound against all3controls; candidates checked ascendingRMSE;2000draws seed20261029.','budget':{'prior_cumulative_estimate_usd':1.3999938388888888,'cumulative_limit_usd':5,'max_worker_seconds':3600},'hypothesis_literature':['https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/','https://doi.org/10.1029/2024SW004125','https://doi.org/10.1029/2025EA004523'],'holdout_boundary':'Original Fold5 hash/discard without parsing or retaining; no historical final outputs supplied to worker','frozen_sources':json.loads((P/'protocol.json').read_text())['frozen_sources']}
(P/'m12_protocol.json').write_text(json.dumps(protocol,indent=2))
transport=(P/'transport.py').read_text().replace('/workspace/next-windnet','/workspace/regional-windnet').replace("{'stage':'m11'}","{'stage':'m12'}").replace('Only preregistered M11','Only preregistered M12').replace("'next_loop.worker'","'next_loop.m12_entry'").replace('M11 worker','M12 worker')
(P/'m12_transport.py').write_text(transport)
token=secrets.token_urlsafe(32)
request={'name':'windnet-M12-training-only-cpu','image':'python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159','cloud':'SECURE','cpu':{'id':'cpu3c','vcpuCount':2},'disk':10,'ports':['8888/http'],'dataCenterIds':['EU-RO-1'],'env':{'EVENT_TOKEN':token,'M12_BOOTSTRAP':base64.b64encode(transport.encode()).decode()},'entrypoint':['python','-u','-c'],'cmd':['import os,base64;exec(compile(base64.b64decode(os.environ["M12_BOOTSTRAP"]),"m12_transport.py","exec"))']}
(ROOT/'.secrets/m12_request.json').write_text(json.dumps(request));(ROOT/'.secrets/m12_pod.json').write_text(json.dumps({'token':token}))
print('Prepared M12 protocol, exact control reference, CPU metric functions and private request')
