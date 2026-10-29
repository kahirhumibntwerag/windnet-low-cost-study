"""Build a reproducible static dashboard from aggregate-only campaign exports."""
import datetime as dt,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/matched';D=ROOT/'matched_dashboard/dist'
def main():
    D.mkdir(parents=True,exist_ok=True)
    files={p.stem:json.loads(p.read_text()) for p in R.glob('*.json') if p.name not in ['source_manifest.json']}
    ledger=files.pop('ledger',{'experiments':[]})
    event_dir=ROOT/'results/event_loop';event_protocol=None
    if (ROOT/'event_loop/protocol.json').exists() and (event_dir/'ledger.json').exists():
        event_protocol=json.loads((ROOT/'event_loop/protocol.json').read_text())
        ledger['experiments'].extend(json.loads((event_dir/'ledger.json').read_text())['experiments'])
        for p in event_dir.glob('*.json'):
            if p.name!='ledger.json':files[p.stem]=json.loads(p.read_text())
    titles={'M1':'Original image embeddings and regularized temporal heads','M2':'Nonlinear kernels on the same frozen image embeddings','M3':'Matched loss ablations for sequence and variance terms','M4':'Coronal-hole spatial summaries and feature ablations','M5':'Mean-normalized fast-wind weighting with an all-metric guard','M6':'Amplitude calibration and validation stability audit','F1':'Frozen final historical comparison'}
    for entry in ledger['experiments']:
        entry.setdefault('title',titles.get(entry['label'],entry['label']))
    data={'updated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'campaign_url':'https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa','source_url':'https://github.com/kahirhumibntwerag/windnet-low-cost-study/tree/codex/windnet-matched-loop','protocol':json.loads((ROOT/'matched/protocol.json').read_text()),'ledger':ledger,'results':files}
    if event_protocol:
        data['event_phase']=True
        data['protocol']={**data['protocol'],'inputs':event_protocol['input_information'],'development':event_protocol['validation']+' '+event_protocol['hypothesis_boundary'],'success':event_protocol['adoption'],'final':event_protocol['final'],'stopping':event_protocol['budget']}
    (D/'data.json').write_text(json.dumps(data,indent=2))
    for p in R.glob('*.json'):
        (D/p.name).write_bytes(p.read_bytes())
    if event_protocol:
        for p in event_dir.glob('*.json'):
            if p.name!='ledger.json':(D/p.name).write_bytes(p.read_bytes())
        (D/'event_protocol.json').write_text(json.dumps(event_protocol,indent=2),encoding='utf8')
    for p in [*R.glob('*.csv'),R/'README.md']:
        if p.exists():(D/p.name).write_bytes(p.read_bytes())
    (D/'protocol.json').write_text(json.dumps(data['protocol'],indent=2),encoding='utf8')
    print(json.dumps({'dashboard_files':len(list(D.iterdir())),'aggregate_result_files':len(files),'raw_test_data_included':False}))
if __name__=='__main__':main()
