"""Build a reproducible static dashboard from aggregate-only campaign exports."""
import datetime as dt,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/matched';D=ROOT/'matched_dashboard/dist'
def main():
    D.mkdir(parents=True,exist_ok=True)
    files={p.stem:json.loads(p.read_text()) for p in R.glob('*.json') if p.name not in ['source_manifest.json']}
    ledger=files.pop('ledger',{'experiments':[]})
    data={'updated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'campaign_url':'https://fixrelay.vercel.app/campaigns/e324d24c-b179-427c-87c1-365b857658aa','source_url':'https://github.com/kahirhumibntwerag/windnet-low-cost-study/tree/codex/windnet-matched-loop','protocol':json.loads((ROOT/'matched/protocol.json').read_text()),'ledger':ledger,'results':files}
    (D/'data.json').write_text(json.dumps(data,indent=2))
    for p in R.glob('*.json'):
        (D/p.name).write_bytes(p.read_bytes())
    (D/'protocol.json').write_bytes((ROOT/'matched/protocol.json').read_bytes())
    print(json.dumps({'dashboard_files':len(list(D.iterdir())),'aggregate_result_files':len(files),'raw_test_data_included':False}))
if __name__=='__main__':main()
