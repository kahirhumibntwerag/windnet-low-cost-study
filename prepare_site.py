"""Build a credential-free static dashboard snapshot from verified artifacts."""
import csv,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WEB=ROOT/'web';ASSETS=WEB/'assets';ASSETS.mkdir(parents=True,exist_ok=True)
summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
summary['loops']={name:json.loads((ROOT/f'results/{name}.json').read_text(encoding='utf-8')) for name in ['loop1','loop2','loop3','loop4']}
(WEB/'data.json').write_text(json.dumps(summary,separators=(',',':')),encoding='utf-8')
with (ROOT/'results/matched_predictions.csv').open(newline='') as f:
    predictions=list(csv.DictReader(f))
for r in predictions:
    for k in r:
        if k!='date':r[k]=int(r[k]) if k in ['row','fold','block','offset'] else float(r[k])
(WEB/'predictions.json').write_text(json.dumps(predictions,separators=(',',':')),encoding='utf-8')
for name in ['comparison.csv','ablations.json','experiment_ledger.json','matched_predictions.csv']:
    shutil.copy2(ROOT/'results'/name,ASSETS/name)
for path in (ROOT/'output/figures').glob('*.png'):shutil.copy2(path,ASSETS/path.name)
paper=ROOT/'output/pdf/windnet-study.pdf'
if paper.exists():shutil.copy2(paper,ASSETS/'windnet-study.pdf')
dist=ROOT/'dashboard/dist';dist.mkdir(parents=True,exist_ok=True)
shutil.copytree(WEB,dist,dirs_exist_ok=True)
print(json.dumps({'web':str(WEB),'dist':str(dist),'experiments':len(summary['ledger']['experiments']),'prediction_rows':len(predictions),'pdf_included':paper.exists()}))
