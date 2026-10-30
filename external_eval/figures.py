"""Standard plotting of aggregate exports only; no individual test values read."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'external_eval';OUT=ROOT/'artifacts/external_eval'
KEYS=['mean_rmse','fisher_r','mean_chi2','mean_hse_ts']
LABELS=['RMSE (km/s)','Fisher-z mean correlation','Original mean reduced χ²','Original mean HSE threat score']
MODELS={'original_windnet':('Original WindNet','#6d7665'),'M13':('M13','#265e91')}

def main():
    d=json.loads((E/'external_results.json').read_text(encoding='utf8'))
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    years=np.arange(2020,2026)
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.4),layout='constrained')
    for ax,key,label in zip(axes.ravel(),KEYS,LABELS):
        for model,(name,color) in MODELS.items():
            values=[d['yearly'][str(y)][model][key] if d['yearly'][str(y)][model] else None for y in years]
            ax.plot(years,[np.nan if v is None else v for v in values],marker='o',ms=4,lw=1.8,color=color,label=name)
        ax.set(title=label,xlabel='Target calendar year (UTC)',ylabel=label,xticks=years)
        if key=='fisher_r':ax.set_ylim(-1,1)
        else:ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.22)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Frozen WindNet–M13 external comparison, 2020–2025',fontsize=14)
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fig.text(.02,.025,'Source: sealed Runpod evaluation; same eligible days for both methods. Five replica scores averaged; correlation uses Fisher-z.\nAnnual HSE retains partial event blocks at year boundaries. No test-period tuning. Individual images, labels and forecasts are excluded.',fontsize=8)
    for ext in ['svg','png']:fig.savefig(OUT/f'metrics_by_year.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.4,6.5),layout='constrained')
    for ax,key,label in zip(axes.ravel(),KEYS,LABELS):
        a=d['overall']['M13'][key];b=d['overall']['original_windnet'][key]
        ci=d['paired_uncertainty']['metrics'][key]['M13_minus_original_delta_95']
        ax.axvline(0,color='#888888',lw=1,linestyle='--')
        if a is not None and b is not None and ci is not None:
            point=a-b;ax.hlines(0,ci[0],ci[1],lw=2.5,color='#265e91');ax.plot(point,0,'o',color='#265e91',ms=7)
            ax.text(.03,.12,f'Δ = {point:.4f}; 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]',transform=ax.transAxes,fontsize=9)
        else:ax.text(.5,.5,'Undefined',transform=ax.transAxes,ha='center')
        ax.set(title=label,xlabel='M13 − original WindNet'+(' (km/s)' if key=='mean_rmse' else ''),yticks=[],ylim=(-.6,.6))
        ax.text(.03,.85,'Lower favors M13' if key in ['mean_rmse','mean_chi2'] else 'Higher favors M13',transform=ax.transAxes,fontsize=9)
        ax.grid(axis='x',alpha=.18)
    fig.suptitle('Paired differences on common 2020–2025 target days',fontsize=14)
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fig.text(.02,.025,'Source: sealed aggregate export. Conditional 95% intervals: 2,000 paired calendar28-day bootstrap draws within block-start year, seed20261030.\nEvery replica and method shares each resample; repeated dates are not treated as independent observations. Point-score wins are separate from interval inference.',fontsize=8)
    for ext in ['svg','png']:fig.savefig(OUT/f'paired_differences.{ext}',dpi=180)
    plt.close(fig)
    with (E/'aggregate_comparison.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['period','model',*KEYS,'unique_target_days','positive_sigma_days'])
        for period,group in [('2020-2025',d['overall']),*d['yearly'].items()]:
            for model,row in group.items():
                if row:w.writerow([period,model,*[row[k] for k in KEYS],row['unique_target_days'],row['positive_sigma_days']])
    files=[{'file':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.glob('*')) if p.suffix in ['.svg','.png']]
    (E/'figures_manifest.json').write_bytes((json.dumps({'figures':files,'source_results_sha256':hashlib.sha256((E/'external_results.json').read_bytes()).hexdigest(),'individual_test_data_read':False},indent=2)+'\n').encode('utf8'))
    print(json.dumps({'figures':len(files),'csv':'external_eval/aggregate_comparison.csv','source':'aggregate results only'}))

if __name__=='__main__':main()
