"""Independently verify metrics, merge the released reference, and build figures."""
import csv,datetime as dt,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'output/figures';FIG.mkdir(parents=True,exist_ok=True)
LABELS={'training_mean':'Training mean','oldest_persistence':'Oldest persistence','latest_persistence':'Latest persistence',
    'linear_svr':'Linear SVM','rbf_svr':'RBF SVM','short_ridge':'Four-speed ridge','recurrence_ridge':'Recurrence ridge',
    'multi_ridge':'Multivariate ridge','recurrence_tree':'Recurrence tree','multi_tree':'Multivariate tree',
    'residual_nn':'Residual neural ensemble','ridge_nn_blend':'Ridge + neural blend','convex_ensemble':'Convex ensemble',
    'selected_model':'Frozen selection rule','released_windnet':'Released WindNet'}
ALABEL={'short_4':'Four recent speeds','recent_only':'Longer recent history only','recent_plus_27':'Recent + 27-day band',
    'recent_plus_54':'Recent + 54-day band','speed_recurrence':'All speed recurrence','full':'Full multivariate ridge',
    'no_27_band':'Remove 27-day band','no_54_band':'Remove 54-day band','no_magnetic':'Remove magnetic measurements','no_plasma':'Remove plasma measurements'}

def readcsv(path):return list(csv.DictReader(path.open(newline='')))
rows=readcsv(OUT/'predictions.csv');ref={int(r['row']):r for r in readcsv(OUT/'reference_predictions.csv')}
for r in rows:
    for k in ['row','fold','block','offset']:r[k]=int(r[k])
    for k in list(r):
        if k not in ['row','fold','block','offset','date']:r[k]=float(r[k])
    assert abs(r['y']-float(ref[r['row']]['y']))<1e-6
    r['released_windnet']=float(ref[r['row']]['released_windnet'])
assert len(rows)==2198 and len({r['row'] for r in rows})==2198
results=json.loads((OUT/'final_results.json').read_text());references=json.loads((OUT/'reference_metrics.json').read_text())
metrics=results['metrics'];metrics['released_windnet']={'folds':references['folds'],'mean_rmse':references['mean_rmse'],
    'rmse_se':references['rmse_se'],'fisher_mean_r':references['fisher_mean_r'],'mean_mae':np.mean([r['mae'] for r in references['folds']])}
for name,values in metrics.items():
    for fold in range(1,6):
        subset=[r for r in rows if r['fold']==fold];rmse=np.sqrt(np.mean([(r[name]-r['y'])**2 for r in subset]))
        assert abs(rmse-values['folds'][fold-1]['rmse'])<1e-8

def cluster_bootstrap(fold,reference):
    sub=[r for r in rows if r['fold']==fold];intervals=[]
    for b in sorted(set(r['block'] for r in sub)):
        days=[np.datetime64(r['date']) for r in sub if r['block']==b]
        intervals.append((min(days)-np.timedelta64(6,'D'),max(days),b))
    clusters=[];ends=[]
    for start,end,b in sorted(intervals):
        if clusters and start<=ends[-1]:clusters[-1].append(b);ends[-1]=max(ends[-1],end)
        else:clusters.append([b]);ends.append(end)
    indices=[np.array([i for i,r in enumerate(sub) if r['block'] in c]) for c in clusters]
    choices=np.random.default_rng(20261029).integers(0,len(indices),size=(5000,len(indices)))
    counts=np.array([len(ix) for ix in indices])[choices].sum(axis=1)
    a=np.array([sum((sub[i]['y']-sub[i]['selected_model'])**2 for i in ix) for ix in indices])
    b=np.array([sum((sub[i]['y']-sub[i][reference])**2 for i in ix) for ix in indices])
    differences=np.sqrt(a[choices].sum(axis=1)/counts)-np.sqrt(b[choices].sum(axis=1)/counts)
    return {'ci95':np.quantile(differences,[.025,.975]).tolist(),'clusters':len(clusters),'resamples':5000,
        'difference_definition':'Frozen selection RMSE minus reference RMSE, km/s'}

uncertainty={name:cluster_bootstrap(5,name) for name in ['released_windnet','short_ridge','recurrence_ridge']}
ab=json.loads((OUT/'ablations.json').read_text())
summary={'date':'2026-10-29','repository':'https://github.com/kahirhumibntwerag/windnet-low-cost-study',
    'dashboard':'https://windnet-research-lab-oct29.wellbat32.chatgpt.site','metrics':metrics,'uncertainty_fold5':uncertainty,
    'ablations':ab,'ledger':json.loads((OUT/'experiment_ledger.json').read_text(encoding='utf-8')),
    'split_audit':json.loads((ROOT/'data/split_audit.json').read_text()),'compute':json.loads((OUT/'compute.json').read_text()),
    'labels':LABELS,'ablation_labels':ALABEL,'reference_alignment':{'rows':2198,'max_error':0.0},
    'caveats':['Five-fold averages are exploratory because Folds 1–4 informed model design.',
        'Original Fold 5 was sealed during the four hypothesis cycles; newly requested ablations are post-evaluation exploratory.',
        'Same target rows, different information: compact models use longer in-situ history and additional measurements; WindNet uses images.',
        'Released WindNet checkpoints were selected using the held-out fold in the original code.',
        'Random block folds do not establish chronological or operational generalization.']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'comparison.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['model','mean_rmse_kms','se_rmse_kms','mean_mae_kms','fisher_mean_r','fold5_rmse_kms','fold5_r'])
    for n,m in metrics.items():w.writerow([LABELS[n],m['mean_rmse'],m['rmse_se'],m['mean_mae'],m['fisher_mean_r'],m['folds'][4]['rmse'],m['folds'][4]['r']])
with (OUT/'matched_predictions.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':200})
fig,ax=plt.subplots(figsize=(7.15,2.75))
for name,color,marker in [('short_ridge','#8f9baa','s'),('selected_model','#245e86','o'),('released_windnet','#bc6942','D')]:
    ax.plot(range(1,6),[m['rmse'] for m in metrics[name]['folds']],marker=marker,color=color,label=LABELS[name],linewidth=1.8,markersize=5)
ax.axvspan(4.75,5.25,color='#edf3f7',zorder=-2);ax.set_xticks(range(1,6));ax.set_xlabel('Original held-out fold (Fold 5 was sealed during design)');ax.set_ylabel('RMSE (km/s)');ax.grid(axis='y',alpha=.2)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncol=3,frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(FIG/'original_folds.png',bbox_inches='tight');plt.close(fig)

names=['no_27_band','no_54_band','no_magnetic','no_plasma'];delta=[ab['variants'][n]['delta_rmse_vs_full'] for n in names]
fig,ax=plt.subplots(figsize=(7.15,2.2));ax.barh(range(4),delta,color=['#245e86','#6291ad','#a9bdc9','#a9bdc9'],height=.55)
ax.set_yticks(range(4),[ALABEL[n] for n in names]);ax.invert_yaxis();ax.set_xlabel('Change in mean RMSE from full ridge (km/s; positive = worse)');ax.set_ylabel('Removed feature group');ax.axvline(0,color='#506070',lw=.7);ax.grid(axis='x',alpha=.15)
for i,value in enumerate(delta):ax.text(max(value,0)+.08,i,f'{value:+.2f}',va='center',fontsize=9)
ax.set_xlim(-.15,5.2);fig.tight_layout();fig.savefig(FIG/'feature_ablations.png',bbox_inches='tight');plt.close(fig)

sub=sorted([r for r in rows if r['fold']==5],key=lambda r:(r['date'],r['row']));unique={}
for r in sub:unique.setdefault(r['date'],r)
first=min(unique);start=np.datetime64(first);days=np.arange(start,start+np.timedelta64(90,'D'))
fig,ax=plt.subplots(figsize=(7.15,2.55))
for name,color,lw in [('y','#1e2833',1.7),('released_windnet','#bc6942',1.3),('selected_model','#245e86',1.5)]:
    vals=[unique.get(str(day),{}).get(name,np.nan) for day in days]
    ax.plot(days,vals,color=color,label={'y':'Observed','released_windnet':'Released WindNet','selected_model':'Frozen compact method'}[name],lw=lw)
ax.set_ylabel('Wind speed (km/s)');ax.set_xlabel('Date (first 90 calendar days containing Fold 5 targets; gaps preserved)');ax.grid(axis='y',alpha=.15)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=3,frameon=False,fontsize=9);fig.autofmt_xdate(rotation=0);fig.tight_layout();fig.savefig(FIG/'fold5_forecasts.png',bbox_inches='tight');plt.close(fig)

print(json.dumps({'all_2198_metric_rows_verified':True,'fold5_uncertainty':uncertainty,'figures':str(FIG),'mean_rmse_reduction_pct':100*(1-metrics['selected_model']['mean_rmse']/metrics['released_windnet']['mean_rmse'])},indent=2))
