"""Publication figures from continuation development results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures'
OUT.mkdir(parents=True, exist_ok=True)
r8 = json.loads((ROOT/'results/continuation/h8_results.json').read_text())
r9 = json.loads((ROOT/'results/continuation/h9_results.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})

names = ['Wind history', '+ ESP bands', '+ EUV lines', '+ All EVE', 'Lines + fast-case weighting']
ids = [r9['feature_ablations_unweighted'][g]['candidate'] for g in
       ['wind_only', 'wind_esp', 'wind_lines', 'wind_all_eve']] + [r9['selected']]
metrics = [r9['candidate_reports'][n]['metrics'] for n in ids]
colors = ['#a2afba', '#80aeb3', '#418e99', '#2b657c', '#d2954b']
fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), sharey=True)
for ax, key, title in zip(axes, ['overall', 'ge_500', 'ge_600'],
                         ['All wind speeds', 'Observed ≥500 km/s', 'Observed ≥600 km/s']):
    values = [m['overall']['rmse'] if key == 'overall' else
              m['fast_wind'][key]['continuous_observed_fast']['rmse'] for m in metrics]
    ax.barh(np.arange(5), values, color=colors, height=.66)
    ax.set_yticks(np.arange(5), names)
    ax.invert_yaxis()
    ax.set_title(title, pad=14)
    ax.set_xlabel('RMSE (km/s); lower is better')
    ax.set_xlim(0, max(values)*1.17)
    ax.xaxis.grid(True, color='#dce2e5', lw=.6)
    ax.set_axisbelow(True)
    for row, value in enumerate(values):
        ax.text(value+max(values)*.025, row, f'{value:.1f}', va='center', fontsize=10)
fig.suptitle('SDO feature ablations and fast-wind weighting', x=.03, ha='left', fontsize=16, weight='bold')
fig.text(.03, .91, 'Training-side CV only · same 25% ridge / 75% tree pair in every row · no SDO images', color='#516472')
fig.text(.03, .04, 'First four rows use unweighted fitting. Last row blends 25% of a fast-case-weighted fit with 75% unweighted.\n'
         '1,512 validation windows; 291 observed ≥500 and 82 observed ≥600. Selected CV scores are optimistic development estimates.', fontsize=9, color='#516472')
fig.subplots_adjust(left=.235, right=.98, top=.80, bottom=.23, wspace=.22)
for ext in ['png', 'svg']:
    fig.savefig(OUT/f'continuation_ablations.{ext}', dpi=200, facecolor='white')
plt.close(fig)

reports = [r8['candidate_reports'][r8['incumbent']]['metrics'],
           r8['candidate_reports'][r8['selected']]['metrics'],
           r9['candidate_reports'][r9['selected']]['metrics']]
labels = ['Starting wind-only model', 'SDO trial (H8)', 'Weighted EUV-line model (H9)']
fig, ax = plt.subplots(figsize=(9, 4.7))
x = np.arange(3)
for j, (key, label, color, count) in enumerate([
    ('all', 'All speeds (n=1,128)', '#a2afba', 1128),
    ('ge_500', 'Observed ≥500 (n=218)', '#418e99', 218),
    ('ge_600', 'Observed ≥600 (n=60)', '#d2954b', 60)]):
    values = [m['persistence_skill']['persistence_27day'][key]['mse_skill_score'] for m in reports]
    ax.bar(x+(j-1)*.24, values, width=.23, label=label, color=color)
    for location, value in zip(x+(j-1)*.24, values):
        ax.text(location, value+(0.009 if value >= 0 else -.012), f'{value:.3f}', ha='center',
                va='bottom' if value >= 0 else 'top', fontsize=9)
ax.axhline(0, color='#344956', linewidth=.8)
ax.set_xticks(x, labels)
ax.set_ylabel('MSE skill against 27-day persistence')
ax.set_ylim(-.15, .39)
ax.yaxis.grid(True, color='#dce2e5', lw=.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=9)
fig.suptitle('Fast-wind skill improves while overall skill falls', x=.09, ha='left', fontsize=15, weight='bold')
fig.text(.09, .90, 'Training CV only · positive skill is better · model and persistence use identical cases', color='#516472')
fig.text(.09, .03, '27-day reference is available for 1,128 of 1,512 windows; excluded sources remain masked.\n'
         'Better conditional error does not imply good event detection: H9 still misses every observed ≥600 event.', fontsize=9, color='#516472')
fig.subplots_adjust(top=.73, bottom=.20, left=.09, right=.98)
for ext in ['png', 'svg']:
    fig.savefig(OUT/f'continuation_skill.{ext}', dpi=200, facecolor='white')
plt.close(fig)
print('Saved four training-only figure files.')
