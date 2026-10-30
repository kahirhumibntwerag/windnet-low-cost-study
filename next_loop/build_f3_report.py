"""Human-readable fixed-panel report, generated only from completed aggregate exports."""
import csv,json
from pathlib import Path
from next_loop.f3_dashboard import LABELS,METRICS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/next_loop'
def fmt(v):return 'undefined' if v is None else f'{v:.6f}'

def main():
    r=json.loads((OUT/'f3_results.json').read_text());v=json.loads((OUT/'f3_verification.json').read_text())
    assert v['passed'] and r['final_evaluation_passes']==1 and r['post_test_model_changes']==0
    gates=r['gates'];text='# F3: fixed regional-tail model on the original WindNet folds\n\n'
    for key,label in [('released_windnet','released original WindNet'),('prior_M10','same-runtime retrained M10')]:
        g=gates[key];text+=f'The frozen primary **'+('passed' if g['all_four'] else 'failed')+f'** the strict all-four numerical gate against {label}. '
        text+='Individual metric directions: '+', '.join(k+(' pass' if passed else ' fail') for k,passed in g['checks'].items())+'.\n\n'
    text+='All 157 original blocks and 2,198 evaluation windows are included. Every trainable model fitted the same other four original folds and received the same four 211 Å images. No candidate was substituted after testing. Hypotheses and the primary recipe were chosen from training-side validation.\n\n'
    text+='## Complete fixed panel\n\n| Model or ablation | RMSE (km/s) ↓ | Fisher r ↑ | Reduced χ² ↓ | Original HSE TS ↑ |\n| --- | ---: | ---: | ---: | ---: |\n'
    for row in r['rows']:text+='| '+LABELS[row['model']]+' | '+' | '.join(fmt(row['summary'][k]) for k,title in METRICS)+' |\n'
    text+='\nThe regional-only, unit-target-weight, zero-regional-precision and global 1×1 entries change only the stated component of the primary. The no-outer-calibration ablation removes gains and biases from the regional head and M10 outer mixture while retaining M6’s inner gain. Full recipes are fixed in [the freeze](f3_freeze.json).\n\n'
    text+='## Conditional uncertainty\n\nPrimary minus comparator, paired whole-block bootstrap with 2,000 draws within the original folds, seed 20261029. Negative differences favor the primary for RMSE and χ²; positive differences favor it for correlation and event score.\n\n| Comparator | RMSE difference, 95% interval | Fisher r difference | χ² difference | HSE TS difference |\n| --- | ---: | ---: | ---: | ---: |\n'
    for ref,ci in r['primary_uncertainty'].items():
        cells=[]
        for key in ['rmse','r','chi2','hse_ts']:
            bounds=ci['metrics'][key]['delta_95'];cells.append('undefined' if bounds is None else f'[{bounds[0]:+.6f}, {bounds[1]:+.6f}]')
        text+='| '+LABELS[ref]+' | '+' | '.join(cells)+' |\n'
    text+='\nThese are conditional intervals, not adjusted for adaptive selection or repeated historical evaluation. A numerical improvement does not establish that every metric’s interval excludes zero. Original blocks may overlap.\n\n'
    text+='## Provenance and boundaries\n\n'
    text+='- The complete source/data/panel freeze was created and pushed before acquiring Fold5. Its SHA256 is `'+r['freeze_sha256']+'`.\n'
    text+='- All five folds and all trainable panel entries were fitted before exporting any metric. The predictor interface received image-derived features only.\n'
    text+='- The final runtime was Python 3.11.11, NumPy 2.1.2, SciPy 1.16.1, scikit-learn 1.7.2 and PyTorch 2.8.0.dev20250319+cu128 on an NVIDIA A40. All trainable comparators shared it.\n'
    text+='- The released TensorFlow forecasts are a separate historical comparator; the matched neural port has documented implementation differences. Exact cross-hardware numerical reproduction of archived M10 is not claimed.\n'
    text+='- The original nominal delay and HSE implementation conventions are preserved. Image-plane intensity proxies are not physical source-region attribution.\n'
    text+='- Folds 1–4 were adaptively reused for development, and earlier historical aggregate results were already known. This locked exploratory benchmark is not fresh blind confirmation or a prospective chronological test.\n'
    text+='- Historical outcomes cannot choose a replacement or the next hypothesis. Any further model idea must come from training-side evidence and a separately registered protocol.\n\n'
    text+=f'The independent export verifier checked aggregate arithmetic, both gates, original fold counts, {v["source_files_verified"]} source identities and {v["checkpoints_verified"]} checkpoint-container hashes. It did not independently refit models or rerun metrics from raw labels. Raw holdout observations and individual historical predictions are excluded from exported checkpoints.\n\n'
    text+='[Full results and fold metrics](f3_results.json) · [Row and label-boundary audit](f3_audit.json) · [Verification](f3_verification.json) · [Artifact digest](f3_artifacts.json) · [Reproduction guide](../../next_loop/README.md)\n'
    (OUT/'F3_RESULTS.md').write_text(text,encoding='utf8')
    with (OUT/'f3_comparison.csv').open('w',newline='',encoding='utf8') as out:
        w=csv.writer(out);w.writerow(['model',*[k for k,title in METRICS]])
        for row in r['rows']:w.writerow([row['model'],*[row['summary'][k] for k,title in METRICS]])
    print('Generated F3_RESULTS.md and the complete fixed-panel CSV')

if __name__=='__main__':main()
