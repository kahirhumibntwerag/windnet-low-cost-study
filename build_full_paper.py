"""Build a readable eight-page paper from completed, immutable E13 artifacts.

The parent ran the PDF operation marker before creating this source. This
script does not load source observations, fit models, or select a test winner.
Paper-sized figures only re-layout the fixed reporting panel for legibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)


ROOT = Path(__file__).resolve().parent
LOCKED = ROOT / 'results/locked_final'
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/full-paper'
FIGOUT = ROOT / 'output/figures'
REPO = 'https://github.com/kahirhumibntwerag/windnet-low-cost-study'
BRANCH = REPO + '/tree/codex/windnet-study'
CAMPAIGN = 'https://fixrelay.vercel.app/campaigns/bb47504f-7bd4-45bf-954b-b0b077ac5aee'
WIDTH = A4[0] - 88
NAVY = colors.HexColor('#15394d')
BLUE = colors.HexColor('#2b6d86')
RUST = colors.HexColor('#b46e43')
INK = colors.HexColor('#23333e')
MUTED = colors.HexColor('#536773')
LINE = colors.HexColor('#d8e1e5')
PALE = colors.HexColor('#edf4f6')
ROLE_IDS = ['released_windnet', 'old_frozen_selected', 'P0', 'starting_wind', 'h9_fast_specialist']
SHORT = {'released_windnet': 'WindNet', 'old_frozen_selected': 'Earlier winner',
         'P0': 'Prior neural recipe', 'starting_wind': 'Wind primary',
         'h9_fast_specialist': 'Fast primary', 'P1': '+ ESP bands',
         'P2': '+ EUV lines', 'P3': '+ Both blocks'}
LABEL = {'released_windnet': 'Released WindNet', 'old_frozen_selected': 'Earlier frozen winner',
         'P0': 'P0: prior neural/tree recipe', 'P1': 'P1: prior recipe + ESP',
         'P2': 'P2: prior recipe + EUV lines', 'P3': 'P3: prior recipe + all EVE',
         'P4': 'P4: prior + 25% fast model', 'P5': 'P5: prior + 50% fast model',
         'P6': 'P6: prior + 75% fast model', 'starting_wind': 'Wind primary',
         'h9_fast_specialist': 'Fast primary (H9)', 'h8_sdo': 'H8: classical + all EVE',
         'matched_classical_wind': 'Matched classical wind',
         'h9_unweighted_lines': 'Classical + lines, unweighted',
         'h9_weight4_lines': 'Classical + lines, weight 4'}
WORDS: list[str] = []


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f(value, places=2):
    return '-' if value is None else f'{value:,.{places}f}'


def pct(value):
    return '-' if value is None else f'{100*value:.1f}%'


def style_fonts():
    pdfmetrics.registerFont(TTFont('Paper', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('PaperBold', 'C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFont(TTFont('PaperItalic', 'C:/Windows/Fonts/ariali.ttf'))
    pdfmetrics.registerFontFamily('Paper', normal='Paper', bold='PaperBold', italic='PaperItalic', boldItalic='PaperBold')
    return {
        'title': ParagraphStyle('Title', fontName='PaperBold', fontSize=22, leading=25.5,
                                textColor=NAVY, spaceAfter=9),
        'deck': ParagraphStyle('Deck', fontName='Paper', fontSize=11.2, leading=15,
                               textColor=MUTED, spaceAfter=13),
        'h1': ParagraphStyle('H1', fontName='PaperBold', fontSize=15.2, leading=19,
                             textColor=NAVY, spaceAfter=10),
        'h2': ParagraphStyle('H2', fontName='PaperBold', fontSize=11.4, leading=14.5,
                             textColor=NAVY, spaceBefore=8, spaceAfter=5),
        'body': ParagraphStyle('Body', fontName='Paper', fontSize=10.2, leading=13.6,
                               textColor=INK, spaceAfter=8),
        'small': ParagraphStyle('Small', fontName='Paper', fontSize=8.9, leading=11.5,
                                textColor=MUTED, spaceAfter=6),
        'table': ParagraphStyle('Table', fontName='Paper', fontSize=8.7, leading=11.0,
                                textColor=INK),
        'th': ParagraphStyle('TH', fontName='PaperBold', fontSize=8.6, leading=10.8,
                             textColor=colors.white),
        'ref': ParagraphStyle('Ref', fontName='Paper', fontSize=8.8, leading=11.2,
                              textColor=INK, spaceAfter=5),
    }


def para(text, key='body', track=True):
    if re.search('[\u2010-\u2015\u2212]', text):
        raise ValueError('Use ASCII hyphens in authored prose')
    if track:
        WORDS.append(re.sub('<[^>]+>', ' ', text))
    return Paragraph(text, ST[key])


def table(headers, rows, widths, small=False):
    data = [[para(str(v), 'th') for v in headers]]
    data += [[para(escape(str(v)), 'table') for v in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, 0), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
        ('TOPPADDING', (0, 1), (-1, -1), 4.5 if small else 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4.5 if small else 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('LINEBELOW', (0, 0), (-1, 0), .5, NAVY),
        ('LINEBELOW', (0, -1), (-1, -1), .6, LINE),
    ]))
    return t


def callout(text):
    t = Table([[para(text)]], colWidths=[WIDTH])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), PALE),
                           ('BOX', (0, 0), (-1, -1), .6, LINE),
                           ('LEFTPADDING', (0, 0), (-1, -1), 12),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                           ('TOPPADDING', (0, 0), (-1, -1), 10),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return t


def image(path, max_height):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(WIDTH / w, max_height / h)
    return Image(str(path), width=w*scale, height=h*scale)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(44, 38, A4[0]-44, 38)
    canvas.setFont('Paper', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(44, 25, 'WindNet low-cost study | Locked historical benchmark | 29 October 2026')
    canvas.drawRightString(A4[0]-44, 25, str(doc.page))
    canvas.restoreState()


def make_paper_figures(result):
    """Presentation-only layouts of predeclared groups; no new statistics."""
    TMP.mkdir(parents=True, exist_ok=True)
    FIGOUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 11, 'savefig.dpi': 200})
    paths = {}
    groups = [
        ('roles', ROLE_IDS, [SHORT[n] for n in ROLE_IDS]),
        ('neural', ['P0', 'P1', 'P2', 'P3'], ['Wind only', '+ ESP', '+ EUV lines', '+ Both']),
        ('classical', ['matched_classical_wind', 'h9_unweighted_lines', 'h9_weight4_lines', 'h9_fast_specialist'],
         ['Wind only', '+ Lines', 'Lines, weight 4', 'H9 blend']),
    ]
    for name, ids, labels in groups:
        fig, ax = plt.subplots(figsize=(7.05, 3.05))
        x = np.arange(len(ids))
        series = [('overall', 'Mean-fold all speeds', '#9baeb8'),
                  ('ge_500', 'Observed ≥500', '#2d8190'),
                  ('ge_600', 'Observed ≥600', '#c58a43')]
        for j, (key, label, color) in enumerate(series):
            values = [result['metrics'][n]['mean_rmse'] if key == 'overall' else
                      result['metrics'][n]['pooled']['fast_wind'][key]['continuous_observed_fast']['rmse'] for n in ids]
            loc = x + (j-1)*.24
            ax.bar(loc, values, .23, color=color, label=label)
            for at, val in zip(loc, values):
                ax.text(at, val+2.2, f'{val:.1f}', ha='center', va='bottom', rotation=90, fontsize=8.3)
        ax.set_xticks(x, labels)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylabel('RMSE (km/s)')
        ax.set_ylim(0, 225)
        ax.yaxis.grid(True, color='#dde4e7', lw=.6)
        ax.set_axisbelow(True)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=8.6)
        fig.subplots_adjust(left=.09, right=.995, bottom=.17, top=.84)
        target = FIGOUT/f'paper_{name}.png'
        fig.savefig(target, facecolor='white')
        plt.close(fig)
        paths[name] = target
    matrix = np.asarray([[result['metrics'][n]['pooled']['reference_skill'][ref][t]['mse_skill_score']
                          for ref in ['released_windnet', 'old_frozen_selected'] for t in ['all', 'ge_500', 'ge_600']]
                         for n in ROLE_IDS])
    lim = max(float(np.nanmax(np.abs(matrix))), .1)
    fig, ax = plt.subplots(figsize=(7.05, 2.9))
    im = ax.imshow(matrix, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(5):
        for j in range(6):
            v = matrix[i, j]
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(v) > .65*lim else '#20313b')
    ax.set_yticks(range(5), [SHORT[n] for n in ROLE_IDS], fontsize=9)
    ax.set_xticks(range(6), ['WN\nAll', 'WN\n≥500', 'WN\n≥600', 'Earlier\nAll', 'Earlier\n≥500', 'Earlier\n≥600'], fontsize=8.7)
    colorbar = fig.colorbar(im, ax=ax, fraction=.025, pad=.025)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.205, right=.92, bottom=.19, top=.98)
    target = FIGOUT/'paper_reference_skill.png'
    fig.savefig(target, facecolor='white')
    plt.close(fig)
    paths['skill'] = target
    return paths


def max_cumulative(record):
    found = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if ('cumulative' in key or 'total_estimate' in key) and isinstance(item, (int, float)):
                    found.append(float(item))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(record)
    return max(found) if found else None


def build():
    global ST
    r = load(LOCKED/'final_results.json')
    summary = load(ROOT/'output/report_summary.json')
    verdict = load(LOCKED/'outcome_verdict.json')
    if r.get('status') != 'completed' or r.get('experiment') != 'E13' or r.get('selection_from_final_test') is not False:
        raise ValueError('Paper requires the complete locked E13 evaluation')
    if summary.get('status') != 'completed_reporting' or summary.get('source_results_sha256') != sha(LOCKED/'final_results.json'):
        raise ValueError('Reporting summary does not match the completed frozen result')
    if verdict['predeclared_overall_gate_passed'] or verdict['predeclared_fast_gate_passed']:
        raise ValueError('This authored interpretation expects the recorded failed E13 gates; review narrative before generating a different study')
    if sha(LOCKED/'predictions.npz') != r['prediction_file_sha256']:
        raise ValueError('Frozen predictions differ')
    ST = style_fonts()
    figs = make_paper_figures(r)
    OUT.mkdir(parents=True, exist_ok=True)
    w = r['metrics']['released_windnet']
    old = r['metrics']['old_frozen_selected']
    overall = r['metrics']['starting_wind']
    fast = r['metrics']['h9_fast_specialist']
    prior = r['metrics']['P0']
    p = lambda model, t=500: r['metrics'][model]['pooled']['fast_wind'][f'ge_{t}']
    role_rows = [[SHORT[n], f(r['metrics'][n]['mean_rmse'])+' ± '+f(r['metrics'][n]['rmse_se']),
                  f(r['metrics'][n]['mean_mae']), f(r['metrics'][n]['fisher_mean_r'], 3),
                  f(p(n)['continuous_observed_fast']['rmse'])] for n in ROLE_IDS]
    n500, n600 = p('released_windnet')['continuous_observed_fast']['n'], p('released_windnet', 600)['continuous_observed_fast']['n']
    story = []
    add = story.append
    newpage = lambda title: (add(PageBreak()), add(para(title, 'h1')))

    # Page 1: scope and main conclusion before implementation detail.
    add(para('Compact solar features for<br/>solar-wind forecasting', 'title'))
    add(para('A locked comparison with WindNet and the earlier campaign winner', 'deck'))
    add(para('REPRODUCIBLE RESEARCH REPORT | 29 OCTOBER 2026', 'small'))
    add(para('Abstract', 'h2'))
    add(para(f'We tested whether inexpensive scalar measurements from the Solar Dynamics Observatory (SDO) help compact solar-wind forecasts. The study preserves WindNet\'s original five folds and 2,198 evaluation windows. Development used only Folds 1-4; recipes, code and data identities were frozen before a single final scoring pass. The predefined wind primary achieved mean-fold RMSE {f(overall["mean_rmse"])} km/s, compared with {f(w["mean_rmse"])} for released WindNet and {f(old["mean_rmse"])} for the earlier frozen winner. The fast primary achieved observed ≥500 km/s RMSE {f(p("h9_fast_specialist")["continuous_observed_fast"]["rmse"])} km/s, versus {f(p("released_windnet")["continuous_observed_fast"]["rmse"])} and {f(p("old_frozen_selected")["continuous_observed_fast"]["rmse"])}. However, fast-wind detection remained weaker than WindNet, and neither registered balanced-improvement gate passed. These are historical comparisons: the original holdouts had already been viewed in the earlier campaign, so the results are not fresh confirmation.'))
    add(callout('<b>Main result.</b> Lower continuous error did not establish an overall advance. The fast model reduces error on observed high-speed cases, but it misses many threshold exceedances. All negative ablations and the original strong references remain in the report.'))
    add(Spacer(1, 7))
    add(para('1. Research question and scope', 'h2'))
    add(para('WindNet uses solar images to forecast daily solar-wind speed [1]. Our budget limited direct image processing. We therefore continued a compact in-situ forecasting study using precomputed irradiances from SDO\'s Extreme ultraviolet Variability Experiment (EVE). The intended advance was a useful accuracy trade-off at low compute cost, not a claim that the new models improve WindNet\'s image architecture.'))
    add(para('Two roles were fixed before final scoring: the <b>wind primary</b>, chosen for overall development error, and the <b>fast primary</b>, chosen for the prespecified fast-wind objective. We also retained both the earlier campaign\'s actual saved winner and a faithful retraining of its accepted neural/tree recipe. This distinction corrects an initially missing prior-model baseline.'))
    add(para(f'Open artifacts: <link href="{BRANCH}">code, exact feature bundles and checkpoints</link>; <link href="{CAMPAIGN}">public hypothesis and experiment trail</link>.', 'small'))

    # Page 2: complete input and partition description.
    newpage('2. Data and the preserved splits')
    add(para('The target is daily mean solar-wind speed in km/s, aligned to the authors\' 211-Angstrom release. The bundled OMNI record covers 2011-2018 [2]. We recovered labels and original partition identities from the approximately 1.04 GB processed author archive, discarded its image arrays, and verified the compact target alignment. No image is a predictor in this study. The later EVE retrieval is a 20.7 MB scalar science table [3].'))
    add(para('WindNet\'s nominal history 4 / delay 3 parser creates 14 target windows from each 20-record block. We preserve evaluation offsets 6-19, while fitting uses only offsets 6-18, exactly retaining the authors\' 13-versus-14 asymmetry. Original observation steps are not always uninterrupted calendar days; we do not repair gaps or claim a uniform three-calendar-day lead.'))
    split = [[str(row['heldout_fold']), str(blocks), f"{row['fit_rows']:,}", f"{row['test_rows']:,}"]
             for row, blocks in zip(r['split_audit'], [37, 25, 26, 20, 49])]
    add(table(['Held-out fold', 'Original blocks', 'Fitting windows', 'Evaluation windows'], split,
              [90, 110, 140, WIDTH-340]))
    add(para('Table 1. Each final fit uses the other four original folds. The 157 blocks produce 2,198 scored windows. Duplicate issue-target aggregation is secondary: fixed equal-weight averaging leaves 2,053 pairs, without altering the primary original-window results.', 'small'))
    add(para('Inputs and missing values', 'h2'))
    add(table(['Block', 'What enters the model'], [
        ['Wind history: 34 columns', 'Four original observations; calendar speed lags 3-16, 24-31 and 51-58. The latter bands represent approximately 27- and 54-day recurrence.'],
        ['Other in-situ: 44 columns', 'Bx, By, Bz; their three RMS measures; temperature, proton density, and three dispersion estimates, each at lags 3-6.'],
        ['EVE: 62 optional columns', 'Nine log-positive irradiances at lags 5-8; four-day means and endpoint trends; eight log-ratio values. Components: 18 ESP and 44 emission-line features.']
    ], [145, WIDTH-145]))
    add(para('Temperature, density and their two dispersion inputs use log1p after nonpositive values are clamped. Density and its dispersion at or above 999 become missing. Ridge and neural preprocessing fit a mean imputer, missing indicators and standard scaling on fitting rows only; trees handle missing values natively. Historical measurements sourced from the withheld fold are masked on fitting rows before summaries or preprocessing. Entire fitting blocks would be purged if their target dates overlapped evaluation dates.'))
    add(para('During development, original Fold 5 targets and historical values from that fold were excluded. After freeze, the final fivefold protocol uses Fold 5 as training data for outer tests 1-4, just as any other training fold; its own outer test is fit only on Folds 1-4. No fold supplies its target labels to its own fit.'))

    # Page 3: features, exact methods and loop outcomes.
    newpage('3. Solar features and model development')
    add(para('EVE supplies six integrated emission lines at approximately 17.107, 19.512, 21.133, 30.378, 33.541 and 58.433 nm, plus three EUV SpectroPhotometer (ESP) bands labelled 171, 257 and 304 [3]. Each scalar uses target-day lags 5-8. Ratios are Fe XII / Fe XIV and Fe IX / Fe XVI, computed as log differences. Mean features require the four lag values; trends use the endpoints. Nonpositive fills remain missing, without gap interpolation.'))
    add(para('The short-wavelength detector failure in May 2014 creates substantial missing line coverage. Channels 1216 and 366 were excluded before fitting because of undocumented temporal support in a Fourier filter and a documented stability problem, respectively. These are retrospectively calibrated EVE V8 measurements, not an as-issued operational archive. The nominal lag guard does not erase that provenance limitation or recover the spatial coronal-hole information present in images.'))
    add(para('Fixed model families', 'h2'))
    add(para('Classical pairs combine ridge regression and shallow histogram gradient boosting (HGB). HGB uses 150 iterations, learning rate 0.05, seven leaves, minimum leaf size 30, L2=10 and seed 2026 in the frozen panel. The wind primary is 75% ridge (alpha 1000) and 25% tree. Matched feature and weighting controls instead fix the pair at 25% ridge / 75% tree. H9 blends 75% unweighted EUV-line forecasts with 25% forecasts trained with raw weight four on fitting speeds ≥500 km/s. Explicit imputation/scaling is unweighted; HGB weights also alter internal histogram construction and effective regularization [4].'))
    add(para('The earlier accepted recipe P0 is 50% tree and 50% three-seed residual neural mean. A ridge alpha-10 base supplies in-sample residual targets divided by 100. The network is 32/16 SiLU units, dropout 0.2, and a zero-initialized final layer; AdamW uses learning rate 0.002 and weight decay 0.1. Seeds 19/43/101 take exactly 35/1/15 full-batch steps. No new checkpoint search occurs. No old neural weights survived, so P0 is recipe retraining, not checkpoint resumption; the actual earlier frozen winner remains a separate comparator.'))
    add(table(['Stage', 'Hypothesis / observation'], [
        ['H7 / H10', 'Cancelled before fitting: prospective channel quality screening; later, the missing prior-neural control was corrected before further claims.'],
        ['H8', 'SDO features passed the training fast-error rule, with some overall degradation. Fifty fixed CV forecasts were retained.'],
        ['H9', 'Raw fast-case weighting passed its per-step rule. The selected 44-line-feature model improved fast error; cumulative overall degradation reached 1.49%.'],
        ['H11', 'Carried P0 forward and added ESP, lines, both, and fixed diversity mixtures. None met the registered 1% overall-gain gate; P0 was retained within this family.'],
        ['H12 / E13', 'Fixed conditional stability audit, then one locked scoring pass. H12 made no selection; E13 does not return to tuning.']
    ], [65, WIDTH-65], small=True))

    # Page 4: main final comparisons and aggregate interpretation.
    newpage('4. Locked comparison with both strong references')
    add(para('All 15 forecasts were fixed before evaluation. The released WindNet column is the authors\' saved forecast, not a retrained or newly selected checkpoint. The earlier frozen winner uses the previous campaign\'s saved forecasts; it is not replaced by the potentially weaker P0 refit. Original row, fold, block, offset, date and target identities were checked before comparison.'))
    add(para('Root-mean-square error (RMSE) emphasizes large mistakes. Mean absolute error (MAE) averages error magnitudes. Both are better when smaller; correlation r measures how well forecasts track variation.'))
    add(table(['Fixed role', 'RMSE ± SE', 'MAE', 'Fisher r', '≥500 RMSE'], role_rows,
              [136, 112, 65, 65, WIDTH-378]))
    add(para('Table 2. Overall RMSE and MAE are unweighted means across the five original folds; SE is descriptive fold SD / √5. Correlations are Fisher-z averaged. The last column pools 400 observed ≥500 windows. These are different declared aggregations, not interchangeable denominators. Errors are km/s.', 'small'))
    add(image(figs['roles'], 225))
    add(para('Figure 1. Fixed roles, without score-based promotion. All-speed bars use mean-fold RMSE; fast bars pool windows conditioned on observed speed. Every missed fast forecast remains in the fast-error calculation.', 'small'))
    add(para(f'The wind primary lowers mean-fold RMSE by {f(w["mean_rmse"]-overall["mean_rmse"])} km/s versus WindNet, but by only {f(old["mean_rmse"]-overall["mean_rmse"])} versus the earlier winner. Its MAE {f(overall["mean_mae"])} remains above WindNet\'s {f(w["mean_mae"])}. WindNet also retains the higher mean correlation ({f(w["fisher_mean_r"],3)} versus {f(overall["fisher_mean_r"],3)}). Thus a lower RMSE alone is not an overall superiority result.'))
    add(para('The registered overall gate required at least 1% RMSE reduction against both references with no MAE degradation. The fast gate additionally required improved error, no worse CSI, and a bounded overall-error cost. Neither gate passed. This verdict is preserved even though individual ablations have favorable numerical scores.'))

    # Page 5: high speed is more than conditional RMSE.
    newpage('5. Fast-wind error, detection and skill')
    add(para(f'We define fast wind as a daily mean ≥500 km/s (primary) and ≥600 km/s (secondary), yielding {n500} and {n600} original windows. These are declared thresholds, not universal definitions or matched high-speed-stream episodes [5]. Conditional RMSE measures errors on observed fast cases. Detection separately asks whether the forecast crosses the same threshold.'))
    detection = []
    for name in ['released_windnet', 'old_frozen_selected', 'h9_fast_specialist']:
        for threshold in (500, 600):
            fast_case = p(name, threshold)
            d = fast_case['detection']
            detection.append([SHORT[name], f'≥{threshold}', f(fast_case['continuous_observed_fast']['rmse']),
                              f"{d['tp']}/{d['observed_fast_days']}", str(d['fp']), f(d['pod'],3), f(d['far'],3), f(d['csi'],3)])
    add(table(['Role', 'Speed', 'RMSE', 'Hits / fast', 'FP', 'POD', 'FAR', 'CSI'], detection,
              [108, 45, 55, 72, 36, 60, 60, WIDTH-436]))
    add(para('Table 3. POD=TP/(TP+FN); FAR=FP/(TP+FP); CSI=TP/(TP+FP+FN). A hit is a forecast and observation both exceeding the cutoff. These counts retain original-window duplicate weights and are not independent events.', 'small'))
    add(para(f'The fast primary reduces ≥500 RMSE to {f(p("h9_fast_specialist")["continuous_observed_fast"]["rmse"])} km/s, but its CSI is {f(p("h9_fast_specialist")["detection"]["csi"],3)}, below WindNet\'s {f(p("released_windnet")["detection"]["csi"],3)}. At ≥600, it detects only {p("h9_fast_specialist",600)["detection"]["tp"]} of {n600} cases, compared with {p("released_windnet",600)["detection"]["tp"]} for WindNet. Its smaller false-alarm count does not compensate for the many missed extremes.'))
    add(image(figs['skill'], 205))
    add(para('Figure 2. Paired MSE skill = 1 - MSE(model)/MSE(reference), on identical finite cases. WN denotes released WindNet. Positive values favor the row model, but do not imply better threshold detection.', 'small'))
    pers = fast['pooled']['reference_skill']['persistence_27day']
    add(para(f'Both causal persistence references are retained. Latest-speed persistence covers all 2,198 windows; 27-day persistence covers {pers["all"]["n_common_finite"]:,}. Its ≥500/≥600 paired subsets contain {pers["ge_500"]["n"]}/{pers["ge_600"]["n"]} windows. H9\'s corresponding MSE skill is {f(pers["ge_500"]["mse_skill_score"],3)}/{f(pers["ge_600"]["mse_skill_score"],3)}: it still loses to 27-day persistence at the higher cutoff. No missing reference was filled to manufacture coverage.'))

    # Page 6: additions isolated from mixture and loss changes.
    newpage('6. Controlled ablations: what each addition changes')
    add(para('The neural-feature ablation fixes the entire prior recipe, seeds and training steps. Only the appended feature block changes. The classical ablation fixes the 25% ridge / 75% tree mixture, separating the effect of EUV lines from the effect of raw fast-case weighting and the final H9 mixture. This prevents a changed ensemble weight from being mislabelled as a pure solar-feature gain.'))
    add(image(figs['neural'], 210))
    add(para('Figure 3. Prior neural/tree recipe: wind only (P0), +18 ESP features, +44 EUV-line features, and +62 combined features. All data transformations and learner settings are otherwise fixed.', 'small'))
    add(image(figs['classical'], 210))
    add(para('Figure 4. Matched classical family: adding lines, giving observed fast fitting cases raw weight four, then blending 25% weighted with 75% unweighted line forecasts. Stronger weighting reduces tail error while increasing ordinary-case error.', 'small'))
    esp = r['metrics']['P1']['mean_rmse']
    line = r['metrics']['h9_unweighted_lines']['mean_rmse']
    raw = r['metrics']['h9_weight4_lines']['mean_rmse']
    add(para(f'The final ESP ablation has mean-fold RMSE {f(esp)} km/s versus P0\'s {f(prior["mean_rmse"])}. It remains a descriptive locked-panel result: H11 had not adopted it using training-only CV, and we do not promote it after seeing test scores. In the classical family, unweighted lines give {f(line)}, raw weight four gives {f(raw)}, and the frozen H9 blend gives {f(fast["mean_rmse"])}. These outcomes expose the accuracy trade-off rather than hiding rejected alternatives.'))

    # Page 7: conditional uncertainty, limitations and reproducibility.
    newpage('7. Uncertainty, limitations and reproducibility')
    add(para('The training-only H12 audit recomputed every reported forecast, component and fold metric. It used 5,000 paired resamples of 99 connected block components and, separately, 52 fixed 54-day bins. H11 made no adoption, so H12 supplied no new positive model-selection claim. Final uncertainty uses 5,000 paired 54-day UTC-bin resamples: 54 occupied bins across all windows and 32 on original Fold 5. Predictions, roles and thresholds stay fixed.'))
    uncertainty_rows = []
    for scope, model, ref, cond in [
        ('all_original_windows','starting_wind','released_windnet','all'),
        ('all_original_windows','h9_fast_specialist','released_windnet','ge_500'),
        ('all_original_windows','h9_fast_specialist','old_frozen_selected','ge_500'),
        ('original_fold5','starting_wind','released_windnet','all'),
        ('original_fold5','h9_fast_specialist','released_windnet','ge_500'),
        ('original_fold5','h9_fast_specialist','old_frozen_selected','ge_500')]:
        v = r['paired_final_uncertainty']['scopes'][scope]['comparisons'][model][ref][cond]
        ci = v['percentile95_km_s']
        uncertainty_rows.append([SHORT[model]+' vs '+SHORT[ref], 'All' if scope=='all_original_windows' else 'Fold 5',
                                 'All' if cond=='all' else '≥500', f(v['delta_rmse_km_s']), f'[{f(ci[0])}, {f(ci[1])}]'])
    add(table(['Comparison', 'Scope', 'Cases', 'Δ RMSE', 'Conditional 95% interval'], uncertainty_rows,
              [185, 47, 42, 60, WIDTH-334], small=True))
    add(para('Table 4. Paired pooled RMSE(model) minus RMSE(reference), km/s; negative favors the model. These intervals refer to pooled errors, not the mean-fold RMSE in Table 2. They are conditional stability diagnostics, not selection-adjusted confidence statements.', 'small'))
    add(para('Fold 5 does not confirm the pooled fast-wind advantage against WindNet: that difference is positive and its interval spans zero. More broadly, original Folds 1-4 informed architecture and model choices, and Fold 5 had already been viewed in the earlier campaign. Random-block historical folds are not an operational chronological replay. Bootstrap resampling cannot remove that adaptive selection, model-fitting uncertainty or all remaining temporal dependence.'))
    add(para('The input information also differs from WindNet: long in-situ history and extra plasma/magnetic measurements are added while solar-disk geometry is lost. The experiment therefore does not isolate an image-architecture improvement. Retrospective EVE calibration, missing post-2014 lines, a limited number of extreme cases, and persistent fast-speed underprediction further constrain the claim. New validation would require genuinely unused data, not another round of tuning on these holdouts.'))
    runtime = r['environment']
    costfile = LOCKED/'costs.json'
    cost = max_cumulative(load(costfile)) if costfile.exists() else None
    costphrase = f'The conservative cumulative estimate is US${cost:.3f}, below the US$5 cap; it is not a settled invoice.' if cost is not None else 'The public cost ledger tracks the US$5 cumulative cap; recorded estimates are not settled invoices.'
    add(para(f'Reproduction uses immutable source/code hashes, the public feature-only bundles, explicit freeze-manifest checks and a one-shot execution marker. E13 used Python {runtime["python"]}, PyTorch {runtime["torch"]} on CPU, NumPy 2.3.1, SciPy 1.16.1 and scikit-learn 1.9.1; the locked evaluator took {r["seconds"]:.1f} seconds inside the worker. Forty portable fitted fold checkpoints are retained in a {r["fitted_checkpoints"]["archive_bytes"]/1e6:.2f} MB archive, fixing the earlier lack of saved weights. {costphrase}'))

    # Page 8: full fixed-order panel and concise references.
    newpage('8. Complete fixed panel and open artifacts')
    add(para('Table 5 includes every final model in protocol order. It is not a leaderboard used to choose a new winner. The wind and fast primary roles remain the choices made before final scoring; the full training-CV grids, null scores, per-fold reports, prediction arrays and provenance are public.', 'body'))
    full_rows = []
    for name in r['model_ids']:
        m = r['metrics'][name]
        full_rows.append([LABEL[name], f(m['mean_rmse']), f(m['mean_mae']),
                          f(p(name)['continuous_observed_fast']['rmse']),
                          f(p(name,600)['continuous_observed_fast']['rmse']),
                          f(p(name)['detection']['csi'],3)])
    add(table(['Fixed model', 'RMSE', 'MAE', '≥500 RMSE', '≥600 RMSE', 'CSI ≥500'], full_rows,
              [188, 51, 51, 73, 73, WIDTH-436], small=True))
    add(para('Overall RMSE/MAE are means across original folds; fast columns pool observed exceedances. All speeds/errors are km/s. Appendix-level details, including POD/FAR/FPR/TSS/HSS, bias, paired coverage, undefined draws and all bootstrap scopes, remain in the machine-readable report rather than being suppressed.', 'small'))
    add(para(f'<b>Open materials.</b> <link href="{BRANCH}">Public code, original instructions and frozen artifacts</link>; <link href="{BRANCH}/results/locked_final">locked comparison and checkpoints</link>; <link href="{CAMPAIGN}">experiment DAG and protocols</link>. The paper-generation source is included. The initial three-page report remains a separate historical artifact.', 'small'))
    add(para('References', 'h2'))
    refs = [
        '[1] Upendran V, Cheung MCM, Hanasoge S, Krishnamurthi G (2020). Solar wind prediction using deep learning. Space Weather 18, e2020SW002478. <link href="https://doi.org/10.1029/2020SW002478">doi:10.1029/2020SW002478</link>. Original code: Vishal-Upendran/WindNet, commit a996baa075ecdbd2b1c15533002692e29db43c19.',
        '[2] NASA OMNIWeb. <link href="https://omniweb.gsfc.nasa.gov/html/ow_data.html">OMNI data documentation</link>. Daily near-Earth solar-wind quantities used through the authors\' bundled release.',
        '[3] LASP. <link href="https://lasp.colorado.edu/eve/data_access/eve_data/products/level3/EVE_L3_V8_README.pdf">SDO/EVE Level 3 Version 8 science product README</link> (2024); official merged scalar product, retrieved 29 October 2026. Related ESP V8 documentation records channel limitations.',
        '[4] scikit-learn 1.9.1. <link href="https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/ensemble/_hist_gradient_boosting/binning.py">Histogram binning implementation</link> and gradient-boosting source. Training weights influence internal binning as well as fitting.',
        '[5] Owens MJ (2018). Time-Window Approaches to Space-Weather Forecast Metrics: A Solar Wind Case Study. <link href="https://doi.org/10.1029/2018SW002059">doi:10.1029/2018SW002059</link>. Our inclusive daily 500/600 km/s definitions are explicitly fixed study choices.',
    ]
    for reference in refs:
        add(para(reference, 'ref'))

    path = OUT/'windnet-full-study.pdf'
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=44, rightMargin=44,
                            topMargin=43, bottomMargin=49, title='Compact solar features for solar-wind forecasting',
                            author='WindNet low-cost research study', subject='Locked historical benchmark with original WindNet splits, scalar SDO/EVE features and fast-wind verification')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    pages = PdfReader(path).pages
    text = '\n'.join(page.extract_text() or '' for page in pages)
    report = {'pdf': str(path.resolve()), 'pages': len(pages), 'extracted_words': len(text.split()),
              'sha256': sha(path), 'source_results_sha256': sha(LOCKED/'final_results.json'),
              'source_report_summary_sha256': sha(ROOT/'output/report_summary.json'),
              'new_model_selection': False, 'balanced_gate_passed': False,
              'original_short_pdf_untouched': True}
    required_numbers = ['77.44', '80.28', '77.87', '122.45', '138.91', '130.92', '4/110', '19/110', '0.213']
    report['numeric_text_checks'] = {value: value in text for value in required_numbers}
    report['paper_figures'] = {name: str(path.resolve()) for name, path in figs.items()}
    report['body_font_pt'] = ST['body'].fontSize
    report['missing_glyphs'] = '\u25a0' in text or '\ufffd' in text
    if not all(report['numeric_text_checks'].values()) or report['missing_glyphs'] or len(pages) != 8:
        raise ValueError('Final paper text/page checks failed')
    (TMP/'paper_check.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))
    return path


if __name__ == '__main__':
    build()
