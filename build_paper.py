"""Build the concise, three-page report from the verified study results."""
from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
DATA = json.loads((ROOT / 'results/summary.json').read_text(encoding='utf-8'))
M = DATA['metrics']
A = DATA['ablations']['variants']

NAVY = colors.HexColor('#18384c')
BLUE = colors.HexColor('#245e86')
RUST = colors.HexColor('#a95736')
INK = colors.HexColor('#202d36')
MUTED = colors.HexColor('#566875')
LINE = colors.HexColor('#d8e0e5')
PALE = colors.HexColor('#f0f5f8')
PAPER = colors.HexColor('#fbfcfd')
WIDTH = A4[0] - 84

STYLES = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=21.5,
                            leading=25, textColor=NAVY, spaceAfter=7),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=9.2,
                               leading=12, textColor=MUTED, spaceAfter=13),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=16,
                         leading=20, textColor=NAVY, spaceAfter=9),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11.5,
                         leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10.2,
                           leading=13.5, textColor=INK, spaceAfter=7),
    'caption': ParagraphStyle('caption', fontName='Helvetica', fontSize=8.8,
                              leading=11.5, textColor=MUTED, spaceAfter=7),
    'table': ParagraphStyle('table', fontName='Helvetica', fontSize=9.2,
                            leading=11.8, textColor=INK),
    'tablehead': ParagraphStyle('tablehead', fontName='Helvetica-Bold', fontSize=8.8,
                                leading=11, textColor=NAVY),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.6,
                            leading=11.1, textColor=MUTED, spaceAfter=4),
}


def p(text, style='body'):
    return Paragraph(text, STYLES[style])


def table(headers, rows, widths, highlight=None):
    content = [[p(str(x), 'tablehead') for x in headers]]
    content += [[p(str(x), 'table') for x in row] for row in rows]
    t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('LINEBELOW', (0, 0), (-1, 0), .65, LINE),
        ('LINEBELOW', (0, -1), (-1, -1), .65, LINE),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 5.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
    ]
    for row in range(2, len(content), 2):
        commands.append(('BACKGROUND', (0, row), (-1, row), PAPER))
    if highlight is not None:
        commands.append(('BACKGROUND', (0, highlight + 1), (-1, highlight + 1), PALE))
    t.setStyle(TableStyle(commands))
    return t


def figure(name, width=WIDTH):
    path = ROOT / 'output/figures' / name
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw, hAlign='CENTER')


def metric_row(label, name):
    v = M[name]
    return [label, f"{v['mean_rmse']:.2f}", f"{v['fisher_mean_r']:.3f}",
            f"{v['folds'][4]['rmse']:.2f}"]


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.5)
    canvas.line(42, 31, A4[0] - 42, 31)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(42, 19, 'WindNet low-cost study | 29 October 2026 | Reproducible research report')
    canvas.drawRightString(A4[0] - 42, 19, str(doc.page))
    canvas.restoreState()


def build():
    story = []
    story += [
        p('Affordable solar-wind forecasting<br/>on WindNet\'s original holdouts', 'title'),
        p('A compact experimental study with controlled ablations | 29 October 2026', 'subtitle'),
        p('<b>Abstract.</b> We tested inexpensive solar-wind forecasts using the original WindNet evaluation '
          'folds, without using solar images as predictors. Four hypothesis cycles led from short-history '
          'regression to recurrence features and a small ensemble. The frozen selection rule lowered mean '
          'root-mean-square error (RMSE) from <b>80.28 to 77.87 km/s</b> across five folds, but correlation '
          'fell from <b>0.550 to 0.515</b>. On Fold 5, which stayed sealed during model design, WindNet '
          'performed better. Feature ablations identify the 27-day recurrence band as the main useful '
          'addition. These findings support a cheap, transparent baseline, <b>not an established overall '
          'improvement over WindNet</b>.'),
        p('1. Same evaluation rows, different inputs', 'h2'),
        p('We recovered the authors\' released 211-Angstrom folds and retained all <b>2,198 forecast rows</b>. '
          'Every target matched the authors\' saved WindNet predictions exactly; their mean RMSE and '
          'correlation reproduce the published 80.28 km/s and 0.55 [1, 2]. The forecast uses four days of '
          'recent input with a three-day delay (H = 4, D = 3). Our models replace images with daily OMNI '
          'speed history and optional plasma and magnetic measurements [3].'),
        p('2. Main comparison', 'h2'),
    ]
    rows = [
        metric_row('Released WindNet', 'released_windnet'),
        metric_row('Four-speed ridge', 'short_ridge'),
        metric_row('Speed-recurrence ridge', 'recurrence_ridge'),
        metric_row('Multivariate ridge', 'multi_ridge'),
        metric_row('Residual neural ensemble', 'residual_nn'),
        metric_row('Convex ensemble', 'convex_ensemble'),
        metric_row('Frozen selection rule', 'selected_model'),
    ]
    story.append(table(['Method', 'Mean RMSE<br/>(km/s)', 'Mean r', 'Fold 5 RMSE<br/>(km/s)'],
                       rows, [213, 105, 78, WIDTH - 396], highlight=6))
    story += [
        Spacer(1, 5),
        p('Five-fold means are exploratory: Folds 1-4 informed model design. Correlations are averaged in '
          'Fisher-z space. Fold 5 is the primary comparison; lower RMSE is better.', 'caption'),
        figure('original_folds.png'),
        p('<b>Figure 1.</b> Error on each original held-out fold. The compact method has lower mean error, '
          'but the untouched Fold 5 favors WindNet (76.35 versus 78.06 km/s).', 'caption'),
        PageBreak(),
        p('3. What actually helped?', 'h1'),
        p('The following additions keep the model family fixed to <b>ridge regression</b>. Each row uses '
          'the same held-out targets and training-only preprocessing, with its regularization penalty '
          'chosen inside the training folds. These requested ablations were planned after the final '
          'holdout was opened, so their interpretation is exploratory.'),
    ]
    additions = [
        ('Four recent speeds', 'short_4'),
        ('Longer recent history only', 'recent_only'),
        ('Add the 27-day recurrence band', 'recent_plus_27'),
        ('Add the 54-day band (all speed features)', 'speed_recurrence'),
        ('Add magnetic and plasma measurements', 'full'),
    ]
    addition_rows = []
    previous = None
    for label, name in additions:
        value = A[name]
        delta = '-' if previous is None else f"{value['mean_rmse'] - previous:+.2f}"
        addition_rows.append([label, f"{value['mean_rmse']:.2f}", delta, f"{value['folds'][4]['rmse']:.2f}"])
        previous = value['mean_rmse']
    story.append(table(['Feature additions', 'Mean RMSE<br/>(km/s)', 'Change from<br/>previous', 'Fold 5 RMSE<br/>(km/s)'],
                       addition_rows, [235, 91, 89, WIDTH - 415]))
    story += [
        Spacer(1, 4),
        p('A negative change is an error reduction. Extending recent history alone did not help. The '
          '27-day band produced the largest incremental gain. Improvements in the five-fold average '
          'did not consistently carry over to Fold 5.', 'caption'),
        p('Remove one component from the full model', 'h2'),
        p('A complementary test removes one feature group from the full 78-column ridge model. Its '
          'reference RMSE is 78.38 km/s. This separates the 27-day and 54-day bands from the two '
          'measurement groups.'),
        figure('feature_ablations.png'),
        p('<b>Figure 2.</b> Mean error change when a feature group is removed. The 27-day band provides '
          'most of the measured benefit in this model family.', 'caption'),
    ]
    removal_rows = []
    for label, name in [
        ('Remove 27-day band', 'no_27_band'),
        ('Remove 54-day band', 'no_54_band'),
        ('Remove magnetic measurements', 'no_magnetic'),
        ('Remove plasma measurements', 'no_plasma'),
    ]:
        value = A[name]
        removal_rows.append([label, f"{value['mean_rmse']:.2f}", f"{value['delta_rmse_vs_full']:+.2f}",
                             f"{value['folds'][4]['rmse']:.2f}"])
    story.append(table(['Controlled removal', 'Mean RMSE<br/>(km/s)', 'Change from<br/>full model', 'Fold 5 RMSE<br/>(km/s)'],
                       removal_rows, [235, 91, 89, WIDTH - 415]))
    story += [
        Spacer(1, 6),
        p('<b>Interpretation.</b> Removing the 27-day band costs 4.50 km/s; removing the 54-day band '
          'costs 0.67 km/s. Magnetic measurements add only 0.21 km/s of mean benefit. Plasma measurements '
          'add essentially none: their removal changes RMSE by -0.0006 km/s. Neural and ensemble '
          'results in Table 1 are model-family comparisons, not clean feature-removal tests.'),
        PageBreak(),
        p('4. Hypothesis, experiment, observation', 'h1'),
    ]
    loop_rows = [
        ['H1', 'Regularize four-speed regression.', '88.78 km/s on development Fold 4; slightly better than fixed SVMs.'],
        ['H2', 'Add recurrence and in-situ measurements.', '74.83 km/s; a 15.72% reduction from H1.'],
        ['H3', 'Add a GPU residual neural network.', '74.47 km/s, only 0.90% below augmented ridge; the 2% criterion failed.'],
        ['H4', 'Average different model errors.', '73.85 km/s; passed the 1% gain criterion against the accepted tree rule.'],
        ['E5', 'Freeze and evaluate original holdouts.', 'Lower five-fold mean error, but no confirmed advantage on untouched Fold 5.'],
        ['E6', 'Remove feature groups one at a time.', '27-day recurrence helps most; plasma adds essentially no average benefit.'],
    ]
    story.append(table(['Step', 'Question tested', 'Observation'], loop_rows, [39, 190, WIDTH - 229]))
    story += [
        Spacer(1, 4),
        p('H1-H4 used Folds 1-3 for fitting and Fold 4 for development; their scores are not independent '
          'test results. All six substantive experiments, including negative findings, were logged in '
          'RogNet/FixRelay. E6 is explicitly post-evaluation.', 'caption'),
        p('5. Evaluation safeguards and limits', 'h2'),
        p('The original release contains 157 blocks of 20 days, distributed across the five folds as '
          '37, 25, 26, 20 and 49 blocks. We preserved the original 14 test windows per block (target '
          'offsets 6-19) and the training script\'s 13 windows. Repeated dates inside a fold were retained; '
          'no date crossed folds. All fitting lag values originating in withheld folds were masked '
          'before preprocessing. Missing values were handled inside training data only.'),
        p('Speed inputs use lags 3-16, 24-31 and 51-58 days, alongside the original four recent speeds. '
          'Auxiliary measurements use lags 3-6. For each outer holdout, hyperparameters were selected '
          'on an inner training fold before refitting. The neural model used three fixed seeds. This '
          'changes the information available to the model relative to WindNet, so the comparison '
          'does not isolate an architectural advance.'),
        p('On Fold 5, the compact method\'s RMSE was 1.71 km/s higher than WindNet\'s. A paired bootstrap '
          'with 5,000 resamples and 33 clusters of overlapping original blocks gave a 95% interval '
          'of <b>-6.16 to +7.95 km/s</b> for compact-minus-WindNet RMSE. The interval includes zero. '
          'The original WindNet code also selected checkpoints using its held-out fold. Random block '
          'splits and this small retrospective study cannot establish operational, chronological '
          'forecasting skill.'),
        p('6. Reproduce and extend', 'h2'),
        p('The compact input file is about 265 KB. No raw SDO archive was downloaded and no image was '
          'used as a predictor; bundled processed images were discarded while recovering labels and '
          'fold identities. Main experiments ran on a Runpod 24 GB GPU slice, with estimated compute '
          'spending of about US$0.12 (billing had not settled). The temporary GPU and volume were '
          'deleted. Controlled ridge ablations ran locally. Code, exact predictions, split checks and '
          'experiment records are available in the public repository:'),
        p('<link href="https://github.com/kahirhumibntwerag/windnet-low-cost-study" color="#245e86">'
          'github.com/kahirhumibntwerag/windnet-low-cost-study</link>', 'small'),
        p('<b>References</b>', 'h2'),
        p('[1] Upendran, V., Cheung, M. C. M., Hanasoge, S., &amp; Krishnamurthi, G. (2020). '
          '<i>Solar wind prediction using deep learning.</i> Space Weather, 18, e2020SW002478. '
          '<link href="https://doi.org/10.1029/2020SW002478" color="#245e86">doi:10.1029/2020SW002478</link>.', 'small'),
        p('[2] Authors\' code and linked fold/prediction release: '
          '<link href="https://github.com/Vishal-Upendran/WindNet" color="#245e86">github.com/Vishal-Upendran/WindNet</link> '
          '(revision a996baa). All 2,198 reference targets were matched exactly.', 'small'),
        p('[3] NASA SPDF. <i>OMNIWeb data documentation.</i> '
          '<link href="https://omniweb.gsfc.nasa.gov/html/ow_data.html" color="#245e86">'
          'omniweb.gsfc.nasa.gov/html/ow_data.html</link>.', 'small'),
    ]
    doc = SimpleDocTemplate(
        str(OUT / 'windnet-study.pdf'), pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=35, bottomMargin=41, title='Affordable solar-wind forecasting on WindNet\'s original holdouts',
        author='WindNet Low-Cost Study', subject='Original-holdout comparison, hypothesis loops and controlled ablations',
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUT / 'windnet-study.pdf')


if __name__ == '__main__':
    build()
