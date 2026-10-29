/* This page reads only the explicit training-only continuation artifact. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits = 1) => finite(value) ? value.toFixed(digits) : '—';
  const integer = value => finite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 0}) : '—';
  const thresholds = [500, 600];
  const experiments = {
    H8: {file: './continuation.json', title: 'Can compact solar features help?', url: 'https://fixrelay.vercel.app/experiments/a772eb55-1daf-4787-84df-4fdd8f662421'},
    H9: {file: './fast_loss.json', title: 'Can fast-wind-aware training help?', url: 'https://fixrelay.vercel.app/experiments/f3956e16-c32c-42cb-9d67-6b8ac6aa1970'}
  };
  const groups = {wind_only: ['Wind only', 0], wind_esp: ['Wind + ESP diodes', 18], wind_lines: ['Wind + EVE lines', 44], wind_all_eve: ['Wind + all EVE', 62]};
  const parameter = new URLSearchParams(location.search || '').get('experiment')?.toUpperCase();
  const state = {data: null, rows: [], viewed: null, selected: null, experiment: null, requested: experiments[parameter] ? parameter : 'auto', loadToken: 0};
  const incumbentLabel = () => state.experiment === 'H9' ? 'H8 reference' : 'Wind-only incumbent';
  const make = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  const titleCase = value => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
  const getMetrics = report => report?.metrics || report?.report?.metrics || report?.scores || report || {};
  const fast = (row, threshold) => row.metrics.fast_wind?.[`ge_${threshold}`] || row.metrics.fast?.[`ge_${threshold}`] || {};
  const fastError = (row, threshold) => fast(row, threshold).continuous_observed_fast || fast(row, threshold).continuous || {};
  const referenceAliases = {
    latest_persistence: ['latest_persistence', 'latest', 'persistence_latest'],
    persistence_27day: ['persistence_27day', '27day', 'persistence_27_day', 'rotation_persistence']
  };
  const referenceLabel = name => name === 'latest_persistence' ? 'Latest available speed' : name === 'persistence_27day' ? '27-day persistence' : titleCase(name);
  const skill = (row, name, scope = 'all') => {
    const values = row.metrics.persistence_skill || row.metrics.skill || {};
    for (const key of referenceAliases[name] || [name]) if (values[key]) return values[key][scope] || {};
    return {};
  };

  function normalize(data) {
    let entries;
    if (data.candidate_reports && typeof data.candidate_reports === 'object') entries = Object.entries(data.candidate_reports);
    else {
      const source = data.candidates || data.rows || data.experiments || data.results || [];
      entries = Array.isArray(source) ? source.map((row, index) => [String(row.id || row.name || `candidate_${index + 1}`), row]) : Object.entries(source);
    }
    const specs = data.candidate_specs || data.specs || {};
    const representatives = new Set(Object.values(data.best_by_family || {}));
    const selected = typeof data.selected === 'string' ? data.selected : data.selected?.id || data.selected_candidate || null;
    return entries.map(([id, report], index) => {
      const meta = {...(report.metadata || {}), ...(specs[id] || {})};
      const family = meta.family || report.family || '';
      const usesEve = typeof meta.uses_eve === 'boolean' ? meta.uses_eve : typeof report.uses_eve === 'boolean' ? report.uses_eve : /^eve_|\beve\b/i.test(`${family} ${id}`);
      const spec = meta.spec || report.spec || {};
      const params = spec.params || {};
      let label = report.label || meta.label;
      if (!label) {
        if (meta.feature_group) {
          const group = groups[meta.feature_group]?.[0] || titleCase(meta.feature_group);
          if (finite(meta.weighted_pair_fraction)) label = `${group} · loss ×${meta.weighted_loss_multiplier} · ${Math.round(meta.weighted_pair_fraction * 100)}% weighted / ${Math.round((1 - meta.weighted_pair_fraction) * 100)}% unweighted`;
          else label = `${group} · ${meta.loss_weight_observed_ge_500 === 1 ? 'unweighted pair' : `loss ×${meta.loss_weight_observed_ge_500} · 100% weighted pair`}`;
        } else if (family === 'h8_reference') label = 'H8 reference · 25% ridge + 75% trees · all EVE';
        else if (Array.isArray(meta.members) && meta.members.length) {
          const members = meta.members.map((member, memberIndex) => {
            const item = specs[member] || {};
            const kind = item.wind_kind || (/recurrence/.test(member) ? 'recurrence' : /multi/.test(member) ? 'multi' : '');
            const learnerType = `${item.spec?.family || ''} ${item.family || ''} ${member}`;
            const learner = /ridge/.test(learnerType) ? 'ridge' : /tree|hgb|gradient/.test(learnerType) ? 'trees' : member;
            return `${finite(meta.weights?.[memberIndex]) ? `${Math.round(meta.weights[memberIndex] * 100)}% ` : ''}${kind} ${learner}`.trim();
          });
          label = `Blend · ${members.join(' + ')} · ${usesEve ? 'wind + EVE' : 'wind only'}`;
        }
      }
      if (!label) {
        const parts = [];
        if (/blend/i.test(family || id)) parts.push('Blend');
        else if (spec.family === 'ridge' || /ridge/i.test(family || id)) parts.push(finite(params.alpha) ? `Ridge · α ${params.alpha}` : 'Ridge');
        else if (/tree|hgb|gradient/i.test(`${family} ${spec.family} ${id}`)) parts.push(finite(params.max_leaf_nodes) ? `Trees · ${params.max_leaf_nodes} leaves / min ${params.min_samples_leaf}` : 'Boosted trees');
        else parts.push(titleCase(family) || id);
        const windKind = meta.wind_kind || report.wind_kind || (/multi/.test(family || id) ? 'multi' : /recurrence/.test(family || id) ? 'recurrence' : '');
        if (windKind) parts.push(windKind === 'multi' ? 'multi-rotation' : 'recurrence');
        parts.push(usesEve ? 'wind + EVE' : 'wind only');
        label = parts.join(' · ');
      }
      return {id, index, report, meta, family, usesEve, label, metrics: getMetrics(report),
        selected: id === selected || report.selected === true,
        incumbent: id === data.incumbent,
        representative: representatives.has(id),
        eligibility: data.candidate_eligibility?.[id] || data.eve_eligibility?.[id] || report.eligibility || null};
    });
  }

  function metricCell(value, digits = 1, direction = null) {
    const cell = make('td', fmt(value, digits), 'num');
    if (!finite(value)) { cell.classList.add('undefined'); cell.title = 'Not reported or mathematically undefined'; }
    else if (direction && value !== 0) cell.classList.add((direction === 'positive' ? value > 0 : value < 0) ? 'good' : 'bad');
    return cell;
  }

  function countCell(value) {
    const cell = make('td', integer(value), 'num');
    if (!finite(value)) {cell.classList.add('undefined'); cell.title = 'Not reported';}
    return cell;
  }

  function setViewed(id) {
    if (!state.rows.some(row => row.id === id)) return;
    state.viewed = id;
    $('model-select').value = id;
    const row = state.rows.find(item => item.id === id);
    $('view-note').textContent = `Inspecting: ${row.label}${row.selected ? ' · recorded selection' : row.incumbent ? ` · ${incumbentLabel()}` : ''}. Changing this view does not change the recorded selection.`;
    renderSummary(row);
    renderSkills(row);
    renderDetection(row);
    renderDuplicate(row);
    renderCandidates();
    renderChart();
  }

  function renderSummary(row) {
    const values = [
      ['Overall RMSE', row.metrics.overall?.rmse, 'km/s', `${integer(row.metrics.overall?.n)} finite validation-window rows`, 1],
      ['Observed ≥500 RMSE', fastError(row, 500).rmse, 'km/s', `${integer(fastError(row, 500).n)} observed fast-wind rows · primary`, 1],
      ['Observed ≥600 RMSE', fastError(row, 600).rmse, 'km/s', `${integer(fastError(row, 600).n)} observed fast-wind rows · secondary`, 1],
      ['Skill vs 27-day persistence', skill(row, 'persistence_27day').mse_skill_score, '', `${integer(skill(row, 'persistence_27day').n)} common finite rows · positive is better`, 3]
    ];
    const cards = values.map(([label, value, unit, note, digits]) => {
      const card = make('div', null, 'stat');
      const number = make('div', fmt(value, digits), 'stat-value');
      if (unit) number.append(make('small', unit));
      card.append(make('div', label, 'stat-label'), number, make('div', note, 'stat-note'));
      return card;
    });
    $('summary-stats').replaceChildren(...cards);
  }

  function renderChart() {
    const all = $('chart-scope').value === 'all';
    let rows = state.rows.filter(row => all || row.representative || row.selected || row.incumbent || row.id === state.viewed);
    if (!rows.length) rows = state.rows.slice(0, 8);
    const values = rows.flatMap(row => [row.metrics.overall?.rmse, ...thresholds.map(threshold => fastError(row, threshold).rmse)]).filter(finite);
    const max = values.length ? Math.max(...values) : null;
    const ceiling = finite(max) && max > 0 ? Math.ceil(max / 20) * 20 : 1;
    const chart = $('error-chart');
    const scale = make('div', null, 'chart-scale');
    [0, ceiling / 2, ceiling].forEach(value => scale.append(make('span', finite(max) ? `${value} km/s` : '—')));
    const elements = [scale];
    for (const row of rows) {
      const group = make('div', null, `bar-row${row.id === state.viewed ? ' inspected' : ''}`);
      const label = make('button', row.label, 'bar-name'); label.type = 'button';
      label.append(make('small', row.selected ? 'Recorded selection' : row.incumbent ? incumbentLabel() : row.representative ? 'Family CV representative' : titleCase(row.family)));
      label.addEventListener('click', () => setViewed(row.id));
      const bars = make('div', null, 'bar-values');
      const series = [['all', 'All speeds', row.metrics.overall?.rmse], ['fast500', 'Observed at least 500 km/s', fastError(row, 500).rmse], ['fast600', 'Observed at least 600 km/s', fastError(row, 600).rmse]];
      series.forEach(([type, name, value]) => {
        const line = make('div', null, 'bar-line');
        line.setAttribute('aria-label', `${row.label}, ${name}: ${finite(value) ? `${fmt(value)} km/s` : 'not reported or undefined'}`);
        const track = make('div', null, 'bar-track'); track.setAttribute('aria-hidden', 'true');
        if (finite(value)) {const fill = make('div', null, `bar-fill ${type}`); fill.style.width = `${Math.max(0, Math.min(100, value / ceiling * 100))}%`; track.append(fill);}
        const number = make('span', fmt(value), 'bar-number'); number.setAttribute('aria-hidden', 'true');
        line.append(track, number); bars.append(line);
      });
      group.append(label, bars); elements.push(group);
    }
    chart.replaceChildren(...elements);
    $('chart-note').textContent = `${rows.length} of ${state.rows.length} candidates shown. ${all ? 'Every recorded candidate is included.' : 'Family representatives, the incumbent, the recorded selection, and the inspected candidate are shown; choose “Every candidate” for the full chart.'} All bars use the same scale. Click a model label to inspect its detailed scores.`;
  }

  function renderSkills(row) {
    const rows = [];
    const names = Object.keys(row.metrics.persistence_skill || row.metrics.skill || {});
    const expected = ['latest_persistence', 'persistence_27day'];
    const otherNames = names.filter(name => !Object.values(referenceAliases).flat().includes(name));
    for (const name of [...expected, ...otherNames]) {
      for (const scope of ['all', 'ge_500', 'ge_600']) {
        const result = skill(row, name, scope);
        const tr = make('tr');
        tr.append(make('th', referenceLabel(name)), make('td', scope === 'all' ? 'All wind speeds' : `Observed ≥${scope.slice(3)} km/s`), countCell(result.n), metricCell(result.model?.rmse), metricCell(result.reference?.rmse), metricCell(result.mse_skill_score, 3, 'positive'), metricCell(result.delta_rmse, 1, 'negative'));
        tr.firstChild.setAttribute('scope', 'row'); rows.push(tr);
      }
    }
    $('skill-body').replaceChildren(...rows);
    $('skill-candidate').textContent = `${row.label}. Candidate and reference use the same finite cases in each row.`;
  }

  function renderDetection(row) {
    const rows = thresholds.map(threshold => {
      const result = fast(row, threshold).detection || {};
      const tr = make('tr');
      const title = make('th', `≥${threshold} km/s${threshold === 500 ? ' · primary' : ' · secondary'}`); title.scope = 'row'; tr.append(title);
      ['n', 'tp', 'fp', 'fn', 'tn'].forEach(key => tr.append(countCell(result[key])));
      ['pod', 'far', 'fpr', 'csi', 'tss', 'hss', 'frequency_bias'].forEach(key => tr.append(metricCell(result[key], 3)));
      return tr;
    });
    $('detection-body').replaceChildren(...rows);
  }

  function renderDuplicate(row) {
    const duplicate = row.report.unique_issue_target_day;
    const metrics = duplicate ? getMetrics(duplicate) : null;
    const detail = state.data.secondary_deduplication || {};
    $('duplicate-note').hidden = !metrics;
    if (metrics) $('duplicate-note').textContent = `Duplicate sensitivity for this candidate: ${integer(detail.n_input || row.metrics.overall?.n)} original validation-window rows become ${integer(detail.n_unique_pairs || metrics.overall?.n)} unique issue–target pairs. Forecasts for identical pairs are averaged with fixed equal weights. Unique-pair RMSE: ${fmt(metrics.overall?.rmse)} km/s; observed ≥500 RMSE: ${fmt(metrics.fast_wind?.ge_500?.continuous_observed_fast?.rmse)} km/s. This secondary view does not alter the recorded selection.`;
  }

  function renderCandidates() {
    const query = $('candidate-search').value.trim().toLowerCase();
    const input = $('input-filter').value;
    const order = $('order-select').value;
    const rows = state.rows.filter(row => (!query || `${row.id} ${row.label} ${row.family}`.toLowerCase().includes(query)) && (input === 'all' || (input === 'eve') === row.usesEve));
    if (order === 'name') rows.sort((a, b) => a.label.localeCompare(b.label));
    if (order === 'rmse' || order === 'fast500') {
      const score = row => order === 'rmse' ? row.metrics.overall?.rmse : fastError(row, 500).rmse;
      rows.sort((a, b) => (finite(score(a)) ? score(a) : Infinity) - (finite(score(b)) ? score(b) : Infinity) || a.index - b.index);
    }
    const body = rows.map(row => {
      const tr = make('tr', null, row.id === state.viewed ? 'inspected-row' : '');
      const identity = make('td');
      const link = make('button', row.label, 'candidate-name'); link.type = 'button'; link.addEventListener('click', () => setViewed(row.id));
      identity.append(link, make('small', row.id, 'candidate-id'));
      const inputs = make('td'); inputs.append(make('span', row.usesEve ? 'Wind + EVE' : 'Wind only', `tag${row.usesEve ? ' solar' : ''}`));
      tr.append(identity, inputs, metricCell(row.metrics.overall?.rmse), metricCell(fastError(row, 500).rmse), metricCell(fastError(row, 600).rmse), metricCell(skill(row, 'latest_persistence').mse_skill_score, 3, 'positive'), metricCell(skill(row, 'persistence_27day').mse_skill_score, 3, 'positive'));
      const outcome = make('td');
      const labels = [];
      if (row.selected) labels.push(['Recorded selection', 'selected']);
      if (row.incumbent && !row.selected) labels.push([incumbentLabel(), '']);
      if (row.eligibility && !row.selected) labels.push([row.eligibility.eligible ? 'Selection guards passed' : 'Selection guards not met', '']);
      if (row.representative && !labels.length) labels.push(['Family representative', '']);
      if (!labels.length) labels.push(['Evaluated', '']);
      labels.forEach(([label, style]) => outcome.append(make('span', label, `tag ${style}`)));
      tr.append(outcome); return tr;
    });
    if (!body.length) {const tr = make('tr'); const cell = make('td', 'No candidates match these filters.', 'empty-row'); cell.colSpan = 8; tr.append(cell); body.push(tr);}
    $('candidate-body').replaceChildren(...body);
    $('filter-count').textContent = `${rows.length} of ${state.rows.length} shown`;
  }

  function renderProvenance() {
    const data = state.data;
    const fields = Array.isArray(data.eve?.fields) ? data.eve.fields : [];
    if (fields.length) {
      const diodes = fields.filter(field => /^diode/.test(field)).length;
      $('eve-source-summary').textContent = `Public LASP Level 3 tables provide scalar irradiances from ${fields.length - diodes} emission-line channels and ${diodes} diode channels. No SDO images are downloaded or used by this continuation.`;
    }
    if (finite(data.eve?.features)) $('eve-feature-summary').textContent = `The ${data.eve.features} derived inputs contain log irradiance at target-day lags 5–8, four-day means, endpoint trends, and two channel log-ratios. A two-day publication guard is applied relative to the nominal target-minus-three-day issue.`;
    const pairs = [['Experiment', data.experiment], ['Run created (UTC)', data.created_utc], ['Evaluation scope', data.evaluation_scope], ['Scoring unit', data.selection_unit], ['Latest persistence', data.persistence_definition?.latest_persistence], ['27-day persistence', data.persistence_definition?.persistence_27day], ['EVE feature bundle SHA-256', data.eve?.feature_bundle_sha256]];
    if (data.loss_weighting) pairs.push(['Fitting loss multipliers', data.loss_weighting.multipliers?.join(', ')], ['Weight normalization', data.loss_weighting.weight_normalization], ['Weight scope', data.loss_weighting.sample_weight_scope]);
    if (data.h8_reproduction) pairs.push(['H8 reference reproduction', data.h8_reproduction.passed ? 'Passed the registered tolerance check' : 'Not confirmed']);
    for (const [name, value] of Object.entries(data.input_hashes || {})) pairs.push([`${name} SHA-256`, value]);
    const nodes = [];
    pairs.filter(([, value]) => value !== undefined && value !== null).forEach(([label, value]) => {nodes.push(make('dt', label)); const dd = make('dd'); dd.append(make(/sha-256/i.test(label) ? 'code' : 'span', value)); nodes.push(dd);});
    $('provenance-list').replaceChildren(...nodes);
    const sources = make('div', null, 'source-links');
    for (const source of data.eve?.manifest?.sources || []) {
      try {const url = new URL(source.url); if (url.protocol !== 'https:') continue; const link = make('a', `${source.table || 'EVE source table'} ↗`); link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; sources.append(link);} catch { /* Invalid metadata links stay unrendered. */ }
    }
    $('source-links').replaceChildren(sources);
    $('provenance-detail').hidden = !nodes.length;
  }

  function renderComponentAblations() {
    const entries = Object.entries(state.data.feature_ablations_unweighted || {});
    $('component-ablations').hidden = !entries.length;
    if (!entries.length) return;
    const items = entries.map(([group, value]) => ({group, value, row: state.rows.find(row => row.id === value.candidate)}));
    const maximum = Math.max(1, ...items.flatMap(({row}) => row ? [row.metrics.overall?.rmse, ...thresholds.map(t => fastError(row, t).rmse)] : []).filter(finite));
    const ceiling = Math.ceil(maximum / 20) * 20;
    const chart = [];
    const table = [];
    const scale = make('div', null, 'chart-scale'); [0, ceiling / 2, ceiling].forEach(value => scale.append(make('span', `${value} km/s`))); chart.push(scale);
    for (const {group, value, row} of items) {
      const label = groups[group]?.[0] || titleCase(group);
      const count = groups[group]?.[1];
      const delta = value.delta_from_unweighted_wind_only_km_s || {};
      const tr = make('tr'); const heading = make('th', label); heading.scope = 'row';
      tr.append(heading, countCell(count), metricCell(row?.metrics.overall?.rmse), metricCell(row ? fastError(row, 500).rmse : null), metricCell(row ? fastError(row, 600).rmse : null), metricCell(delta.all_rmse, 1, 'negative'), metricCell(delta.fast500_rmse, 1, 'negative'), metricCell(delta.fast600_rmse, 1, 'negative')); table.push(tr);
      const groupRow = make('div', null, 'bar-row'); const name = make('button', label, 'bar-name'); name.type = 'button'; name.append(make('small', `+${count ?? '—'} EVE inputs · unweighted`)); if (row) name.addEventListener('click', () => setViewed(row.id));
      const bars = make('div', null, 'bar-values');
      [['all', 'All speeds', row?.metrics.overall?.rmse], ['fast500', 'Observed ≥500', row ? fastError(row, 500).rmse : null], ['fast600', 'Observed ≥600', row ? fastError(row, 600).rmse : null]].forEach(([type, scope, score]) => {
        const line = make('div', null, 'bar-line'); line.setAttribute('aria-label', `${label}, ${scope}: ${finite(score) ? `${fmt(score)} km/s` : 'undefined'}`);
        const track = make('div', null, 'bar-track'); track.setAttribute('aria-hidden', 'true'); if (finite(score)) {const fill = make('div', null, `bar-fill ${type}`); fill.style.width = `${Math.max(0, score / ceiling * 100)}%`; track.append(fill);} line.append(track, make('span', fmt(score), 'bar-number')); bars.append(line);
      });
      groupRow.append(name, bars); chart.push(groupRow);
    }
    $('component-chart').replaceChildren(...chart); $('component-body').replaceChildren(...table);
  }

  function renderExperimentContext() {
    const h9 = state.experiment === 'H9';
    const config = experiments[state.experiment];
    $('experiment-select').value = state.experiment;
    $('experiment-eyebrow').textContent = `${state.experiment} · Reproducible model development`;
    $('page-title').textContent = config.title;
    document.title = `WindNet study · ${state.experiment} ${h9 ? 'Fast-wind loss' : 'Solar features'}`;
    $('experiment-lead').textContent = h9 ? 'Continue from H8’s retained model: keep the learner pair fixed, isolate ESP and emission-line components, and test whether placing more training weight on fast wind changes the trade-off.' : 'A continuation of the low-cost WindNet study: add scalar SDO/EVE irradiance features, measure what each addition changes, and check performance when solar wind is fast.';
    $('candidate-intro').textContent = h9 ? 'Every new H9 forecast and the exact H8 reference remain visible. Compare component groups, loss multipliers, and fixed weighted/unweighted forecast mixtures. Evaluation scores are always unweighted.' : 'Compare wind-only models with the corresponding solar-feature models, recurrence with multiple rotations, and individual learners with blends. All reported candidates remain visible.';
    $('selection-protocol').textContent = `A ${h9 ? 'new candidate' : 'solar-feature candidate'} must keep overall RMSE within 1% of ${h9 ? 'the exact H8 reference' : 'the wind-only incumbent'} and reduce observed ≥500 km/s RMSE by at least 2%. Eligible candidates minimize an equally weighted normalized overall/fast-wind MSE objective. If none qualify, ${h9 ? 'the exact H8 reference' : 'the wind-only incumbent'} is retained. The ≥600 km/s threshold is secondary and does not select the model.`;
    $('loss-protocol').hidden = !h9;
    $('loss-protocol').textContent = 'H9 fixes every learner pair at 25% ridge (α=1000) + 75% trees (7 leaves, minimum 30 samples per leaf). Observed ≥500 km/s fitting rows receive weight 1, 2, or 4; other rows receive 1. Weights act only on the learner loss and are not normalized, so they also change the effective regularization scale. Preprocessing and evaluation remain unweighted. Forecast mixtures use 25%, 50%, or 75% of a weighted pair with the remaining fraction from its unweighted counterpart.';
    document.querySelectorAll('a[href*="fixrelay.vercel.app/experiments/"]').forEach(link => {link.href = config.url; if (link.closest('.sidebar-bottom')) link.textContent = `${state.experiment} experiment on FixRelay ↗`;});
  }

  function renderLoaded() {
    const data = state.data;
    renderExperimentContext();
    const selected = state.rows.find(row => row.selected) || state.rows.find(row => row.id === data.incumbent) || state.rows[0];
    state.selected = selected.id;
    if (typeof data.improvement_retained === 'boolean') {
      $('decision-title').textContent = data.improvement_retained ? 'A new candidate passed the prescribed development guards.' : 'The exact H8 reference was retained.';
      $('decision-detail').textContent = `${selected.label}. ${data.improvement_retained ? 'The recorded selection met both development guards against H8.' : 'No new candidate met both development guards against H8.'} This uses selection CV and does not establish a held-out improvement.`;
    } else if (typeof data.eve_retained === 'boolean') {
      $('decision-title').textContent = data.eve_retained ? 'An EVE addition passed the prescribed development guards.' : 'The wind-only incumbent was retained.';
      $('decision-detail').textContent = `${selected.label}. ${data.eve_retained ? 'The recorded selection met the overall-error and primary fast-wind conditions.' : 'No EVE candidate met both the overall-error and primary fast-wind conditions.'} This decision uses selection CV and does not establish a held-out improvement.`;
    } else {
      $('decision-title').textContent = 'Development results are available.';
      $('decision-detail').textContent = 'Inspect the recorded candidates below. A selection decision was not supplied in this artifact.';
    }
    const options = state.rows.map(row => {const option = make('option', `${row.label}${row.selected ? ' [recorded selection]' : ''}`); option.value = row.id; return option;});
    $('model-select').replaceChildren(...options);
    $('candidate-count').textContent = finite(data.new_candidate_count) ? `${data.new_candidate_count} new + ${state.rows.length - data.new_candidate_count} reference` : `${state.rows.length} candidates`;
    $('run-stamp').textContent = `${data.experiment || 'Continuation'}${data.created_utc ? ` · ${String(data.created_utc).slice(0, 10)}` : ''} · Folds 1–4 selection CV`;
    $('results-overview').hidden = false;
    document.querySelectorAll('.results-section').forEach(section => {section.hidden = false;});
    $('load-status').hidden = true;
    renderProvenance();
    renderComponentAblations();
    setViewed(state.rows.some(row => row.id === state.viewed) ? state.viewed : selected.id);
  }

  async function load(requested = state.requested) {
    if (typeof requested !== 'string') requested = state.requested;
    state.requested = requested;
    const token = ++state.loadToken;
    const status = $('load-status');
    status.hidden = false; status.classList.remove('error');
    status.textContent = 'Loading the published continuation results…';
    $('reload-results').disabled = true;
    try {
      let experiment = requested === 'auto' ? 'H9' : requested;
      let response = await fetch(experiments[experiment].file, {cache: 'no-store'});
      let pending = false;
      if (!response.ok && response.status === 404 && experiment === 'H9') {pending = true; experiment = 'H8'; response = await fetch(experiments.H8.file, {cache: 'no-store'});}
      if (!response.ok) throw new Error(response.status === 404 ? 'The continuation results have not been published to this page yet.' : `The results file could not be loaded (HTTP ${response.status}).`);
      const data = await response.json();
      if (token !== state.loadToken) return;
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('The results artifact has an unexpected format.');
      const rows = normalize(data);
      if (!rows.length) throw new Error('The continuation artifact does not contain any completed candidate reports yet.');
      state.data = data; state.rows = rows; state.experiment = experiment;
      $('experiment-notice').hidden = !pending;
      $('experiment-notice').textContent = 'H9 has not published completed results yet. H8 is shown below. Select H9 or reload to check again; no H9 metrics are implied.';
      renderLoaded();
    } catch (error) {
      if (token !== state.loadToken) return;
      if (state.experiment) $('experiment-select').value = state.experiment;
      status.classList.add('error');
      status.textContent = `${error.message || 'The results could not be loaded.'} ${state.rows.length ? 'The previously loaded results remain visible below.' : 'No numerical results are shown. Use “Reload results” after the artifact is available; the protocol and public experiment links remain accessible below.'}`;
    } finally { if (token === state.loadToken) $('reload-results').disabled = false; }
  }

  $('model-select').addEventListener('change', event => setViewed(event.target.value));
  $('reset-selected').addEventListener('click', () => setViewed(state.selected));
  $('candidate-search').addEventListener('input', renderCandidates);
  $('input-filter').addEventListener('change', renderCandidates);
  $('order-select').addEventListener('change', renderCandidates);
  $('chart-scope').addEventListener('change', renderChart);
  $('reload-results').addEventListener('click', load);
  $('experiment-select').addEventListener('change', event => load(event.target.value));
  const navLinks = [...document.querySelectorAll('nav a')];
  const activeNav = () => navLinks.forEach(link => link.classList.toggle('active', link.hash === (location.hash || '#overview')));
  window.addEventListener('hashchange', activeNav); activeNav();
  load();
})();
