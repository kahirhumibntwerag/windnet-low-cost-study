/* This page reads only the explicit training-only continuation artifact. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits = 1) => finite(value) ? value.toFixed(digits) : '—';
  const integer = value => finite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 0}) : '—';
  const thresholds = [500, 600];
  const state = {data: null, rows: [], viewed: null, selected: null};
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
        eligibility: data.eve_eligibility?.[id] || report.eligibility || null};
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
    $('view-note').textContent = `Inspecting: ${row.label}${row.selected ? ' · recorded selection' : row.incumbent ? ' · wind-only incumbent' : ''}. Changing this view does not change the recorded selection.`;
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
      label.append(make('small', row.selected ? 'Recorded selection' : row.incumbent ? 'Wind-only incumbent' : row.representative ? 'Family CV representative' : titleCase(row.family)));
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
      if (row.incumbent && !row.selected) labels.push(['Wind-only incumbent', '']);
      if (row.eligibility && !row.selected) labels.push([row.eligibility.eligible ? 'EVE guards passed' : 'EVE guards not met', '']);
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

  function renderLoaded() {
    const data = state.data;
    const selected = state.rows.find(row => row.selected) || state.rows.find(row => row.id === data.incumbent) || state.rows[0];
    state.selected = selected.id;
    if (typeof data.eve_retained === 'boolean') {
      $('decision-title').textContent = data.eve_retained ? 'An EVE addition passed the prescribed development guards.' : 'The wind-only incumbent was retained.';
      $('decision-detail').textContent = `${selected.label}. ${data.eve_retained ? 'The recorded selection met the overall-error and primary fast-wind conditions.' : 'No EVE candidate met both the overall-error and primary fast-wind conditions.'} This decision uses selection CV and does not establish a held-out improvement.`;
    } else {
      $('decision-title').textContent = 'Development results are available.';
      $('decision-detail').textContent = 'Inspect the recorded candidates below. A selection decision was not supplied in this artifact.';
    }
    const options = state.rows.map(row => {const option = make('option', `${row.label}${row.selected ? ' [recorded selection]' : ''}`); option.value = row.id; return option;});
    $('model-select').replaceChildren(...options);
    $('candidate-count').textContent = `${state.rows.length} candidates`;
    $('run-stamp').textContent = `${data.experiment || 'Continuation'}${data.created_utc ? ` · ${String(data.created_utc).slice(0, 10)}` : ''} · Folds 1–4 selection CV`;
    $('results-overview').hidden = false;
    document.querySelectorAll('.results-section').forEach(section => {section.hidden = false;});
    $('load-status').hidden = true;
    renderProvenance();
    setViewed(state.rows.some(row => row.id === state.viewed) ? state.viewed : selected.id);
  }

  async function load() {
    const status = $('load-status');
    status.hidden = false; status.classList.remove('error');
    status.textContent = 'Loading the published continuation results…';
    $('reload-results').disabled = true;
    try {
      const response = await fetch('./continuation.json', {cache: 'no-store'});
      if (!response.ok) throw new Error(response.status === 404 ? 'The continuation results have not been published to this page yet.' : `The results file could not be loaded (HTTP ${response.status}).`);
      const data = await response.json();
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('The results artifact has an unexpected format.');
      const rows = normalize(data);
      if (!rows.length) throw new Error('The continuation artifact does not contain any completed candidate reports yet.');
      state.data = data; state.rows = rows; renderLoaded();
    } catch (error) {
      status.classList.add('error');
      status.textContent = `${error.message || 'The results could not be loaded.'} ${state.rows.length ? 'The previously loaded results remain visible below.' : 'No numerical results are shown. Use “Reload results” after the artifact is available; the protocol and public experiment links remain accessible below.'}`;
    } finally { $('reload-results').disabled = false; }
  }

  $('model-select').addEventListener('change', event => setViewed(event.target.value));
  $('reset-selected').addEventListener('click', () => setViewed(state.selected));
  $('candidate-search').addEventListener('input', renderCandidates);
  $('input-filter').addEventListener('change', renderCandidates);
  $('order-select').addEventListener('change', renderCandidates);
  $('chart-scope').addEventListener('change', renderChart);
  $('reload-results').addEventListener('click', load);
  const navLinks = [...document.querySelectorAll('nav a')];
  const activeNav = () => navLinks.forEach(link => link.classList.toggle('active', link.hash === (location.hash || '#overview')));
  window.addEventListener('hashchange', activeNav); activeNav();
  load();
})();
