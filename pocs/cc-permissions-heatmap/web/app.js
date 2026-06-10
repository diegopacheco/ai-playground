const HEAT = [
  [255, 247, 224],
  [255, 213, 128],
  [247, 148, 56],
  [226, 75, 46],
  [168, 33, 28]
];

function heat(ratio) {
  const r = Math.max(0, Math.min(1, ratio));
  const seg = r * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(seg));
  const t = seg - i;
  const c = HEAT[i].map((v, k) => Math.round(v + (HEAT[i + 1][k] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function fmt(n) { return n.toLocaleString(); }
function prettyTool(name) {
  if (name.startsWith('mcp:')) return name.replace('mcp:plugin_context-mode_context-mode', 'context-mode').replace('mcp:', '') + ' (MCP)';
  return name;
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

function renderCards(m) {
  const cards = [
    { k: 'Prompts triggered', v: fmt(m.totalPrompts), s: 'modeled from your allow-list', hot: true },
    { k: 'Tool calls', v: fmt(m.totalToolCalls), s: `${fmt(m.totalNoPrompt)} never prompt` },
    { k: 'Auto-allowed', v: fmt(m.totalAutoAllowed), s: 'matched an allow-rule' },
    { k: 'Recorded denials', v: fmt(m.recordedDenials), s: 'rejected / interrupted' },
    { k: 'Sessions scanned', v: fmt(m.sessions), s: `${fmt(m.projects)} projects` }
  ];
  const root = document.getElementById('cards');
  root.innerHTML = '';
  for (const c of cards) {
    const card = el('div', 'card' + (c.hot ? ' hot' : ''));
    card.append(el('div', 'k', c.k), el('div', 'v', c.v), el('div', 's', c.s));
    root.append(card);
  }
}

function renderHeatBars(rootId, rows, labelFn, max) {
  const root = document.getElementById(rootId);
  root.innerHTML = '';
  if (!rows.length) { root.append(el('div', 'empty', 'Nothing to show.')); return; }
  for (const row of rows) {
    const bar = el('div', 'hbar');
    bar.append(el('div', 'lab', labelFn(row)));
    const track = el('div', 'track');
    const fill = el('div', 'fill');
    const ratio = row.prompts / max;
    fill.style.width = Math.max(2, ratio * 100) + '%';
    fill.style.background = heat(ratio);
    track.append(fill);
    bar.append(track);
    const total = row.total != null && row.total !== row.prompts ? ` <small>/ ${fmt(row.total)}</small>` : '';
    bar.append(el('div', 'num', fmt(row.prompts) + total));
    root.append(bar);
  }
}

function renderDenials(list) {
  const root = document.getElementById('denials');
  root.innerHTML = '';
  if (!list.length) { root.append(el('div', 'empty', 'No rejections or interruptions recorded. Smooth sailing.')); return; }
  for (const d of list) {
    const row = el('div', 'denial');
    row.append(el('span', 'tag ' + d.reason, d.reason));
    const cmd = d.cmd ? ` <code>${d.cmd}</code>` : '';
    row.append(el('span', 'name', d.tool + cmd));
    row.append(el('span', 'meta', `${d.project}<br>${d.date}`));
    root.append(row);
  }
}

function renderCalendar(days) {
  const root = document.getElementById('calendar');
  root.innerHTML = '';
  if (!days.length) { root.append(el('div', 'empty', 'No dated activity.')); return; }
  const max = Math.max(...days.map(d => d.prompts));
  const everyN = days.length > 18 ? Math.ceil(days.length / 12) : 1;
  days.forEach((d, idx) => {
    const cell = el('div', 'cal-cell');
    const ratio = d.prompts / max;
    cell.style.height = (16 + ratio * 90) + 'px';
    cell.style.background = heat(ratio);
    cell.setAttribute('data-tip', `${d.date} · ${d.prompts} prompts`);
    if (idx % everyN === 0) cell.append(el('span', 'cap', d.date.slice(5)));
    root.append(cell);
  });
}

function renderMatrix(matrix) {
  const root = document.getElementById('matrix');
  root.innerHTML = '';
  if (!matrix.tools.length) { root.append(el('div', 'empty', 'Not enough data.')); return; }
  let max = 0;
  for (const r of matrix.cells) for (const v of r) max = Math.max(max, v);
  const table = el('table', 'matrix');
  const thead = el('tr');
  thead.append(el('th', '', ''));
  for (const p of matrix.projects) thead.append(el('th', 'col', p));
  table.append(thead);
  matrix.tools.forEach((tool, i) => {
    const tr = el('tr');
    tr.append(el('th', 'row', prettyTool(tool)));
    matrix.cells[i].forEach(v => {
      const td = el('td', v ? '' : 'zero', v || '·');
      if (v) td.style.background = heat(v / max);
      td.setAttribute('title', `${v} prompts`);
      tr.append(td);
    });
    table.append(tr);
  });
  root.append(table);
  const legend = el('div', 'legend');
  legend.append(el('span', '', 'fewer'));
  const scale = el('div', 'scale');
  for (let i = 0; i <= 4; i++) { const sw = el('span', 'sw'); sw.style.background = heat(i / 4); scale.append(sw); }
  legend.append(scale, el('span', '', 'more'));
  root.append(legend);
}

function renderMethodology(m) {
  const root = document.getElementById('methodology');
  const rules = (m.allowRules || []).map(r => `<span class="rule">${r}</span>`).join('') || '<span class="rule">none</span>';
  const modes = Object.entries(m.modeCounts).filter(([, v]) => v > 0).map(([k, v]) => `${k}: ${fmt(v)}`).join(' · ');
  root.innerHTML = `
    <h2>How a "prompt" is counted</h2>
    <p>This reads your transcripts under <code>~/.claude/projects</code> and replays each tool call against your permission rules and the permission mode recorded for that session (${modes}).</p>
    <ul>
      <li>Read-only tools (<code>Read</code>, <code>Glob</code>, <code>Grep</code>, search, task bookkeeping) never prompt.</li>
      <li>File edits (<code>Edit</code>, <code>Write</code>) prompt in <code>default</code> mode but are auto-accepted in <code>auto</code> mode.</li>
      <li><code>Bash</code>, <code>WebFetch</code> and MCP tools prompt unless an allow-rule matches.</li>
    </ul>
    <p>Your active allow-rules:</p>
    <div class="rules">${rules}</div>`;
}

async function main() {
  const data = await (await fetch('data.json?_=' + Date.now())).json();
  const m = data.meta;

  document.getElementById('scaninfo').innerHTML =
    `<b>${fmt(m.filesScanned)}</b> transcripts · <b>${fmt(m.projects)}</b> projects<br>scanned ${new Date(m.scannedAt).toLocaleString()}`;
  document.getElementById('dirpath').textContent = m.claudeDir + '/projects';

  renderCards(m);

  const toolRows = data.groups.filter(g => g.prompts > 0).map(g => ({ label: g.group, prompts: g.prompts, total: g.total }));
  const toolMax = Math.max(...toolRows.map(r => r.prompts), 1);
  renderHeatBars('toolHeat', toolRows, r => prettyTool(r.label), toolMax);

  const bashRows = data.bash.slice(0, 15);
  const bashMax = Math.max(...bashRows.map(r => r.prompts), 1);
  renderHeatBars('bashHeat', bashRows, r => `<code>${r.cmd}</code>`, bashMax);

  renderDenials(data.denials);
  renderCalendar(data.byDay);
  renderMatrix(data.matrix);
  renderMethodology(m);
}

main().catch(e => {
  document.getElementById('app').innerHTML = `<div class="panel empty">Could not load data.json — run <code>node scan.js</code> first.<br>${e}</div>`;
});
