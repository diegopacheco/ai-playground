import { Engine, valueToString } from './engine.js';

const COLS = 12;
const ROWS = 24;

let measureCtx;
function textWidth(text) {
  if (!measureCtx) {
    measureCtx = document.createElement('canvas').getContext('2d');
    measureCtx.font = '13px "SF Mono", Menlo, Consolas, monospace';
  }
  return measureCtx.measureText(text).width;
}

async function callAI(prompt) {
  const res = await fetch('/api/ai', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.error || `ai ${res.status}`);
  }
  return (await res.json()).text;
}

const engine = new Engine(callAI);
let selected = 'A1';

function colLetter(n) {
  return String.fromCharCode(64 + n);
}

function cellEl(addr) {
  return document.getElementById('cell-' + addr);
}

function renderCell(addr) {
  const td = cellEl(addr);
  if (!td || td.querySelector('input')) return;
  const cell = engine.cells.get(addr);
  td.classList.remove('computing', 'error');
  td.removeAttribute('title');
  if (!cell) { td.textContent = ''; return; }
  if (cell.state === 'computing') { td.classList.add('computing'); td.innerHTML = '<span class="spark">✨</span>'; return; }
  if (cell.state === 'error') {
    td.classList.add('error');
    td.textContent = cell.error;
    if (cell.detail) td.title = cell.detail;
    return;
  }
  td.textContent = valueToString(cell.value);
}

function select(addr) {
  selected = addr;
  document.querySelectorAll('td.selected').forEach((td) => td.classList.remove('selected'));
  const td = cellEl(addr);
  if (td) td.classList.add('selected');
  const cell = engine.cells.get(addr);
  document.getElementById('addr').textContent = addr;
  document.getElementById('formula').value = cell ? cell.raw ?? '' : '';
}

function move(dc, dr) {
  const m = selected.match(/^([A-Z]+)([0-9]+)$/);
  const c = Math.max(1, Math.min(COLS, m[1].charCodeAt(0) - 64 + dc));
  const r = Math.max(1, Math.min(ROWS, Number(m[2]) + dr));
  select(colLetter(c) + r);
}

function saveLocal() {
  localStorage.setItem('sheet', JSON.stringify(serialize()));
}

function commitCell(addr, raw) {
  engine.setRaw(addr, raw);
  saveLocal();
  if (addr === selected) {
    const cell = engine.cells.get(addr);
    document.getElementById('formula').value = cell ? cell.raw ?? '' : '';
  }
}

function editInline(addr, initial) {
  const td = cellEl(addr);
  if (!td || td.querySelector('input')) return;
  const cell = engine.cells.get(addr);
  const original = cell ? cell.raw ?? '' : '';
  const input = document.createElement('input');
  input.className = 'cell-input';
  input.value = initial !== undefined ? initial : original;
  td.textContent = '';
  td.classList.add('editing');
  td.appendChild(input);
  const resize = () => {
    input.style.width = Math.max(td.clientWidth, Math.ceil(textWidth(input.value)) + 22) + 'px';
  };
  input.addEventListener('input', resize);
  resize();
  input.focus();
  if (initial === undefined) input.select();
  let done = false;
  const finish = (value) => {
    if (done) return;
    done = true;
    input.remove();
    td.classList.remove('editing');
    if (value !== null) commitCell(addr, value);
    renderCell(addr);
  };
  input.addEventListener('blur', () => finish(input.value));
  input.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); finish(input.value); move(0, 1); }
    else if (e.key === 'Tab') { e.preventDefault(); finish(input.value); move(1, 0); }
    else if (e.key === 'Escape') { e.preventDefault(); finish(null); }
  });
}

function buildGrid() {
  const table = document.getElementById('grid');
  const header = document.createElement('tr');
  header.appendChild(document.createElement('th'));
  for (let c = 1; c <= COLS; c++) {
    const th = document.createElement('th');
    th.textContent = colLetter(c);
    header.appendChild(th);
  }
  table.appendChild(header);
  for (let r = 1; r <= ROWS; r++) {
    const tr = document.createElement('tr');
    const rowHead = document.createElement('th');
    rowHead.textContent = r;
    tr.appendChild(rowHead);
    for (let c = 1; c <= COLS; c++) {
      const addr = colLetter(c) + r;
      const td = document.createElement('td');
      td.id = 'cell-' + addr;
      td.addEventListener('click', () => select(addr));
      td.addEventListener('dblclick', () => editInline(addr));
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
}

function serialize() {
  const sheet = {};
  for (const [addr, cell] of engine.cells) {
    if (cell.raw !== undefined && cell.raw !== '') sheet[addr] = cell.raw;
  }
  return sheet;
}

function renderAll() {
  for (const addr of engine.cells.keys()) renderCell(addr);
}

async function saveServer() {
  await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sheet: serialize() }),
  });
}

async function loadServer() {
  const data = await (await fetch('/api/load')).json();
  await engine.loadAll(data.sheet || {});
  renderAll();
  select(selected);
  saveLocal();
}

window.addEventListener('DOMContentLoaded', () => {
  buildGrid();
  engine.onUpdate = renderCell;

  const saved = localStorage.getItem('sheet');
  if (saved) {
    try { engine.loadAll(JSON.parse(saved)); } catch {}
  }
  select('A1');

  const formula = document.getElementById('formula');
  formula.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commitCell(selected, formula.value); move(0, 1); }
  });

  document.getElementById('save').addEventListener('click', saveServer);
  document.getElementById('load').addEventListener('click', loadServer);

  const helpOverlay = document.getElementById('help-overlay');
  const closeHelp = () => helpOverlay.classList.add('hidden');
  document.getElementById('help').addEventListener('click', () => helpOverlay.classList.remove('hidden'));
  document.getElementById('help-close').addEventListener('click', closeHelp);
  helpOverlay.addEventListener('click', (e) => { if (e.target === helpOverlay) closeHelp(); });

  document.addEventListener('keydown', (e) => {
    if (!helpOverlay.classList.contains('hidden')) {
      if (e.key === 'Escape') closeHelp();
      return;
    }
    const active = document.activeElement;
    if (active && active.tagName === 'INPUT') return;
    if (e.key === 'Enter') { e.preventDefault(); editInline(selected); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); move(0, 1); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); move(0, -1); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); move(1, 0); return; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); move(-1, 0); return; }
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); commitCell(selected, ''); renderCell(selected); return; }
    if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) { e.preventDefault(); editInline(selected, e.key); }
  });
});
