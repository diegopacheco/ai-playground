const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const banner = document.getElementById('banner');

const KEYWORDS = new Set(['public', 'private', 'static', 'final', 'class', 'new', 'return', 'while', 'for', 'if', 'else', 'continue', 'break', 'import', 'record', 'int', 'boolean', 'void']);

const COLORS = {
  bfs: { main: '#0e8f8b', bright: '#14b8b1', settled: '#8ed2ce' },
  dfs: { main: '#e0731d', bright: '#f0862e', settled: '#f3c294' },
  both: { settled: '#bca6d8', bright: '#8f6cc0' },
  wallTop: '#d4cab4', wallSide: '#b3a78c',
  floorA: '#ece4d1', floorB: '#e6ddc9',
  gold: '#e3b023', start: '#3aa655', ink: '#25221a'
};

const state = {
  mode: 'race', size: 25, data: null, running: false,
  idx: { bfs: 0, dfs: 0 }, acc: 0, lastTime: 0,
  cellState: { bfs: new Map(), dfs: new Map() },
  visited: { bfs: new Set(), dfs: new Set() },
  current: { bfs: -1, dfs: -1 },
  pathReveal: [], pathSet: new Set(),
  finished: { bfs: false, dfs: false }, winner: null,
  ripples: [], openCount: 0,
  yawBase: 0.78, dragging: false, dragX: 0
};

const lineEls = { bfs: [], dfs: [] };
const activeLine = { bfs: -1, dfs: -1 };

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightJava(line) {
  let out = '';
  const re = /("(?:[^"\\]|\\.)*")|(\b\d+\b)|(\b[A-Za-z_]\w*\b)|(.)/g;
  let m;
  while ((m = re.exec(line)) !== null) {
    if (m[1]) out += `<span class="tk-str">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="tk-num">${m[2]}</span>`;
    else if (m[3]) {
      const word = m[3];
      const next = line[re.lastIndex];
      if (KEYWORDS.has(word)) out += `<span class="tk-kw">${word}</span>`;
      else if (/^[A-Z]/.test(word)) out += `<span class="tk-ty">${word}</span>`;
      else if (next === '(') out += `<span class="tk-call">${word}</span>`;
      else out += word;
    } else out += esc(m[4]);
  }
  return out;
}

function renderCode(algo, source) {
  const body = document.getElementById(algo === 'bfs' ? 'codeBfs' : 'codeDfs');
  body.innerHTML = '';
  lineEls[algo] = [];
  source.split('\n').forEach((line, i) => {
    const row = document.createElement('div');
    row.className = 'cl';
    row.innerHTML = `<span class="ln">${i + 1}</span><span class="lc">${highlightJava(line) || ' '}</span>`;
    body.appendChild(row);
    lineEls[algo].push(row);
  });
}

function setLine(algo, line) {
  if (activeLine[algo] === line) return;
  const prev = lineEls[algo][activeLine[algo] - 1];
  if (prev) prev.classList.remove(`active-${algo}`);
  activeLine[algo] = line;
  const el = lineEls[algo][line - 1];
  if (!el) return;
  el.classList.add(`active-${algo}`);
  const body = el.parentElement;
  const target = el.offsetTop - body.clientHeight / 2 + 20;
  if (Math.abs(body.scrollTop - target) > 30) body.scrollTop = target;
}

function clearLines() {
  for (const algo of ['bfs', 'dfs']) {
    const prev = lineEls[algo][activeLine[algo] - 1];
    if (prev) prev.classList.remove(`active-${algo}`);
    activeLine[algo] = -1;
  }
}

async function fetchRace(seed) {
  const q = seed != null ? `&seed=${seed}` : '';
  const res = await fetch(`/api/race?size=${state.size}${q}`);
  state.data = await res.json();
  state.openCount = [...state.data.open].filter(c => c === '1').length;
  resetPlayback();
}

function resetPlayback() {
  state.running = false;
  state.idx = { bfs: 0, dfs: 0 };
  state.acc = 0;
  state.cellState.bfs.clear();
  state.cellState.dfs.clear();
  state.visited.bfs.clear();
  state.visited.dfs.clear();
  state.current = { bfs: -1, dfs: -1 };
  state.pathReveal = [];
  state.pathSet.clear();
  state.finished = { bfs: false, dfs: false };
  state.winner = null;
  state.ripples = [];
  banner.className = 'banner hidden';
  clearLines();
  setStatus('bfs', 'idle', '');
  setStatus('dfs', 'idle', '');
  document.getElementById('pathLen').textContent = '—';
  document.getElementById('pathSub').textContent = 'length';
  document.getElementById('runBtn').textContent = 'Run';
  updateStats();
}

function setStatus(algo, cls, text) {
  const el = document.getElementById(algo === 'bfs' ? 'bfsStatus' : 'dfsStatus');
  el.className = `pane-status ${cls}`;
  el.textContent = text || cls;
}

function activeAlgos() {
  return state.mode === 'race' ? ['bfs', 'dfs'] : [state.mode];
}

function consumeStep(algo) {
  const steps = state.data[algo];
  if (state.idx[algo] >= steps.length) return false;
  const [type, cell, line] = steps[state.idx[algo]++];
  setLine(algo, line);
  if (type === 'visit' || type === 'scan') {
    state.visited[algo].add(cell);
    state.cellState[algo].set(cell, performance.now());
    state.current[algo] = cell;
    state.ripples.push({ cell, t: performance.now(), algo });
    if (state.ripples.length > 50) state.ripples.shift();
  } else if (type === 'frontier') {
    if (!state.visited[algo].has(cell)) state.cellState[algo].set(cell, -1);
  } else if (type === 'path') {
    if (!state.finished[algo]) {
      state.finished[algo] = true;
      state.current[algo] = -1;
      const visits = state.visited[algo].size;
      setStatus(algo, 'done', `done · ${visits} visits`);
      if (!state.winner) {
        state.winner = algo;
        announceWinner(algo);
      }
    }
    if (!state.pathSet.has(cell)) {
      state.pathSet.add(cell);
      state.pathReveal.push({ cell, t: performance.now() });
    }
  }
  return true;
}

function announceWinner(algo) {
  const name = algo.toUpperCase();
  const visits = state.visited[algo].size;
  const pathLen = state.data[algo].filter(s => s[0] === 'path').length;
  document.getElementById('pathLen').textContent = pathLen;
  document.getElementById('pathSub').textContent = 'cells start to exit';
  if (state.mode === 'race') {
    const other = algo === 'bfs' ? 'dfs' : 'bfs';
    const otherTotal = state.data[other].filter(s => s[0] === 'scan' || s[0] === 'visit').length;
    banner.textContent = `${name} escapes first · ${visits} visits vs ${otherTotal}`;
  } else {
    banner.textContent = `${name} solved it · ${visits} visits · path ${pathLen}`;
  }
  banner.className = `banner win-${algo}`;
}

function updateStats() {
  document.getElementById('bfsVisits').textContent = state.visited.bfs.size;
  document.getElementById('dfsVisits').textContent = state.visited.dfs.size;
  document.getElementById('bfsBar').style.width = `${(state.visited.bfs.size / Math.max(1, state.openCount)) * 100}%`;
  document.getElementById('dfsBar').style.width = `${(state.visited.dfs.size / Math.max(1, state.openCount)) * 100}%`;
}

function tick(now) {
  if (!state.lastTime) state.lastTime = now;
  const dt = Math.min(0.1, (now - state.lastTime) / 1000);
  state.lastTime = now;
  if (state.running && state.data) {
    const speed = Number(document.getElementById('speed').value);
    state.acc += speed * dt;
    const whole = Math.floor(state.acc);
    state.acc -= whole;
    let alive = false;
    for (const algo of activeAlgos()) {
      for (let i = 0; i < whole; i++) {
        if (!consumeStep(algo)) break;
      }
      if (state.idx[algo] < state.data[algo].length) alive = true;
      else state.current[algo] = -1;
    }
    updateStats();
    if (!alive) {
      state.running = false;
      document.getElementById('runBtn').textContent = 'Replay';
    }
  }
  draw(now);
  requestAnimationFrame(tick);
}

function hexRgb(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

function shade(hex, f) {
  const [r, g, b] = hexRgb(hex);
  return `rgb(${Math.round(r * f)},${Math.round(g * f)},${Math.round(b * f)})`;
}

function mix(hexA, hexB, t) {
  const a = hexRgb(hexA), b = hexRgb(hexB);
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
}

function floorColor(cell, now) {
  const tb = state.cellState.bfs.get(cell);
  const td = state.cellState.dfs.get(cell);
  const vb = state.visited.bfs.has(cell);
  const vd = state.visited.dfs.has(cell);
  if (vb && vd) {
    const t = Math.min(1, (now - Math.max(tb, td)) / 900);
    return mix(COLORS.both.bright, COLORS.both.settled, t);
  }
  if (vb) {
    const t = Math.min(1, (now - tb) / 900);
    return mix(COLORS.bfs.bright, COLORS.bfs.settled, t);
  }
  if (vd) {
    const t = Math.min(1, (now - td) / 900);
    return mix(COLORS.dfs.bright, COLORS.dfs.settled, t);
  }
  const x = cell % state.data.size, y = (cell / state.data.size) | 0;
  return (x + y) % 2 ? COLORS.floorA : COLORS.floorB;
}

function quad(px, py, a, b, c, d, color) {
  ctx.beginPath();
  ctx.moveTo(px[a], py[a]);
  ctx.lineTo(px[b], py[b]);
  ctx.lineTo(px[c], py[c]);
  ctx.lineTo(px[d], py[d]);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function prism(pts, h, zs, topColor, sideColor, sideBright, glow) {
  const top = pts.map(p => [p[0], p[1] - h * zs]);
  const sides = [];
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    sides.push({ i, j, depth: (pts[i][2] + pts[j][2]) / 2 });
  }
  sides.sort((a, b) => a.depth - b.depth);
  for (const s of sides) {
    ctx.beginPath();
    ctx.moveTo(pts[s.i][0], pts[s.i][1]);
    ctx.lineTo(pts[s.j][0], pts[s.j][1]);
    ctx.lineTo(top[s.j][0], top[s.j][1]);
    ctx.lineTo(top[s.i][0], top[s.i][1]);
    ctx.closePath();
    ctx.fillStyle = shade(sideColor, sideBright[s.i]);
    ctx.fill();
  }
  if (glow) {
    ctx.save();
    ctx.shadowColor = glow;
    ctx.shadowBlur = 18;
  }
  ctx.beginPath();
  ctx.moveTo(top[0][0], top[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(top[i][0], top[i][1]);
  ctx.closePath();
  ctx.fillStyle = topColor;
  ctx.fill();
  if (glow) ctx.restore();
}

function easeOutBack(t) {
  const c = 1.70158;
  return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2);
}

function draw(now) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.data) return;

  const G = state.data.size;
  const yaw = state.yawBase + (state.dragging ? 0 : 0.13 * Math.sin(now * 0.00012));
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const span = G * (Math.abs(cosY) + Math.abs(sinY));
  const s = Math.min((w * 0.9) / span, (h * 0.86) / (span * 0.52 + 1.6));
  const zs = s * 0.62;
  const cx = w / 2, cy = h / 2 + G * 0.02 * s;

  const n = G + 1;
  const px = new Float64Array(n * n);
  const py = new Float64Array(n * n);
  const pd = new Float64Array(n * n);
  for (let gy = 0; gy < n; gy++) {
    for (let gx = 0; gx < n; gx++) {
      const wx = gx - G / 2, wy = gy - G / 2;
      const rx = wx * cosY - wy * sinY;
      const ry = wx * sinY + wy * cosY;
      const i = gy * n + gx;
      px[i] = cx + rx * s;
      py[i] = cy + ry * s * 0.52;
      pd[i] = ry;
    }
  }

  ctx.save();
  ctx.translate(0, G * 0.04 * s + 10);
  ctx.beginPath();
  const outer = [0, G, n * n - 1, (n - 1) * n].map(i => [px[i], py[i]]);
  ctx.moveTo(outer[0][0], outer[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(outer[i][0], outer[i][1]);
  ctx.closePath();
  ctx.fillStyle = 'rgba(37,34,26,0.13)';
  ctx.filter = 'blur(8px)';
  ctx.fill();
  ctx.restore();

  const corners = (x, y) => [y * n + x, y * n + x + 1, (y + 1) * n + x + 1, (y + 1) * n + x];

  const open = state.data.open;
  for (let y = 0; y < G; y++) {
    for (let x = 0; x < G; x++) {
      const cell = y * G + x;
      if (open[cell] !== '1') continue;
      const [a, b, c, d] = corners(x, y);
      quad(px, py, a, b, c, d, floorColor(cell, now));
      const fb = state.cellState.bfs.get(cell) === -1 && !state.visited.bfs.has(cell);
      const fd = state.cellState.dfs.get(cell) === -1 && !state.visited.dfs.has(cell);
      if (fb || fd) {
        const mx = (px[a] + px[c]) / 2, my = (py[a] + py[c]) / 2;
        ctx.beginPath();
        ctx.arc(mx, my, s * 0.13, 0, Math.PI * 2);
        ctx.fillStyle = fb && fd ? COLORS.both.bright : fb ? COLORS.bfs.main : COLORS.dfs.main;
        ctx.globalAlpha = 0.8;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
  }

  state.ripples = state.ripples.filter(r => now - r.t < 650);
  for (const r of state.ripples) {
    const t = (now - r.t) / 650;
    const x = r.cell % G, y = (r.cell / G) | 0;
    const [a, , c] = corners(x, y);
    const mx = (px[a] + px[c]) / 2, my = (py[a] + py[c]) / 2;
    ctx.beginPath();
    ctx.ellipse(mx, my, s * (0.2 + t * 1.2), s * 0.52 * (0.2 + t * 1.2), 0, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS[r.algo].main;
    ctx.globalAlpha = 0.5 * (1 - t);
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  const lightAngle = -2.1;
  const sideBright = [];
  for (let i = 0; i < 4; i++) {
    const worldAngle = [-Math.PI / 2, 0, Math.PI / 2, Math.PI][i];
    sideBright.push(0.62 + 0.34 * Math.max(0, Math.cos(worldAngle + yaw - lightAngle)));
  }

  const blocks = [];
  const tilePts = (x, y, inset) => {
    const [a, b, c, d] = corners(x, y);
    const ids = [a, b, c, d];
    if (!inset) return ids.map(i => [px[i], py[i], pd[i]]);
    const mx = ids.reduce((t, i) => t + px[i], 0) / 4;
    const my = ids.reduce((t, i) => t + py[i], 0) / 4;
    const md = ids.reduce((t, i) => t + pd[i], 0) / 4;
    return ids.map(i => [px[i] + (mx - px[i]) * inset, py[i] + (my - py[i]) * inset, pd[i] + (md - pd[i]) * inset]);
  };

  for (let y = 0; y < G; y++) {
    for (let x = 0; x < G; x++) {
      const cell = y * G + x;
      if (open[cell] === '1') continue;
      const pts = tilePts(x, y, 0);
      blocks.push({ depth: (pts[0][2] + pts[2][2]) / 2, pts, h: 0.8, top: COLORS.wallTop, side: COLORS.wallSide, glow: null });
    }
  }

  for (const p of state.pathReveal) {
    const t = Math.min(1, (now - p.t) / 380);
    const hgt = 0.32 * easeOutBack(t);
    const x = p.cell % G, y = (p.cell / G) | 0;
    const pts = tilePts(x, y, 0.18);
    blocks.push({ depth: (pts[0][2] + pts[2][2]) / 2, pts, h: hgt, top: COLORS.gold, side: '#b8860b', glow: t < 1 ? COLORS.gold : null });
  }

  for (const algo of activeAlgos()) {
    const cell = state.current[algo];
    if (cell < 0) continue;
    const x = cell % G, y = (cell / G) | 0;
    const pts = tilePts(x, y, 0.12);
    const hgt = 0.5 + 0.1 * Math.sin(now * 0.012);
    blocks.push({ depth: (pts[0][2] + pts[2][2]) / 2, pts, h: hgt, top: COLORS[algo].bright, side: COLORS[algo].main, glow: COLORS[algo].main });
  }

  const markers = [
    { cell: state.data.start, color: COLORS.start, label: 'START' },
    { cell: state.data.goal, color: COLORS.ink, label: 'EXIT', goldTop: true }
  ];
  for (const mk of markers) {
    const x = mk.cell % G, y = (mk.cell / G) | 0;
    const pts = tilePts(x, y, 0.42);
    blocks.push({ depth: (pts[0][2] + pts[2][2]) / 2, pts, h: 1.05, top: mk.goldTop ? COLORS.gold : mk.color, side: mk.color, glow: null, label: mk.label });
  }

  blocks.sort((a, b) => a.depth - b.depth);
  for (const blk of blocks) {
    prism(blk.pts, blk.h, zs, blk.top, blk.side, sideBright, blk.glow);
    if (blk.label) {
      const mx = (blk.pts[0][0] + blk.pts[2][0]) / 2;
      const my = (blk.pts[0][1] + blk.pts[2][1]) / 2 - blk.h * zs - 10;
      ctx.font = `700 ${Math.max(10, s * 0.42)}px "Schibsted Grotesk", sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.ink;
      ctx.fillText(blk.label, mx, my);
    }
  }

  const gx = state.data.goal % G, gy = (state.data.goal / G) | 0;
  const [ga, , gc] = corners(gx, gy);
  const gmx = (px[ga] + px[gc]) / 2, gmy = (py[ga] + py[gc]) / 2;
  const pulse = (now % 1400) / 1400;
  ctx.beginPath();
  ctx.ellipse(gmx, gmy, s * (0.5 + pulse), s * 0.52 * (0.5 + pulse), 0, 0, Math.PI * 2);
  ctx.strokeStyle = COLORS.gold;
  ctx.globalAlpha = 0.6 * (1 - pulse);
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#algoSeg button').forEach(b => b.classList.toggle('active', b.dataset.algo === mode));
  document.getElementById('paneBfs').classList.toggle('hidden', mode === 'dfs');
  document.getElementById('paneDfs').classList.toggle('hidden', mode === 'bfs');
  document.getElementById('statBfs').classList.toggle('dim', mode === 'dfs');
  document.getElementById('statDfs').classList.toggle('dim', mode === 'bfs');
  resetPlayback();
}

document.getElementById('algoSeg').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (btn) setMode(btn.dataset.algo);
});

document.getElementById('sizeSeg').addEventListener('click', async e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  state.size = Number(btn.dataset.size);
  document.querySelectorAll('#sizeSeg button').forEach(b => b.classList.toggle('active', b === btn));
  await fetchRace();
});

document.getElementById('newMaze').addEventListener('click', () => fetchRace());

document.getElementById('runBtn').addEventListener('click', e => {
  if (!state.data) return;
  const allDone = activeAlgos().every(a => state.idx[a] >= state.data[a].length);
  if (allDone && !state.running) {
    const keep = state.data;
    resetPlayback();
    state.data = keep;
  }
  state.running = !state.running;
  e.target.textContent = state.running ? 'Pause' : 'Resume';
  if (state.running) {
    for (const algo of activeAlgos()) {
      if (!state.finished[algo]) setStatus(algo, 'running', state.mode === 'race' ? 'racing' : 'solving');
    }
  }
});

canvas.addEventListener('pointerdown', e => {
  state.dragging = true;
  state.dragX = e.clientX;
  canvas.classList.add('dragging');
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', e => {
  if (!state.dragging) return;
  state.yawBase += (e.clientX - state.dragX) * 0.006;
  state.dragX = e.clientX;
});

canvas.addEventListener('pointerup', () => {
  state.dragging = false;
  canvas.classList.remove('dragging');
});

async function boot() {
  const [bfsSrc, dfsSrc] = await Promise.all([
    fetch('/src/Bfs.java').then(r => r.text()),
    fetch('/src/Dfs.java').then(r => r.text())
  ]);
  renderCode('bfs', bfsSrc);
  renderCode('dfs', dfsSrc);
  await fetchRace();
  requestAnimationFrame(tick);
}

boot();
