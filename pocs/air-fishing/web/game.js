const sea = document.getElementById("sea");
const ctx = sea.getContext("2d");
const W = sea.width, H = sea.height;
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");

const elStatus = document.getElementById("status");
const elClock = document.getElementById("clock");
const elMode = document.getElementById("modelabel");
const elName1 = document.getElementById("nameP1");
const elName2 = document.getElementById("nameP2");
const elScore1 = document.getElementById("scoreP1");
const elScore2 = document.getElementById("scoreP2");
const elMenu = document.getElementById("menu");

const P1 = "#2563eb", P2 = "#f97316";
const WATER_TOP = 70, SAND_TOP = H - 58;
const FISH_MIN_Y = WATER_TOP + 28, FISH_MAX_Y = SAND_TOP - 24;
const MAXFISH = 11, GAME_FRAMES = 60 * 60;
const REEL = 26, GRAB_CD = 26, KB_SPEED = 9;

const KINDS = [
  { color: "#fb923c", belly: "#fdba74", points: 1, r: 16, speed: 0.75, weight: 6 },
  { color: "#f472b6", belly: "#f9a8d4", points: 2, r: 20, speed: 0.95, weight: 3 },
  { color: "#38bdf8", belly: "#7dd3fc", points: 3, r: 26, speed: 0.55, weight: 2 },
  { color: "#facc15", belly: "#fde68a", points: 5, r: 17, speed: 1.5, weight: 1 },
];
const WEIGHT_SUM = KINDS.reduce((a, k) => a + k.weight, 0);

let mode = null;
let phase = "menu";
let countTimer = 0;
let timeLeft = GAME_FRAMES;
let camReady = false, wsReady = false, audioUnlocked = false;
let soundOn = true;

let score = { 1: 0, 2: 0 };
let fish = [];
let bubbles = [];
let popups = [];
let drops = [];
let waveT = 0;

function makeSlot() {
  return { present: false, x: 0.5, y: 0.5, pinch: false };
}
const slots = { L: makeSlot(), R: makeSlot() };
const keys = { up: false, down: false, left: false, right: false, w: false, a: false, s: false, d: false, p1: false, p2: false };

function makeHook(id, color, anchorX, smooth) {
  return { id, color, anchorX, anchorY: 8, x: anchorX, y: 150, tx: anchorX, ty: 150, pinch: false, prev: false, cool: 0, grab: 0, smooth };
}
const hooks = { L: makeHook(1, P1, W * 0.25, 0.3), R: makeHook(2, P2, W * 0.75, 0.3) };
const cpu = { target: null, repick: 0 };

function sfx(name) { if (soundOn) Sea[name](); }

function pickKind() {
  let r = Math.random() * WEIGHT_SUM;
  for (const k of KINDS) { if (r < k.weight) return k; r -= k.weight; }
  return KINDS[0];
}

function spawnFish(fromEdge) {
  const k = pickKind();
  const dir = Math.random() < 0.5 ? 1 : -1;
  const y = FISH_MIN_Y + Math.random() * (FISH_MAX_Y - FISH_MIN_Y);
  const x = fromEdge ? (dir > 0 ? -40 : W + 40) : 60 + Math.random() * (W - 120);
  fish.push({
    kind: k, x, y, baseY: y, dir,
    vx: k.speed * dir * (0.8 + Math.random() * 0.5),
    phase: Math.random() * Math.PI * 2, amp: 8 + Math.random() * 14,
    tail: Math.random() * Math.PI * 2,
    caught: 0, reel: 0, fromX: 0, fromY: 0,
  });
}

function seedFish() {
  fish = [];
  for (let i = 0; i < MAXFISH; i++) spawnFish(false);
}

function seedBubbles() {
  bubbles = [];
  for (let i = 0; i < 26; i++) {
    bubbles.push({ x: Math.random() * W, y: WATER_TOP + Math.random() * (SAND_TOP - WATER_TOP), r: 1.5 + Math.random() * 3.5, sp: 0.3 + Math.random() * 0.8, wob: Math.random() * Math.PI * 2 });
  }
}

function assignHands(hands) {
  const bySide = { L: null, R: null };
  for (const h of hands) {
    const s = h.side;
    if (!bySide[s]) bySide[s] = h;
    else if (Math.abs(h.x - 0.5) < Math.abs(bySide[s].x - 0.5)) bySide[s] = h;
  }
  for (const key of ["L", "R"]) {
    const slot = slots[key], h = bySide[key];
    if (!h) { slot.present = false; slot.pinch = false; continue; }
    slot.present = true;
    slot.x = h.x; slot.y = h.y; slot.pinch = h.pinch;
  }
}

function depthY(ny) {
  return WATER_TOP + 10 + ny * (SAND_TOP - WATER_TOP - 22);
}

function resolveInputs() {
  const L = hooks.L, R = hooks.R;
  if (slots.L.present) {
    L.tx = slots.L.x * W;
    L.ty = depthY(slots.L.y);
    L.pinch = slots.L.pinch;
  } else {
    if (keys.a) L.tx -= KB_SPEED;
    if (keys.d) L.tx += KB_SPEED;
    if (keys.w) L.ty -= KB_SPEED;
    if (keys.s) L.ty += KB_SPEED;
    L.pinch = keys.p1;
  }
  if (mode === "cpu") {
    updateCPU();
  } else if (slots.R.present) {
    R.tx = slots.R.x * W;
    R.ty = depthY(slots.R.y);
    R.pinch = slots.R.pinch;
  } else {
    if (keys.left) R.tx -= KB_SPEED;
    if (keys.right) R.tx += KB_SPEED;
    if (keys.up) R.ty -= KB_SPEED;
    if (keys.down) R.ty += KB_SPEED;
    R.pinch = keys.p2;
  }
  for (const h of [L, R]) {
    h.tx = Math.max(20, Math.min(W - 20, h.tx));
    h.ty = Math.max(WATER_TOP + 8, Math.min(SAND_TOP - 6, h.ty));
  }
}

function updateCPU() {
  const h = hooks.R;
  if (!cpu.target || cpu.target.caught || fish.indexOf(cpu.target) < 0 || cpu.repick <= 0) {
    let best = null, bd = 1e9;
    for (const f of fish) {
      if (f.caught || f.x < 10 || f.x > W - 10) continue;
      const d = (f.x - h.x) ** 2 + (f.y - h.y) ** 2;
      if (d < bd) { bd = d; best = f; }
    }
    cpu.target = best;
    cpu.repick = 26;
  }
  cpu.repick--;
  if (cpu.target) {
    h.tx = cpu.target.x;
    h.ty = cpu.target.y;
    const near = (cpu.target.x - h.x) ** 2 + (cpu.target.y - h.y) ** 2 < (cpu.target.kind.r + 12) ** 2;
    h.pinch = near;
  } else {
    h.pinch = false;
  }
}

function tryGrab(h) {
  let best = null, bd = 1e9;
  for (const f of fish) {
    if (f.caught) continue;
    const rad = f.kind.r + 16;
    const d = (f.x - h.x) ** 2 + (f.y - h.y) ** 2;
    if (d < rad * rad && d < bd) { bd = d; best = f; }
  }
  if (best) {
    best.caught = h.id;
    best.reel = REEL;
    best.fromX = best.x;
    best.fromY = best.y;
    score[h.id] += best.kind.points;
    popups.push({ x: best.x, y: best.y, text: "+" + best.kind.points, color: h.color, life: 50 });
    for (let i = 0; i < 9; i++) drops.push({ x: best.x, y: best.y, vx: (Math.random() - 0.5) * 4, vy: -1 - Math.random() * 3, life: 26, color: h.color });
    h.cool = GRAB_CD;
    h.grab = 12;
    sfx("catch");
    sfx("reel");
  } else {
    h.cool = 10;
    h.grab = 8;
    sfx("plop");
  }
}

function updateHooks() {
  for (const h of [hooks.L, hooks.R]) {
    h.x += (h.tx - h.x) * h.smooth;
    h.y += (h.ty - h.y) * h.smooth;
    if (h.cool > 0) h.cool--;
    if (h.grab > 0) h.grab--;
    if (h.pinch && !h.prev && h.cool === 0) tryGrab(h);
    h.prev = h.pinch;
  }
}

function updateFish() {
  for (let i = fish.length - 1; i >= 0; i--) {
    const f = fish[i];
    f.tail += 0.3;
    if (f.caught) {
      f.reel--;
      const t = 1 - f.reel / REEL;
      const ax = f.caught === 1 ? hooks.L.anchorX : hooks.R.anchorX;
      f.x = f.fromX + (ax - f.fromX) * t;
      f.y = f.fromY + (10 - f.fromY) * t;
      if (f.reel <= 0) { fish.splice(i, 1); spawnFish(true); }
      continue;
    }
    f.phase += 0.04;
    f.x += f.vx;
    f.y = f.baseY + Math.sin(f.phase) * f.amp;
    if (f.x < -60 || f.x > W + 60) { fish.splice(i, 1); spawnFish(true); }
  }
  while (fish.length < MAXFISH) spawnFish(true);
}

function updateBubbles() {
  for (const b of bubbles) {
    b.y -= b.sp;
    b.wob += 0.05;
    b.x += Math.sin(b.wob) * 0.3;
    if (b.y < WATER_TOP + 4) { b.y = SAND_TOP - 4; b.x = Math.random() * W; }
  }
}

function updateParticles() {
  for (let i = popups.length - 1; i >= 0; i--) {
    const p = popups[i];
    p.y -= 0.7; p.life--;
    if (p.life <= 0) popups.splice(i, 1);
  }
  for (let i = drops.length - 1; i >= 0; i--) {
    const d = drops[i];
    d.x += d.vx; d.y += d.vy; d.vy += 0.25; d.life--;
    if (d.life <= 0) drops.splice(i, 1);
  }
}

function update() {
  waveT += 0.03;
  updateBubbles();
  updateParticles();
  if (phase === "count") {
    countTimer--;
    if (countTimer <= 0) { phase = "play"; }
    return;
  }
  if (phase !== "play") return;
  resolveInputs();
  updateHooks();
  updateFish();
  timeLeft--;
  if (timeLeft <= 0) endGame();
}

function startMatch(m) {
  mode = m;
  elMode.textContent = m === "cpu" ? "1P vs CPU" : "P1 vs P2";
  elName1.textContent = "PLAYER 1";
  elName2.textContent = m === "cpu" ? "CPU" : "PLAYER 2";
  hooks.R.smooth = m === "cpu" ? 0.17 : 0.3;
  score = { 1: 0, 2: 0 };
  timeLeft = GAME_FRAMES;
  hooks.L.x = hooks.L.tx = W * 0.25; hooks.L.y = hooks.L.ty = 170; hooks.L.cool = 0;
  hooks.R.x = hooks.R.tx = W * 0.75; hooks.R.y = hooks.R.ty = 170; hooks.R.cool = 0;
  cpu.target = null; cpu.repick = 0;
  popups = []; drops = [];
  seedFish();
  countTimer = 180;
  phase = "count";
  elMenu.classList.add("hidden");
  hideResult();
}

function endGame() {
  phase = "over";
  sfx("fanfare");
  showResult();
}

function restart() {
  phase = "menu";
  hideResult();
  elMenu.classList.remove("hidden");
}

let resultEl = null;
function showResult() {
  if (!resultEl) {
    resultEl = document.createElement("div");
    resultEl.className = "overlay";
    document.querySelector(".shell").appendChild(resultEl);
  }
  const s1 = score[1], s2 = score[2];
  const p2name = mode === "cpu" ? "CPU" : "Player 2";
  let title, tint;
  if (s1 > s2) { title = "PLAYER 1 WINS"; tint = P1; }
  else if (s2 > s1) { title = (mode === "cpu" ? "CPU WINS" : "PLAYER 2 WINS"); tint = P2; }
  else { title = "IT'S A TIE"; tint = "#0e4d6e"; }
  resultEl.innerHTML =
    '<div class="card">' +
    '<h2 style="color:' + tint + '">' + title + '</h2>' +
    '<p style="font-size:22px"><b style="color:' + P1 + '">Player 1</b> ' + s1 + ' &nbsp;&mdash;&nbsp; ' + s2 + ' <b style="color:' + P2 + '">' + p2name + '</b></p>' +
    '<div class="modes"><button id="again" class="big" style="background:linear-gradient(160deg,#34d399,#059669)">PLAY AGAIN</button></div>' +
    '<p class="small">press <b>R</b> to return to the menu</p>' +
    '</div>';
  resultEl.classList.remove("hidden");
  document.getElementById("again").addEventListener("click", () => startMatch(mode));
}
function hideResult() { if (resultEl) resultEl.classList.add("hidden"); }

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

function drawSea() {
  const sky = ctx.createLinearGradient(0, 0, 0, WATER_TOP);
  sky.addColorStop(0, "#eaf7ff");
  sky.addColorStop(1, "#d3eefb");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, WATER_TOP);

  ctx.save();
  const sun = ctx.createRadialGradient(W - 90, 36, 6, W - 90, 36, 60);
  sun.addColorStop(0, "rgba(255,247,205,0.95)");
  sun.addColorStop(1, "rgba(255,247,205,0)");
  ctx.fillStyle = sun;
  ctx.beginPath(); ctx.arc(W - 90, 36, 60, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  const water = ctx.createLinearGradient(0, WATER_TOP, 0, SAND_TOP);
  water.addColorStop(0, "#bfe9f5");
  water.addColorStop(0.5, "#8fd6ea");
  water.addColorStop(1, "#57bcd8");
  ctx.fillStyle = water;
  ctx.fillRect(0, WATER_TOP, W, SAND_TOP - WATER_TOP);

  ctx.fillStyle = "#cdeffb";
  ctx.beginPath();
  ctx.moveTo(0, WATER_TOP);
  for (let x = 0; x <= W; x += 12) {
    ctx.lineTo(x, WATER_TOP + Math.sin(x * 0.03 + waveT) * 4 + 4);
  }
  ctx.lineTo(W, WATER_TOP); ctx.closePath(); ctx.fill();

  ctx.save();
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = "#ffffff";
  for (let i = 0; i < 4; i++) {
    const x = (i + 0.5) * (W / 4) + Math.sin(waveT + i) * 20;
    ctx.beginPath();
    ctx.moveTo(x, WATER_TOP);
    ctx.lineTo(x + 60, SAND_TOP);
    ctx.lineTo(x + 110, SAND_TOP);
    ctx.lineTo(x + 40, WATER_TOP);
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

function drawBubbles() {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  for (const b of bubbles) {
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
  }
  ctx.restore();
}

function drawSand() {
  ctx.fillStyle = "#f4e3b0";
  ctx.beginPath();
  ctx.moveTo(0, SAND_TOP + 6);
  for (let x = 0; x <= W; x += 28) {
    ctx.lineTo(x, SAND_TOP + Math.sin(x * 0.05) * 5);
  }
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fill();

  ctx.fillStyle = "#e7d093";
  for (let i = 0; i < 40; i++) {
    const x = (i * 53) % W;
    const y = SAND_TOP + 14 + ((i * 29) % 36);
    ctx.beginPath(); ctx.arc(x, y, 1.6, 0, Math.PI * 2); ctx.fill();
  }

  ctx.strokeStyle = "#5fb98a";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  for (const sx of [70, 200, 470, 660, 820]) {
    ctx.beginPath();
    ctx.moveTo(sx, SAND_TOP + 6);
    for (let s = 0; s < 5; s++) {
      const yy = SAND_TOP - s * 12;
      ctx.lineTo(sx + Math.sin(waveT * 0.8 + s * 0.9 + sx) * 9, yy);
    }
    ctx.stroke();
  }
}

function drawFish(f) {
  const r = f.kind.r;
  const face = f.dir >= 0 ? 1 : -1;
  ctx.save();
  ctx.translate(f.x, f.y);
  ctx.scale(face, 1);
  if (f.caught) {
    const t = 1 - f.reel / REEL;
    ctx.globalAlpha = Math.max(0.2, 1 - t * 0.6);
    ctx.rotate(Math.sin(f.tail) * 0.3);
  }
  const wig = Math.sin(f.tail) * 0.4;
  ctx.fillStyle = f.kind.color;
  ctx.beginPath();
  ctx.moveTo(-r * 0.9, 0);
  ctx.quadraticCurveTo(-r * 0.4, -r * 0.7, r * 0.7, 0);
  ctx.quadraticCurveTo(-r * 0.4, r * 0.7, -r * 0.9, 0);
  ctx.fill();
  ctx.fillStyle = f.kind.belly;
  ctx.beginPath();
  ctx.ellipse(r * 0.05, r * 0.18, r * 0.55, r * 0.34, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = f.kind.color;
  ctx.beginPath();
  ctx.moveTo(-r * 0.8, 0);
  ctx.lineTo(-r * 1.5, -r * 0.6 + wig * r);
  ctx.lineTo(-r * 1.5, r * 0.6 + wig * r);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.beginPath(); ctx.arc(r * 0.42, -r * 0.18, r * 0.2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#143b53";
  ctx.beginPath(); ctx.arc(r * 0.48, -r * 0.18, r * 0.1, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawHook(h) {
  ctx.save();
  ctx.strokeStyle = "rgba(20,59,83,0.45)";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(h.anchorX, h.anchorY);
  ctx.quadraticCurveTo((h.anchorX + h.x) / 2, h.y * 0.4, h.x, h.y - 12);
  ctx.stroke();

  ctx.strokeStyle = h.color;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(h.x, h.y - 12);
  ctx.lineTo(h.x, h.y + 2);
  ctx.arc(h.x - 5, h.y + 2, 5, 0, Math.PI * (h.grab > 0 ? 1.9 : 1.4), false);
  ctx.stroke();

  if (h.grab > 0 || h.pinch) {
    ctx.strokeStyle = "rgba(22,163,74,0.85)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(h.x, h.y, 16 + (h.grab > 0 ? (12 - h.grab) : 0), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = h.color;
  ctx.beginPath(); ctx.arc(h.x, h.y - 12, 3.2, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = h.color;
  roundRect(ctx, h.anchorX - 16, 0, 32, 10, 4);
  ctx.fill();
  ctx.restore();
}

function drawParticles() {
  for (const d of drops) {
    ctx.globalAlpha = Math.max(0, d.life / 26);
    ctx.fillStyle = d.color;
    ctx.beginPath(); ctx.arc(d.x, d.y, 2.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.font = "bold 22px Trebuchet MS, sans-serif";
  ctx.textAlign = "center";
  for (const p of popups) {
    ctx.globalAlpha = Math.max(0, p.life / 50);
    ctx.fillStyle = p.color;
    ctx.fillText(p.text, p.x, p.y);
  }
  ctx.globalAlpha = 1;
}

function drawCenter(text, sub) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(14,77,110,0.92)";
  ctx.font = "bold 92px Trebuchet MS, sans-serif";
  ctx.fillText(text, W / 2, H / 2 + 8);
  if (sub) {
    ctx.font = "bold 24px Trebuchet MS, sans-serif";
    ctx.fillStyle = "rgba(14,77,110,0.8)";
    ctx.fillText(sub, W / 2, H / 2 + 54);
  }
  ctx.restore();
}

function render() {
  drawSea();
  drawSand();
  drawBubbles();
  for (const f of fish) if (!f.caught) drawFish(f);
  if (phase === "play" || phase === "over") {
    drawHook(hooks.L);
    drawHook(hooks.R);
    for (const f of fish) if (f.caught) drawFish(f);
  }
  drawParticles();
  if (phase === "count") {
    const n = Math.ceil(countTimer / 60);
    drawCenter(n > 0 ? String(n) : "GO", "get ready");
  }
  elScore1.textContent = score[1];
  elScore2.textContent = score[2];
  elClock.textContent = Math.max(0, Math.ceil(timeLeft / 60));
  updateStatus();
}

function updateStatus() {
  if (!wsReady) { elStatus.textContent = "linking to tracker…"; return; }
  if (!camReady) { elStatus.textContent = "enable camera or use keys"; return; }
  const n = (slots.L.present ? 1 : 0) + (slots.R.present ? 1 : 0);
  elStatus.textContent = n === 0 ? "show your hands" : n + (n === 1 ? " hand tracked" : " hands tracked");
}

function drawPip() {
  pctx.save();
  pctx.translate(pip.width, 0);
  pctx.scale(-1, 1);
  if (camReady) {
    pctx.drawImage(cam, 0, 0, pip.width, pip.height);
    pctx.restore();
    for (const key of ["L", "R"]) {
      const s = slots[key];
      if (!s.present) continue;
      const mx = (1 - s.x) * pip.width, my = s.y * pip.height;
      pctx.fillStyle = key === "L" ? P1 : P2;
      pctx.beginPath(); pctx.arc(mx, my, 9, 0, Math.PI * 2); pctx.fill();
      pctx.strokeStyle = s.pinch ? "#16a34a" : "rgba(255,255,255,0.9)";
      pctx.lineWidth = 3;
      pctx.beginPath(); pctx.arc(mx, my, 13, 0, Math.PI * 2); pctx.stroke();
    }
  } else {
    pctx.restore();
    pctx.fillStyle = "#eaf4f9";
    pctx.fillRect(0, 0, pip.width, pip.height);
    pctx.fillStyle = "#6f97ac";
    pctx.font = "13px Trebuchet MS, sans-serif";
    pctx.textAlign = "center";
    pctx.fillText("camera off — keyboard play", pip.width / 2, pip.height / 2);
  }
}

let ws = null, awaiting = false, lastSend = 0;

function connectWS() {
  try {
    ws = new WebSocket("ws://" + location.hostname + ":8765");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsReady = true; };
    ws.onmessage = (e) => {
      awaiting = false;
      try {
        const m = JSON.parse(e.data);
        assignHands(Array.isArray(m.hands) ? m.hands : []);
      } catch (_) {}
    };
    ws.onclose = () => { wsReady = false; awaiting = false; setTimeout(connectWS, 1500); };
    ws.onerror = () => {};
  } catch (_) {
    setTimeout(connectWS, 1500);
  }
}

async function initCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
    cam.srcObject = stream;
    await cam.play();
    camReady = true;
    return true;
  } catch (_) {
    camReady = false;
    return false;
  }
}

function maybeSend(ts) {
  if (!ws || ws.readyState !== 1 || awaiting || !camReady) return;
  if (ts - lastSend < 60) return;
  lastSend = ts;
  awaiting = true;
  gctx.drawImage(cam, 0, 0, grab.width, grab.height);
  grab.toBlob((blob) => {
    if (!blob || !ws || ws.readyState !== 1) { awaiting = false; return; }
    blob.arrayBuffer().then((b) => { if (ws.readyState === 1) ws.send(b); else awaiting = false; });
  }, "image/jpeg", 0.5);
}

function frame(ts) {
  maybeSend(ts);
  update();
  render();
  drawPip();
  requestAnimationFrame(frame);
}

function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  Sea.ensure();
}

window.addEventListener("keydown", (e) => {
  unlockAudio();
  const k = e.key.toLowerCase();
  if (k === "r") { restart(); return; }
  if (k === "w") keys.w = true;
  else if (k === "a") keys.a = true;
  else if (k === "s") keys.s = true;
  else if (k === "d") keys.d = true;
  else if (k === "arrowup") { keys.up = true; e.preventDefault(); }
  else if (k === "arrowdown") { keys.down = true; e.preventDefault(); }
  else if (k === "arrowleft") { keys.left = true; e.preventDefault(); }
  else if (k === "arrowright") { keys.right = true; e.preventDefault(); }
  else if (k === " " || e.key === "Spacebar") { keys.p1 = true; e.preventDefault(); }
  else if (k === "enter") { keys.p2 = true; }
});

window.addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (k === "w") keys.w = false;
  else if (k === "a") keys.a = false;
  else if (k === "s") keys.s = false;
  else if (k === "d") keys.d = false;
  else if (k === "arrowup") keys.up = false;
  else if (k === "arrowdown") keys.down = false;
  else if (k === "arrowleft") keys.left = false;
  else if (k === "arrowright") keys.right = false;
  else if (k === " " || e.key === "Spacebar") keys.p1 = false;
  else if (k === "enter") keys.p2 = false;
});

document.getElementById("mode-cpu").addEventListener("click", () => { unlockAudio(); initCam(); startMatch("cpu"); });
document.getElementById("mode-pvp").addEventListener("click", () => { unlockAudio(); initCam(); startMatch("pvp"); });

document.getElementById("sound").addEventListener("click", (e) => {
  soundOn = !soundOn;
  e.target.textContent = "SOUND: " + (soundOn ? "ON" : "OFF");
});

function toggleFullscreen() {
  const el = document.querySelector(".shell");
  const on = document.fullscreenElement || document.webkitFullscreenElement;
  if (on) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  else (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
}
document.getElementById("fs").addEventListener("click", toggleFullscreen);

seedBubbles();
seedFish();
connectWS();
requestAnimationFrame(frame);
