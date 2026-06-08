const ring = document.getElementById("ring");
const ctx = ring.getContext("2d");
const W = ring.width, H = ring.height;
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");

const elStatus = document.getElementById("status");
const elRound = document.getElementById("roundno");
const elMode = document.getElementById("modelabel");
const elName1 = document.getElementById("nameP1");
const elName2 = document.getElementById("nameP2");
const elMenu = document.getElementById("menu");

const LIFE = 4, WIN = 2;
const PUNCH_DUR = 20, HITSTUN = 28;
const GUARD_Y = 0.8, THRUST = 1.4, RESET = 1.12;
const KO_TIME = 110, CARD_TIME = 150, INTRO_TIME = 110, MATCH_TIME = 260;
const GROUND = H - 64;

const DIFF = {
  1: { tele: 80, recover: 64, cd: [120, 175], block: 0.06, feint: 0.0, hold: 0.22 },
  2: { tele: 58, recover: 50, cd: [90, 135], block: 0.16, feint: 0.10, hold: 0.32 },
  3: { tele: 42, recover: 38, cd: [62, 100], block: 0.30, feint: 0.22, hold: 0.42 },
};

let mode = null;
let phase = "menu";
let phaseTimer = 0;
let round = 1;
let fightFlash = 0;
let lastWinner = null;
let matchWinner = null;
let camReady = false, wsReady = false, audioUnlocked = false;
let soundOn = true;
let facesCaptured = false;

function makeFighter(side, name, trunk) {
  return {
    side, name, trunk,
    baseX: side === "left" ? W * 0.30 : W * 0.70,
    facing: side === "left" ? 1 : -1,
    life: LIFE, score: 0, guard: false,
    hitStun: 0, flash: 0, knock: 0, blockFlash: 0, guardHold: 0,
    ko: false, koAnim: 0,
    punch: { L: { t: 0, res: false }, R: { t: 0, res: false } },
    nextGlove: "R", isCPU: false, opp: null, face: null,
    ai: { state: "guard", timer: 60, react: 0, feint: false },
  };
}

const P1 = makeFighter("left", "PLAYER 1", "#2563eb");
const P2 = makeFighter("right", "PLAYER 2", "#ef4444");
P1.opp = P2; P2.opp = P1;

function makeSlot() {
  return { present: false, x: 0.5, y: 0.5, fist: false, open: false, scale: 0.1, baseline: 0, latched: false, blocking: false, dx: 0.5, dy: 0.5, punchFlash: 0 };
}
const slots = { L: makeSlot(), R: makeSlot() };

const keys = { p1block: false, p2block: false };

const lifeSegs = { left: buildSegs("lifeP1"), right: buildSegs("lifeP2") };
const pipEls = { left: buildPips("pipsP1"), right: buildPips("pipsP2") };

function buildSegs(id) {
  const host = document.getElementById(id);
  const out = [];
  for (let i = 0; i < LIFE; i++) {
    const d = document.createElement("div");
    d.className = "seg";
    host.appendChild(d);
    out.push(d);
  }
  return out;
}

function buildPips(id) {
  const host = document.getElementById(id);
  const out = [];
  for (let i = 0; i < WIN; i++) {
    const d = document.createElement("span");
    d.className = "pip";
    host.appendChild(d);
    out.push(d);
  }
  return out;
}

function updateLife() {
  lifeSegs.left.forEach((s, i) => s.classList.toggle("empty", i >= P1.life));
  lifeSegs.right.forEach((s, i) => s.classList.toggle("empty", i >= P2.life));
}

function updatePips() {
  pipEls.left.forEach((p, i) => p.classList.toggle("on", i < P1.score));
  pipEls.right.forEach((p, i) => p.classList.toggle("on", i < P2.score));
}

function sfx(name) { if (soundOn) Ring[name](); }

function fighterThrow(f, g) {
  if (f.ko || f.hitStun > 0) return;
  if (f.punch[g].t > 0) return;
  f.punch[g] = { t: PUNCH_DUR, res: false };
  sfx("whoosh");
  if (mode === "cpu" && f === P1) {
    const d = DIFF[round];
    if (Math.random() < d.block) P2.ai.react = PUNCH_DUR;
  }
}

function nextGlove(f) {
  const g = f.nextGlove;
  f.nextGlove = g === "L" ? "R" : "L";
  return g;
}

function routePunch(key) {
  if (mode === "cpu") fighterThrow(P1, key);
  else if (key === "L") fighterThrow(P1, nextGlove(P1));
  else fighterThrow(P2, nextGlove(P2));
}

function slotPunchEdge(slot) {
  if (slot.fist && slot.scale > slot.baseline * THRUST && !slot.latched) {
    slot.latched = true;
    slot.punchFlash = 10;
    return true;
  }
  if (slot.scale < slot.baseline * RESET) slot.latched = false;
  return false;
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
    if (!h) { slot.present = false; slot.blocking = false; slot.latched = false; continue; }
    slot.present = true;
    slot.x = h.x; slot.y = h.y; slot.fist = h.fist; slot.open = h.open; slot.scale = h.scale;
    if (slot.baseline === 0) slot.baseline = h.scale;
    slot.baseline += (h.scale - slot.baseline) * 0.04;
    slot.blocking = h.open && h.y < GUARD_Y;
    if (phase === "fight" && slotPunchEdge(slot)) routePunch(key);
  }
}

function setGuard(f, raw) {
  if (raw) f.guardHold = 8;
  f.guard = raw || f.guardHold > 0;
}

function computeGuards() {
  if (mode === "cpu") {
    setGuard(P1, slots.L.blocking || slots.R.blocking || keys.p1block);
  } else {
    setGuard(P1, slots.L.blocking || keys.p1block);
    setGuard(P2, slots.R.blocking || keys.p2block);
  }
}

function updateCPU() {
  const c = P2, p = P1, d = DIFF[round], ai = c.ai;
  if (ai.react > 0) ai.react--;
  if (c.hitStun > 0 || c.ko) { c.guard = ai.react > 0; return; }
  ai.timer--;
  switch (ai.state) {
    case "guard":
      c.guard = p.guard ? true : Math.random() < d.hold;
      if (ai.timer <= 0) {
        ai.feint = p.guard && Math.random() < d.feint;
        ai.state = "telegraph";
        ai.timer = d.tele;
      }
      break;
    case "telegraph":
      c.guard = false;
      if (ai.timer <= 0) {
        if (ai.feint || (p.guard && Math.random() < d.feint)) {
          ai.state = "recover"; ai.timer = d.recover;
        } else {
          fighterThrow(c, Math.random() < 0.5 ? "L" : "R");
          ai.state = "recover"; ai.timer = d.recover;
        }
      }
      break;
    case "recover":
      c.guard = Math.random() < d.hold;
      if (ai.timer <= 0) {
        ai.state = "guard";
        ai.timer = d.cd[0] + Math.random() * (d.cd[1] - d.cd[0]);
      }
      break;
  }
  if (ai.react > 0) c.guard = true;
}

function resolveHit(att, def) {
  if (def.ko) return;
  if (def.guard) {
    sfx("clink");
    def.knock = 6 * -def.facing;
    def.blockFlash = 12;
  } else {
    sfx("thud");
    def.life = Math.max(0, def.life - 1);
    def.flash = 14;
    def.hitStun = HITSTUN;
    def.knock = 15 * -def.facing;
    updateLife();
    if (def.life <= 0) doKO(att, def);
  }
}

function stepPunches(f) {
  for (const g of ["L", "R"]) {
    const p = f.punch[g];
    if (p.t <= 0) continue;
    const progress = (PUNCH_DUR - p.t) / PUNCH_DUR;
    if (!p.res && progress >= 0.5) { p.res = true; resolveHit(f, f.opp); }
    p.t--;
  }
}

function tickTimers(f) {
  if (f.hitStun > 0) f.hitStun--;
  if (f.flash > 0) f.flash--;
  if (f.blockFlash > 0) f.blockFlash--;
  if (f.guardHold > 0) f.guardHold--;
  f.knock *= 0.85;
  if (Math.abs(f.knock) < 0.3) f.knock = 0;
  if (f.ko) f.koAnim = Math.min(1, f.koAnim + 0.03);
}

function doKO(winner, loser) {
  loser.ko = true; loser.koAnim = 0;
  winner.score++;
  updatePips();
  lastWinner = winner;
  phase = "ko"; phaseTimer = KO_TIME;
  sfx("bell"); sfx("crowd");
}

function resetRound() {
  for (const f of [P1, P2]) {
    f.life = LIFE; f.guard = false; f.hitStun = 0; f.flash = 0;
    f.knock = 0; f.ko = false; f.koAnim = 0; f.nextGlove = "R"; f.blockFlash = 0; f.guardHold = 0;
    f.punch = { L: { t: 0, res: false }, R: { t: 0, res: false } };
    f.ai = { state: "guard", timer: 60, react: 0, feint: false };
  }
  updateLife();
}

function startRoundIntro() {
  resetRound();
  elRound.textContent = round;
  phase = "round_intro";
  phaseTimer = INTRO_TIME;
}

function startFight() {
  phase = "fight";
  fightFlash = 45;
  sfx("bell");
}

function phaseAdvance() {
  if (phase === "round_intro") startFight();
  else if (phase === "ko") {
    if (lastWinner.score >= WIN) { matchWinner = lastWinner; phase = "match_over"; phaseTimer = MATCH_TIME; }
    else { phase = "round_over"; phaseTimer = CARD_TIME; }
  } else if (phase === "round_over") {
    round++;
    startRoundIntro();
  } else if (phase === "match_over") {
    startMatch(mode);
  }
}

function startMatch(m) {
  mode = m;
  round = 1;
  P1.score = 0; P2.score = 0;
  matchWinner = null;
  facesCaptured = false; P1.face = null; P2.face = null;
  P2.isCPU = m === "cpu";
  P2.name = m === "cpu" ? "CPU" : "PLAYER 2";
  elName2.textContent = P2.name;
  elMode.textContent = m === "cpu" ? "1P vs CPU" : "P1 vs P2";
  updatePips();
  elMenu.classList.add("hidden");
  startRoundIntro();
}

function restart() {
  if (mode) startMatch(mode);
}

function update() {
  if (fightFlash > 0) fightFlash--;
  for (const s of [slots.L, slots.R]) {
    if (s.punchFlash > 0) s.punchFlash--;
    s.dx += (s.x - s.dx) * 0.3;
    s.dy += (s.y - s.dy) * 0.3;
  }
  for (const f of [P1, P2]) tickTimers(f);
  if (!facesCaptured && camReady && mode) facesCaptured = captureFaces();
  if (phase === "fight") {
    computeGuards();
    if (mode === "cpu") updateCPU();
    for (const f of [P1, P2]) stepPunches(f);
  } else if (phase !== "menu") {
    if (phaseTimer > 0) { phaseTimer--; if (phaseTimer <= 0) phaseAdvance(); }
  }
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}
function fillRR(x, y, w, h, r) { roundRect(ctx, x, y, w, h, r); ctx.fill(); }

function drawScene() {
  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, "#eaf3fc");
  sky.addColorStop(1, "#dbe7f4");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = "#cdd9ea";
  ctx.fillRect(0, 0, W, 120);
  const pastel = ["#f5c9d6", "#cfe9d4", "#f6e2bd", "#cfe0f5", "#e7d4f0"];
  for (let row = 0; row < 3; row++) {
    for (let i = 0; i < 30; i++) {
      ctx.fillStyle = pastel[(i + row) % pastel.length];
      ctx.beginPath();
      ctx.arc(16 + i * 30, 26 + row * 26, 9, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = "#dff0db";
  ctx.fillRect(0, GROUND - 10, W, H - GROUND + 10);
  ctx.fillStyle = "#fbfaf3";
  ctx.beginPath();
  ctx.moveTo(70, GROUND - 8);
  ctx.lineTo(W - 70, GROUND - 8);
  ctx.lineTo(W - 20, H);
  ctx.lineTo(20, H);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#ece5cf";
  ctx.lineWidth = 1;
  for (let i = 1; i < 8; i++) {
    const x = 20 + (W - 40) * (i / 8);
    ctx.beginPath();
    ctx.moveTo(W / 2 + (x - W / 2) * 0.62, GROUND - 8);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  for (const px of [70, W - 70]) {
    ctx.fillStyle = "#c0392b";
    fillRR(px - 7, 150, 14, GROUND - 150, 6);
    ctx.fillStyle = "#e74c3c";
    ctx.beginPath();
    ctx.arc(px, 150, 12, 0, Math.PI * 2);
    ctx.fill();
  }
  const ropes = ["#e74c3c", "#ecf0f1", "#3498db"];
  ropes.forEach((col, i) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = 4;
    const y = 180 + i * 40;
    ctx.beginPath();
    ctx.moveTo(70, y);
    ctx.lineTo(W - 70, y);
    ctx.stroke();
  });
}

function cropFace(src, sx, sw) {
  const f = document.createElement("canvas");
  f.width = 96; f.height = 96;
  f.getContext("2d").drawImage(src, sx, src.height * 0.05, sw, sw, 0, 0, f.width, f.height);
  return f;
}

function captureFaces() {
  if (!camReady) return false;
  const tmp = document.createElement("canvas");
  tmp.width = grab.width; tmp.height = grab.height;
  const tc = tmp.getContext("2d");
  tc.save();
  tc.translate(tmp.width, 0);
  tc.scale(-1, 1);
  tc.drawImage(cam, 0, 0, tmp.width, tmp.height);
  tc.restore();
  const sw = tmp.width * 0.4;
  if (mode === "pvp") {
    P1.face = cropFace(tmp, tmp.width * 0.06, sw);
    P2.face = cropFace(tmp, tmp.width * 0.54, sw);
  } else {
    P1.face = cropFace(tmp, tmp.width * 0.3, sw);
    P2.face = null;
  }
  return true;
}

function drawHead(f, gx, hit) {
  const hy = GROUND - 208, r = 27;
  if (f.face) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(gx, hy, r, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(f.face, gx - r, hy - r, r * 2, r * 2);
    if (hit) { ctx.fillStyle = "rgba(255,90,90,.45)"; ctx.fillRect(gx - r, hy - r, r * 2, r * 2); }
    ctx.restore();
    ctx.strokeStyle = f.trunk;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(gx, hy, r, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    ctx.fillStyle = hit ? "#ffd7d7" : "#f0c9a4";
    ctx.beginPath();
    ctx.arc(gx, hy, 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = f.trunk;
    fillRR(gx - 26, GROUND - 236, 52, 12, 6);
    ctx.fillStyle = "#27313f";
    ctx.beginPath();
    ctx.arc(gx + f.facing * 9, GROUND - 212, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawFighter(f) {
  ctx.save();
  const gx = f.baseX + f.knock;
  if (f.ko) {
    const a = f.koAnim * 1.25 * -f.facing;
    ctx.translate(gx, GROUND);
    ctx.rotate(a);
    ctx.translate(-gx, -GROUND);
  } else if (f.flash > 0) {
    ctx.translate((Math.random() - 0.5) * 6, 0);
  }

  ctx.fillStyle = "rgba(0,0,0,.12)";
  ctx.beginPath();
  ctx.ellipse(gx, GROUND + 4, 50, 12, 0, 0, Math.PI * 2);
  ctx.fill();

  const skin = "#f0c9a4";
  const hit = f.flash > 0;

  ctx.fillStyle = skin;
  fillRR(gx - 26, GROUND - 70, 20, 70, 8);
  fillRR(gx + 6, GROUND - 70, 20, 70, 8);

  ctx.fillStyle = f.trunk;
  fillRR(gx - 32, GROUND - 112, 64, 52, 12);
  ctx.fillStyle = "rgba(255,255,255,.5)";
  fillRR(gx - 4, GROUND - 110, 8, 48, 3);

  ctx.fillStyle = hit ? "#ffd7d7" : skin;
  fillRR(gx - 30, GROUND - 188, 60, 80, 16);

  drawHead(f, gx, hit);

  drawGloves(f, gx);
  ctx.restore();
}

function drawGloves(f, gx) {
  const reachX = f.baseX + (f.opp.baseX - f.baseX) * 0.58;
  for (const g of ["L", "R"]) {
    const front = g === "R";
    let bx = gx + f.facing * (front ? 42 : 14);
    let by = GROUND - (front ? 150 : 120);
    if (f.guard) {
      bx = gx + f.facing * (front ? 30 : 18);
      by = GROUND - (front ? 176 : 162);
    }
    const p = f.punch[g];
    if (p.t > 0) {
      const ext = Math.sin(((PUNCH_DUR - p.t) / PUNCH_DUR) * Math.PI);
      const restX = gx + f.facing * 42;
      bx = restX + (reachX - restX) * ext;
      by = GROUND - 150;
    } else if (f.isCPU && f.ai.state === "telegraph" && front) {
      bx = gx - f.facing * 10;
      by = GROUND - 168;
    }
    ctx.fillStyle = f.trunk;
    ctx.beginPath();
    ctx.arc(bx, by, 16, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,.35)";
    ctx.beginPath();
    ctx.arc(bx - f.facing * 4, by - 5, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,.18)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(bx, by, 16, 0, Math.PI * 2);
    ctx.stroke();
    if (f.blockFlash > 0 && front) drawBlockSpark(bx, by, f.blockFlash / 12);
  }
}

function drawBlockSpark(x, y, a) {
  ctx.save();
  ctx.globalAlpha = a;
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 3;
  for (let i = 0; i < 6; i++) {
    const ang = (i / 6) * Math.PI * 2;
    const r0 = 14, r1 = 14 + 16 * a;
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(ang) * r0, y + Math.sin(ang) * r0);
    ctx.lineTo(x + Math.cos(ang) * r1, y + Math.sin(ang) * r1);
    ctx.stroke();
  }
  ctx.fillStyle = "#fff7e6";
  ctx.beginPath();
  ctx.arc(x, y, 7 * a, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function comic(size) { return "bold " + size + "px Impact, 'Arial Black', sans-serif"; }

function centerText(title, sub, color) {
  ctx.save();
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(255,255,255,.55)";
  ctx.fillRect(0, H / 2 - 90, W, 170);
  ctx.fillStyle = color;
  ctx.font = comic(76);
  ctx.fillText(title, W / 2, H / 2);
  if (sub) {
    ctx.fillStyle = "#374151";
    ctx.font = comic(26);
    ctx.fillText(sub, W / 2, H / 2 + 48);
  }
  ctx.restore();
}

function drawOverlay() {
  if (phase === "round_intro") centerText("ROUND " + round, "GET YOUR GUARD UP", "#f59e0b");
  else if (phase === "fight" && fightFlash > 0) centerText("FIGHT!", null, "#ef4444");
  else if (phase === "ko") centerText("K.O.!", lastWinner.name + " LANDS IT", "#ef4444");
  else if (phase === "round_over") centerText("ROUND " + round, lastWinner.name + " WINS THE ROUND", "#2563eb");
  else if (phase === "match_over") centerText(matchWinner.name + " WINS!", "next match starting… (R now)", "#f59e0b");
}

function render() {
  drawScene();
  drawFighter(P1);
  drawFighter(P2);
  drawOverlay();

  if (!wsReady) { elStatus.textContent = "linking to tracker…"; }
  else if (!camReady) { elStatus.textContent = "enable camera or use keys"; }
  else {
    const n = (slots.L.present ? 1 : 0) + (slots.R.present ? 1 : 0);
    elStatus.textContent = n === 0 ? "show your hands" : n + (n === 1 ? " hand tracked" : " hands tracked");
  }
}

function slotColor(s) {
  if (s.blocking) return "#3b82f6";
  if (s.punchFlash > 0) return "#ef4444";
  return "#cbd5e1";
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
      const mx = s.dx * pip.width, my = s.dy * pip.height;
      pctx.fillStyle = slotColor(s);
      pctx.beginPath();
      pctx.arc(mx, my, 10, 0, Math.PI * 2);
      pctx.fill();
      pctx.strokeStyle = "rgba(255,255,255,.9)";
      pctx.lineWidth = 2;
      pctx.stroke();
    }
  } else {
    pctx.restore();
    pctx.fillStyle = "#0b1020";
    pctx.fillRect(0, 0, pip.width, pip.height);
    pctx.fillStyle = "#9aa3c0";
    pctx.font = "13px Trebuchet MS, sans-serif";
    pctx.textAlign = "center";
    pctx.fillText("camera off — keyboard play", pip.width / 2, pip.height / 2);
  }
  pctx.strokeStyle = camReady ? "rgba(255,255,255,.6)" : "#2a3350";
  pctx.setLineDash([6, 6]);
  pctx.lineWidth = 2;
  pctx.beginPath();
  pctx.moveTo(pip.width / 2, 0);
  pctx.lineTo(pip.width / 2, pip.height);
  pctx.stroke();
  pctx.setLineDash([]);
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
  Ring.ensure();
}

window.addEventListener("keydown", (e) => {
  unlockAudio();
  const k = e.key.toLowerCase();
  if (k === "r") { restart(); return; }
  if (phase !== "fight") return;
  if (k === " " || e.key === "Spacebar") { keys.p1block = true; e.preventDefault(); return; }
  if (k === "enter") { if (mode === "pvp") keys.p2block = true; return; }
  if (e.repeat) return;
  if (k === "f") fighterThrow(P1, "L");
  else if (k === "g") fighterThrow(P1, "R");
  else if (k === "j") { if (mode === "pvp") fighterThrow(P2, "L"); }
  else if (k === "k") { if (mode === "pvp") fighterThrow(P2, "R"); }
});

window.addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (k === " " || e.key === "Spacebar") keys.p1block = false;
  else if (k === "enter") keys.p2block = false;
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

updateLife();
updatePips();
connectWS();
requestAnimationFrame(frame);
