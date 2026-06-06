const game = document.getElementById("game");
const ctx = game.getContext("2d");
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");

const elScore = document.getElementById("score");
const elTime = document.getElementById("time");
const elStatus = document.getElementById("status");
const elAction = document.getElementById("action");
const overlay = document.getElementById("overlay");
const overlayTitle = document.getElementById("overlay-title");
const overlayText = document.getElementById("overlay-text");
const startBtn = document.getElementById("start");
const muteBtn = document.getElementById("mute");
const fsBtn = document.getElementById("fs");
const speedRange = document.getElementById("speed-range");
const speedName = document.getElementById("speed-name");
const fsTarget = document.querySelector(".stage");
const SPEED_NAMES = ["", "Slow", "Normal", "Fast", "Insane"];

const W = game.width;
const H = game.height;
const HORIZON = 176;
const PLAYER_T = 0.9;
const LANES = [-1, 0, 1];
const JUMP_THRESH = 0.05;
const DUCK_THRESH = 0.07;
const SPEED_MUL = [0, 0.2, 0.45, 0.85, 1.5];
let speedLevel = 2;

let phase = "ready";
let steer = 0.5;
let bodyY = 0.5;
let bodyYSmooth = 0.5;
let baselineY = 0.5;
let bodyPresent = false;
let camReady = false;
let wsReady = false;
let lastFace = null;
let faceImg = null;
let hasFace = false;

let ws = null;
let awaiting = false;
let lastSend = 0;

let obstacles = [];
let props = [];
let spawnTimer = 0;
let propTimer = 0;
let speed = 0;
let score = 0;
let startTime = 0;
let elapsed = 0;
let worldScroll = 0;
let frameCount = 0;
let restartAt = 0;
let trexT = 1.45;
let calStart = 0;
let calSamples = [];
let action = "RUN";

const player = { laneX: 0, jumpY: 0, vy: 0, grounded: true, ducking: false };
const KB = { left: false, right: false, jump: false, duck: false };
let lastKbDown = -9999;
let usingKb = false;
const LANE_GAIN = 7.5;
const LANE_DEAD = 0.03;

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function projY(t) { return HORIZON + (H - HORIZON) * t; }
function projHalf(t) { return 16 + 252 * Math.pow(t, 1.62); }
function laneToX(lane, t) { return W / 2 + lane * projHalf(t) * 0.52; }
function edgeX(side, t) { return W / 2 + side * projHalf(t) * 1.16; }

function connectWS() {
  try {
    ws = new WebSocket("ws://" + location.hostname + ":8765");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsReady = true; };
    ws.onmessage = (e) => {
      awaiting = false;
      try {
        const m = JSON.parse(e.data);
        bodyPresent = !!m.present;
        if (m.present) {
          if (typeof m.x === "number") steer = m.x;
          if (typeof m.y === "number") bodyY = m.y;
          if (m.face) lastFace = m.face;
        }
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

function captureFace() {
  if (!lastFace || !camReady) return;
  const vw = cam.videoWidth, vh = cam.videoHeight;
  if (!vw || !vh) return;
  const sx = clamp(lastFace.x, 0, 1) * vw;
  const sy = clamp(lastFace.y, 0, 1) * vh;
  const sw = clamp(lastFace.w, 0.05, 1) * vw;
  const sh = clamp(lastFace.h, 0.05, 1) * vh;
  const fc = document.createElement("canvas");
  fc.width = 180; fc.height = 180;
  const fx = fc.getContext("2d");
  fx.drawImage(cam, sx, sy, sw, sh, 0, 0, 180, 180);
  faceImg = fc;
  hasFace = true;
}

function beginCalibration() {
  phase = "calibrating";
  calStart = performance.now();
  calSamples = [];
  startBtn.style.display = "none";
  overlay.classList.add("hidden");
}

function startRun() {
  obstacles = [];
  props = [];
  spawnTimer = 46;
  propTimer = 10;
  speed = 0.0058;
  score = 0;
  player.laneX = 0; player.jumpY = 0; player.vy = 0; player.grounded = true; player.ducking = false;
  startTime = performance.now();
  elapsed = 0;
  trexT = 1.45;
  phase = "playing";
  overlay.classList.add("hidden");
}

function crash(ts) {
  phase = "dead";
  restartAt = ts + 3000;
  trexT = 1.3;
  action = "CAUGHT";
  overlayTitle.textContent = "CAUGHT!";
  overlayText.textContent = "Survived " + elapsed.toFixed(1) + "s and scored " + score + ". The T-Rex got you — restarting in 3s.";
  startBtn.style.display = "none";
  overlay.classList.remove("hidden");
  if (window.Jungle) Jungle.roar();
}

const OBSTACLES = [
  { kind: "rock", avoid: "jump", w: 0.18 },
  { kind: "raptor", avoid: "jump", w: 0.22 },
  { kind: "ptero", avoid: "duck", w: 0.28 },
  { kind: "tree", avoid: "move", w: 0.14 },
  { kind: "stego", avoid: "move", w: 0.18 },
];

function spawnObstacle() {
  let r = Math.random(), pick = OBSTACLES[0];
  for (const o of OBSTACLES) { if (r < o.w) { pick = o; break; } r -= o.w; }
  const lane = LANES[(Math.random() * 3) | 0];
  obstacles.push({ kind: pick.kind, avoid: pick.avoid, lane, t: 0, resolved: false, cleared: false, seed: Math.random() * 6.28 });
}

function spawnProp() {
  props.push({ t: 0, side: Math.random() < 0.5 ? -1 : 1, off: 0.18 + Math.random() * 0.6, kind: Math.random() < 0.55 ? "fern" : "bush", seed: Math.random() * 6.28 });
}

function readControls(ts) {
  const kbHeld = KB.left || KB.right || KB.jump || KB.duck;
  usingKb = kbHeld || ts - lastKbDown < 200;
  let targetLane = 0, wantJump = false, wantDuck = false;
  if (usingKb) {
    targetLane = (KB.left ? -1 : 0) + (KB.right ? 1 : 0);
    wantJump = KB.jump;
    wantDuck = KB.duck;
  } else if (bodyPresent) {
    let d = steer - 0.5;
    d = Math.sign(d) * Math.max(0, Math.abs(d) - LANE_DEAD);
    targetLane = clamp(d * LANE_GAIN, -1, 1);
    bodyYSmooth = lerp(bodyYSmooth, bodyY, 0.4);
    wantJump = baselineY - bodyYSmooth > JUMP_THRESH;
    wantDuck = bodyYSmooth - baselineY > DUCK_THRESH;
  }
  return { targetLane, wantJump, wantDuck };
}

function updateCalibration(ts) {
  if (bodyPresent) calSamples.push(bodyY);
  if ((ts - calStart) / 1000 >= 3) {
    baselineY = calSamples.length > 4 ? calSamples.reduce((a, b) => a + b, 0) / calSamples.length : 0.5;
    bodyYSmooth = baselineY;
    captureFace();
    startRun();
  }
}

function update(ts) {
  frameCount++;
  if (phase === "calibrating") { updateCalibration(ts); return; }
  if (phase === "dead") {
    trexT = lerp(trexT, 0.99, 0.05);
    worldScroll += 2;
    if (ts >= restartAt) startRun();
    return;
  }
  if (phase !== "playing") return;

  elapsed = (ts - startTime) / 1000;
  speed = Math.min(0.017, 0.0058 + elapsed * 0.00009) * SPEED_MUL[speedLevel];
  worldScroll += speed * 620;

  const c = readControls(ts);
  player.laneX = lerp(player.laneX, c.targetLane, 0.22);
  if (c.wantJump && player.grounded) { player.vy = 13.0; player.grounded = false; if (window.Jungle) Jungle.jump(); }
  if (!player.grounded) {
    player.jumpY += player.vy;
    player.vy -= 0.46;
    if (player.jumpY <= 0) { player.jumpY = 0; player.vy = 0; player.grounded = true; }
  }
  const wasDucking = player.ducking;
  player.ducking = c.wantDuck && player.grounded;
  if (player.ducking && !wasDucking && window.Jungle) Jungle.duck();

  action = !player.grounded ? "JUMP" : player.ducking ? "DUCK" : player.laneX < -0.4 ? "LEFT" : player.laneX > 0.4 ? "RIGHT" : "RUN";

  spawnTimer -= 1;
  if (spawnTimer <= 0) { spawnObstacle(); spawnTimer = Math.max(24, 64 - elapsed * 1.1); }
  propTimer -= 1;
  if (propTimer <= 0) { spawnProp(); propTimer = 12 + Math.random() * 16; }

  for (const o of obstacles) {
    o.t += speed;
    if (!o.resolved && o.t >= PLAYER_T) {
      o.resolved = true;
      const sameCol = Math.abs(player.laneX - o.lane) < 0.6;
      let safe = !sameCol;
      if (!safe) {
        if (o.avoid === "jump") safe = player.jumpY > 14;
        else if (o.avoid === "duck") safe = player.ducking;
        else safe = false;
      }
      if (safe) { score += 10; o.cleared = true; }
      else { crash(ts); break; }
    }
  }
  obstacles = obstacles.filter((o) => o.t < 1.12);
  for (const p of props) p.t += speed;
  props = props.filter((p) => p.t < 1.14);
  if (frameCount % 6 === 0) score += 1;
}

function drawVolcano(cx) {
  ctx.fillStyle = "#241a22";
  ctx.beginPath();
  ctx.moveTo(cx - 120, HORIZON);
  ctx.lineTo(cx - 28, HORIZON - 96);
  ctx.lineTo(cx - 10, HORIZON - 80);
  ctx.lineTo(cx + 8, HORIZON - 98);
  ctx.lineTo(cx + 120, HORIZON);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "rgba(244,120,60,0.8)";
  ctx.beginPath();
  ctx.moveTo(cx - 14, HORIZON - 92);
  ctx.lineTo(cx + 10, HORIZON - 94);
  ctx.lineTo(cx + 4, HORIZON - 72);
  ctx.lineTo(cx - 8, HORIZON - 72);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "rgba(120,110,120,0.35)";
  for (let i = 0; i < 3; i++) {
    const yy = HORIZON - 110 - i * 26 + Math.sin(worldScroll * 0.002 + i) * 6;
    ctx.beginPath();
    ctx.arc(cx - 2 + i * 9, yy, 14 + i * 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawTreeline(par) {
  ctx.fillStyle = "#10241a";
  let x = -40 + (par % 64);
  while (x < W + 40) {
    const h = 40 + ((x * 37) % 30);
    ctx.beginPath();
    ctx.moveTo(x - 30, HORIZON + 4);
    ctx.lineTo(x, HORIZON - h);
    ctx.lineTo(x + 30, HORIZON + 4);
    ctx.closePath();
    ctx.fill();
    x += 48;
  }
}

function drawSauropod(cx, cy, sc, dir) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(dir, 1);
  ctx.fillStyle = "#13251a";
  ctx.beginPath();
  ctx.ellipse(0, -18 * sc, 34 * sc, 18 * sc, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(-26 * sc, -22 * sc);
  ctx.quadraticCurveTo(-72 * sc, -14 * sc, -98 * sc, 0);
  ctx.quadraticCurveTo(-66 * sc, -4 * sc, -22 * sc, -10 * sc);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(18 * sc, -28 * sc);
  ctx.quadraticCurveTo(46 * sc, -62 * sc, 50 * sc, -86 * sc);
  ctx.lineTo(60 * sc, -84 * sc);
  ctx.quadraticCurveTo(56 * sc, -56 * sc, 32 * sc, -22 * sc);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(58 * sc, -88 * sc, 9 * sc, 6 * sc, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(-18 * sc, -6 * sc, 8 * sc, 16 * sc);
  ctx.fillRect(-2 * sc, -6 * sc, 8 * sc, 16 * sc);
  ctx.fillRect(16 * sc, -6 * sc, 8 * sc, 16 * sc);
  ctx.restore();
}

function drawBackground() {
  const sky = ctx.createLinearGradient(0, 0, 0, HORIZON + 70);
  sky.addColorStop(0, "#16243a");
  sky.addColorStop(0.55, "#41524a");
  sky.addColorStop(1, "#d8a85c");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, HORIZON + 70);

  ctx.fillStyle = "rgba(255, 224, 168, 0.55)";
  ctx.beginPath();
  ctx.arc(W * 0.32, HORIZON - 46, 40, 0, Math.PI * 2);
  ctx.fill();

  const par = -player.laneX * 22;
  drawVolcano(W * 0.72 + par * 0.5);
  const d1 = (frameCount * 0.35) % (W + 260) - 130;
  drawSauropod(d1 + par * 0.4, HORIZON - 4, 0.5, 1);
  const d2 = W + 130 - ((frameCount * 0.22) % (W + 260));
  drawSauropod(d2 + par * 0.4, HORIZON - 1, 0.7, -1);
  drawTreeline(par);
}

function drawPath() {
  ctx.fillStyle = "#163021";
  ctx.fillRect(0, HORIZON, W, H - HORIZON);

  ctx.beginPath();
  ctx.moveTo(edgeX(-1, 0), projY(0));
  for (let t = 0; t <= 1.0001; t += 0.05) ctx.lineTo(edgeX(-1, t), projY(t));
  for (let t = 1; t >= 0; t -= 0.05) ctx.lineTo(edgeX(1, t), projY(t));
  ctx.closePath();
  const g = ctx.createLinearGradient(0, HORIZON, 0, H);
  g.addColorStop(0, "#5c4730");
  g.addColorStop(1, "#977244");
  ctx.fillStyle = g;
  ctx.fill();

  const period = 0.11;
  const off = ((worldScroll * 0.0009) % period + period) % period;
  ctx.strokeStyle = "rgba(60,42,22,0.45)";
  for (let t = off; t <= 1; t += period) {
    ctx.beginPath();
    ctx.lineWidth = 1 + t * 6;
    ctx.moveTo(edgeX(-1, t), projY(t));
    ctx.lineTo(edgeX(1, t), projY(t));
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,240,200,0.16)";
  for (const d of [-0.5, 0.5]) {
    ctx.beginPath();
    ctx.lineWidth = 2;
    let on = true;
    for (let t = 0; t <= 1; t += 0.04) {
      const px = laneToX(d * 2, t), py = projY(t);
      if (on) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      on = !on;
    }
    ctx.stroke();
  }
}

function drawProp(p) {
  const x = edgeX(p.side, p.t) + p.side * p.off * projHalf(p.t);
  const y = projY(p.t);
  const s = projHalf(p.t) / projHalf(1) * 1.1;
  if (s < 0.02) return;
  if (p.kind === "bush") {
    ctx.fillStyle = "#1f5a30";
    for (let i = -1; i <= 1; i++) {
      ctx.beginPath();
      ctx.arc(x + i * 16 * s, y - 12 * s, 18 * s, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#2a7340";
    ctx.beginPath();
    ctx.arc(x, y - 26 * s, 16 * s, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.strokeStyle = "#2c8a45";
    ctx.lineWidth = 3 * s;
    for (let i = -2; i <= 2; i++) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.quadraticCurveTo(x + i * 14 * s, y - 44 * s, x + i * 26 * s, y - 70 * s);
      ctx.stroke();
    }
  }
}

function drawRock(x, y, s, seed) {
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.3)";
  ctx.beginPath();
  ctx.ellipse(x, y + 4 * s, 30 * s, 9 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#7d7a78";
  ctx.beginPath();
  ctx.moveTo(x - 30 * s, y);
  ctx.lineTo(x - 18 * s, y - 34 * s);
  ctx.lineTo(x + 6 * s, y - 40 * s);
  ctx.lineTo(x + 28 * s, y - 22 * s);
  ctx.lineTo(x + 30 * s, y);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#9a9794";
  ctx.beginPath();
  ctx.moveTo(x - 18 * s, y - 34 * s);
  ctx.lineTo(x + 6 * s, y - 40 * s);
  ctx.lineTo(x - 2 * s, y - 18 * s);
  ctx.lineTo(x - 22 * s, y - 14 * s);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawTree(x, y, s) {
  ctx.fillStyle = "#5a3c1f";
  ctx.fillRect(x - 9 * s, y - 78 * s, 18 * s, 78 * s);
  ctx.fillStyle = "#1f6b35";
  for (const o of [[-22, -70], [22, -74], [0, -96]]) {
    ctx.beginPath();
    ctx.arc(x + o[0] * s, y + o[1] * s, 30 * s, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#2c8a45";
  ctx.beginPath();
  ctx.arc(x, y - 86 * s, 22 * s, 0, Math.PI * 2);
  ctx.fill();
}

function drawPtero(x, y, s, seed) {
  const flap = Math.sin(worldScroll * 0.02 + seed) * 0.5;
  y -= 86 * s;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.18)";
  ctx.beginPath();
  ctx.ellipse(x, projY(PLAYER_T) + 2, 26 * s, 7 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#3a2c3a";
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.quadraticCurveTo(x - 30 * s, y - (20 + flap * 30) * s, x - 52 * s, y + 4 * s);
  ctx.quadraticCurveTo(x - 26 * s, y + 8 * s, x, y + 4 * s);
  ctx.quadraticCurveTo(x + 26 * s, y + 8 * s, x + 52 * s, y + 4 * s);
  ctx.quadraticCurveTo(x + 30 * s, y - (20 + flap * 30) * s, x, y);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#4a3a4a";
  ctx.beginPath();
  ctx.ellipse(x, y + 6 * s, 8 * s, 11 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x + 6 * s, y + 2 * s);
  ctx.lineTo(x + 22 * s, y - 2 * s);
  ctx.lineTo(x + 7 * s, y + 8 * s);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawRaptor(x, y, s, seed) {
  const lsw = Math.sin(worldScroll * 0.06 + seed) * 7 * s;
  const top = y - 60 * s;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.3)";
  ctx.beginPath(); ctx.ellipse(x, y + 2, 24 * s, 7 * s, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#5f8336";
  ctx.beginPath();
  ctx.moveTo(x - 6 * s, y - 34 * s);
  ctx.quadraticCurveTo(x - 42 * s, y - 50 * s, x - 56 * s, y - 66 * s);
  ctx.quadraticCurveTo(x - 30 * s, y - 40 * s, x + 2 * s, y - 28 * s);
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "#4d6b2b"; ctx.lineWidth = 6 * s; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(x - 5 * s, y - 30 * s); ctx.lineTo(x - 5 * s - lsw, y - 12 * s); ctx.lineTo(x - 9 * s - lsw, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x + 5 * s, y - 30 * s); ctx.lineTo(x + 5 * s + lsw, y - 12 * s); ctx.lineTo(x + 9 * s + lsw, y); ctx.stroke();
  ctx.fillStyle = "#6b8f3a";
  ctx.beginPath(); ctx.ellipse(x, y - 40 * s, 16 * s, 22 * s, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#cdbb7a";
  ctx.beginPath(); ctx.ellipse(x, y - 34 * s, 9 * s, 14 * s, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#5f8336"; ctx.lineWidth = 3.5 * s;
  ctx.beginPath(); ctx.moveTo(x - 8 * s, y - 46 * s); ctx.lineTo(x - 16 * s, y - 38 * s); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x + 8 * s, y - 46 * s); ctx.lineTo(x + 16 * s, y - 38 * s); ctx.stroke();
  ctx.fillStyle = "#6b8f3a";
  ctx.beginPath(); ctx.ellipse(x, top + 8 * s, 12 * s, 13 * s, 0, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x - 4 * s, top + 10 * s);
  ctx.lineTo(x + 19 * s, top + 14 * s);
  ctx.lineTo(x - 2 * s, top + 18 * s);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#fff";
  for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(x + (3 + i * 5) * s, top + 14 * s); ctx.lineTo(x + (5 + i * 5) * s, top + 17 * s); ctx.lineTo(x + (7 + i * 5) * s, top + 14 * s); ctx.closePath(); ctx.fill(); }
  ctx.fillStyle = "#f4d03c"; ctx.beginPath(); ctx.arc(x + 2 * s, top + 6 * s, 3.2 * s, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#1a1a1a"; ctx.beginPath(); ctx.arc(x + 3 * s, top + 6 * s, 1.6 * s, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawStego(x, y, s) {
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.3)";
  ctx.beginPath(); ctx.ellipse(x, y + 2, 38 * s, 8 * s, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#7a6a45";
  ctx.beginPath();
  ctx.moveTo(x - 40 * s, y - 20 * s);
  ctx.quadraticCurveTo(x - 20 * s, y - 48 * s, x + 6 * s, y - 46 * s);
  ctx.quadraticCurveTo(x + 40 * s, y - 44 * s, x + 46 * s, y - 18 * s);
  ctx.quadraticCurveTo(x + 20 * s, y - 8 * s, x - 20 * s, y - 10 * s);
  ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.ellipse(x + 50 * s, y - 16 * s, 10 * s, 8 * s, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#6a5a3a";
  ctx.fillRect(x - 26 * s, y - 14 * s, 9 * s, 16 * s);
  ctx.fillRect(x - 6 * s, y - 14 * s, 9 * s, 16 * s);
  ctx.fillRect(x + 14 * s, y - 14 * s, 9 * s, 16 * s);
  ctx.fillRect(x + 30 * s, y - 14 * s, 9 * s, 16 * s);
  ctx.fillStyle = "#3f8a5a";
  for (const p of [[-26, -38], [-12, -48], [2, -52], [16, -48], [30, -40]]) {
    ctx.beginPath();
    ctx.moveTo(x + (p[0] - 8) * s, y + (p[1] + 14) * s);
    ctx.lineTo(x + p[0] * s, y + p[1] * s);
    ctx.lineTo(x + (p[0] + 8) * s, y + (p[1] + 14) * s);
    ctx.closePath(); ctx.fill();
  }
  ctx.fillStyle = "#cfcfcf";
  for (const dx of [-50, -56]) { ctx.beginPath(); ctx.moveTo(x + dx * s, y - 22 * s); ctx.lineTo(x + (dx - 4) * s, y - 33 * s); ctx.lineTo(x + (dx + 3) * s, y - 25 * s); ctx.closePath(); ctx.fill(); }
  ctx.restore();
}

function drawRex(cx, groundY, sc, open, runP) {
  const col = "#5a8a3a", colD = "#3f6b2a", colDk = "#2c4f22", belly = "#cdbb84";
  const ns = Math.sin(runP), fs = Math.sin(runP + Math.PI);
  ctx.save();
  ctx.translate(cx, groundY);
  ctx.scale(sc, sc);

  ctx.fillStyle = colD;
  ctx.beginPath();
  ctx.moveTo(-20, -150);
  ctx.quadraticCurveTo(-150, -158, -215, -92);
  ctx.quadraticCurveTo(-150, -118, -18, -120);
  ctx.closePath(); ctx.fill();

  ctx.fillStyle = colDk;
  ctx.beginPath(); ctx.ellipse(15, -118, 33, 47, -0.2, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = 17; ctx.strokeStyle = colDk; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath(); ctx.moveTo(14, -92); ctx.lineTo(-4 + fs * 10, -42); ctx.lineTo(24 + fs * 16, -8); ctx.stroke();

  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(-22, -150);
  ctx.quadraticCurveTo(30, -218, 120, -196);
  ctx.quadraticCurveTo(158, -186, 156, -150);
  ctx.quadraticCurveTo(146, -110, 95, -96);
  ctx.quadraticCurveTo(35, -84, -6, -112);
  ctx.quadraticCurveTo(-32, -126, -22, -150);
  ctx.closePath(); ctx.fill();

  ctx.fillStyle = belly;
  ctx.beginPath();
  ctx.moveTo(100, -100); ctx.quadraticCurveTo(40, -86, 2, -110);
  ctx.quadraticCurveTo(60, -104, 118, -118); ctx.closePath(); ctx.fill();

  ctx.fillStyle = col;
  ctx.beginPath(); ctx.ellipse(62, -112, 42, 58, -0.12, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = colD; ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath(); ctx.moveTo(64, -76); ctx.lineTo(48 + ns * 10, -30); ctx.lineTo(92 + ns * 18, -4); ctx.stroke();
  ctx.fillStyle = colDk;
  ctx.beginPath(); ctx.moveTo(74 + ns * 18, -6); ctx.lineTo(118 + ns * 18, -6); ctx.lineTo(114 + ns * 18, 7); ctx.lineTo(78 + ns * 18, 7); ctx.closePath(); ctx.fill();

  ctx.strokeStyle = colD; ctx.lineWidth = 9;
  ctx.beginPath(); ctx.moveTo(126, -150); ctx.lineTo(140, -128); ctx.lineTo(136, -110); ctx.stroke();
  ctx.strokeStyle = "#e8e0c0"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(136, -110); ctx.lineTo(132, -101); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(136, -110); ctx.lineTo(142, -103); ctx.stroke();

  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(120, -190); ctx.quadraticCurveTo(150, -210, 188, -208);
  ctx.lineTo(188, -168); ctx.quadraticCurveTo(150, -162, 128, -162); ctx.closePath(); ctx.fill();

  const hx = 196, hy = -212;
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.moveTo(hx - 18, hy + 44);
  ctx.quadraticCurveTo(hx - 26, hy - 6, hx + 26, hy - 8);
  ctx.quadraticCurveTo(hx + 86, hy - 12, hx + 128, hy + 16);
  ctx.lineTo(hx + 134, hy + 32);
  ctx.quadraticCurveTo(hx + 96, hy + 40, hx + 40, hy + 44);
  ctx.lineTo(hx - 18, hy + 44);
  ctx.closePath(); ctx.fill();

  ctx.fillStyle = "#3a1820";
  ctx.beginPath();
  ctx.moveTo(hx - 12, hy + 46);
  ctx.lineTo(hx + 130, hy + 33);
  ctx.lineTo(hx + 122, hy + 36 + open);
  ctx.lineTo(hx - 12, hy + 50);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#9c4a50";
  ctx.beginPath(); ctx.ellipse(hx + 66, hy + 44 + open * 0.62, 30, 9, -0.08, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = colD;
  ctx.beginPath();
  ctx.moveTo(hx - 16, hy + 48);
  ctx.lineTo(hx + 122, hy + 36 + open);
  ctx.lineTo(hx + 114, hy + 56 + open);
  ctx.quadraticCurveTo(hx + 30, hy + 70 + open, hx - 14, hy + 60 + open);
  ctx.closePath(); ctx.fill();

  ctx.fillStyle = "#f4efe0";
  for (let i = 0; i < 8; i++) {
    const t = 0.18 + i * 0.1, tx = hx - 12 + t * 142, ty = hy + 45 - 12 * t;
    ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx + 4, ty + 11); ctx.lineTo(tx + 8, ty); ctx.closePath(); ctx.fill();
  }
  for (let i = 0; i < 7; i++) {
    const t = 0.28 + i * 0.1, tx = hx - 16 + t * 138, ty = hy + 48 + (open - 12) * t;
    ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx + 4, ty - 10); ctx.lineTo(tx + 8, ty); ctx.closePath(); ctx.fill();
  }

  ctx.fillStyle = "#f4d03c"; ctx.beginPath(); ctx.arc(hx + 60, hy + 10, 11, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#1a1a1a"; ctx.beginPath(); ctx.ellipse(hx + 62, hy + 10, 4, 7, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = colDk; ctx.beginPath(); ctx.moveTo(hx + 44, hy - 2); ctx.lineTo(hx + 78, hy + 2); ctx.lineTo(hx + 72, hy + 10); ctx.lineTo(hx + 48, hy + 8); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.ellipse(hx + 120, hy + 22, 4, 3, 0, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawChaser() {
  const f = frameCount;
  const surge = Math.max(0, Math.sin(f * 0.025));
  const lunge = surge * surge;
  const x = W / 2 - player.laneX * 30 + Math.sin(f * 0.04) * 26;
  const groundY = H + 98 - lunge * 150 + Math.sin(f * 0.3) * 4;
  const open = 26 + lunge * 40;
  drawRex(x, groundY, 0.66, open, f * 0.3);
}

function drawObstacle(o) {
  const x = laneToX(o.lane, o.t);
  const y = projY(o.t);
  const s = projHalf(o.t) / projHalf(PLAYER_T);
  if (s < 0.02) return;
  if (o.kind === "rock") drawRock(x, y, s, o.seed);
  else if (o.kind === "tree") drawTree(x, y, s);
  else if (o.kind === "raptor") drawRaptor(x, y, s, o.seed);
  else if (o.kind === "stego") drawStego(x, y, s);
  else drawPtero(x, y, s, o.seed);
}

function drawFace(x, y, r) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  if (hasFace && faceImg) {
    ctx.drawImage(faceImg, x - r, y - r, r * 2, r * 2);
  } else {
    ctx.fillStyle = "#d8a878";
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
    ctx.fillStyle = "#3a2a1a";
    ctx.fillRect(x - r, y - r, r * 2, r * 0.5);
    ctx.fillStyle = "#26190f";
    ctx.beginPath(); ctx.arc(x - r * 0.36, y, r * 0.12, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x + r * 0.36, y, r * 0.12, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
  ctx.strokeStyle = "#1c2a1c";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
}

function drawRunner() {
  const t = PLAYER_T;
  const x = laneToX(player.laneX, t);
  const ground = projY(t);
  const s = projHalf(t) / projHalf(1) * 1.25;
  const lift = player.jumpY * 0.95;
  const duck = player.ducking;
  const run = player.grounded && !duck;
  const airborne = !player.grounded;
  const hipW = 5 * s, shW = 11 * s;
  const stride = frameCount * 0.4;

  let bodyH = 40 * s, legH = 26 * s, headR = 15 * s;
  if (duck) { bodyH *= 0.5; legH *= 0.6; }

  const shScale = clamp(1 - lift / 130, 0.45, 1);
  ctx.fillStyle = "rgba(0,0,0,0.32)";
  ctx.beginPath();
  ctx.ellipse(x, ground + 3, 24 * s * shScale, 7 * s * shScale, 0, 0, Math.PI * 2);
  ctx.fill();

  const feet = ground - lift;
  const hip = feet - legH;
  const shoulder = hip - bodyH;
  const head = shoulder - headR * 0.6;

  ctx.lineCap = "round";
  ctx.strokeStyle = "#caa06a";
  ctx.lineWidth = 7 * s;
  for (const side of [-1, 1]) {
    const sw = run ? Math.sin(stride + (side > 0 ? Math.PI : 0)) : 0;
    const hipX = x + side * hipW;
    let footX, footY;
    if (airborne) { footX = hipX + side * 4 * s; footY = feet - 15 * s; }
    else if (duck) { footX = hipX + side * 9 * s; footY = feet; }
    else { footX = hipX + sw * 7 * s; footY = feet - Math.max(0, sw) * 10 * s; }
    const kneeX = (hipX + footX) / 2 + side * 2 * s;
    const kneeY = (hip + footY) / 2 + (duck ? 6 * s : 0);
    ctx.beginPath();
    ctx.moveTo(hipX, hip);
    ctx.lineTo(kneeX, kneeY);
    ctx.lineTo(footX, footY);
    ctx.stroke();
  }

  ctx.fillStyle = "#2f7d3f";
  ctx.beginPath();
  ctx.moveTo(x - 13 * s, shoulder);
  ctx.lineTo(x + 13 * s, shoulder);
  ctx.lineTo(x + 9 * s, hip);
  ctx.lineTo(x - 9 * s, hip);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#1f5a2c";
  for (let i = 0; i < 3; i++) {
    ctx.beginPath();
    ctx.arc(x - 6 * s + i * 6 * s, shoulder + (6 + i * 9) * s, 2.4 * s, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = "#caa06a";
  ctx.lineWidth = 5.5 * s;
  for (const side of [-1, 1]) {
    const sw = run ? Math.sin(stride + (side > 0 ? 0 : Math.PI)) : 0;
    const shX = x + side * shW;
    let handX, handY;
    if (airborne) { handX = shX + side * 7 * s; handY = shoulder - 6 * s; }
    else { handX = shX + sw * 4 * s; handY = shoulder + 20 * s - Math.max(0, sw) * 6 * s; }
    const elbowX = (shX + handX) / 2 + side * 3 * s;
    const elbowY = shoulder + 11 * s;
    ctx.beginPath();
    ctx.moveTo(shX, shoulder + 4 * s);
    ctx.lineTo(elbowX, elbowY);
    ctx.lineTo(handX, handY);
    ctx.stroke();
  }

  ctx.strokeStyle = "#caa06a";
  ctx.lineWidth = 5 * s;
  ctx.beginPath(); ctx.moveTo(x, shoulder); ctx.lineTo(x, head + headR * 0.7); ctx.stroke();

  drawFace(x, head, headR);
}

function drawTrex() {
  const p = clamp((1.36 - trexT) / 0.4, 0, 1);
  const groundY = H + 120 - p * 360;
  const sc = 0.85 + p * 0.55;
  const open = 30 + p * 48 + Math.sin(frameCount * 0.3) * 6;
  drawRex(W / 2 - 285 * sc + player.laneX * 14, groundY, sc, open, frameCount * 0.4);
}

function drawCountdown(ts) {
  const left = 3 - (ts - calStart) / 1000;
  const n = Math.max(1, Math.ceil(left));
  ctx.fillStyle = "rgba(6,14,9,0.45)";
  ctx.fillRect(0, 0, W, H);
  ctx.textAlign = "center";
  ctx.fillStyle = "#f4b03c";
  ctx.font = "700 28px Trebuchet MS, sans-serif";
  ctx.fillText("Snapping your face onto the runner…", W / 2, H / 2 - 110);
  ctx.fillStyle = bodyPresent ? "#6fe06f" : "#ef5350";
  ctx.font = "600 18px Trebuchet MS, sans-serif";
  ctx.fillText(bodyPresent ? "body locked — get ready!" : "step back so your whole body shows", W / 2, H / 2 - 74);
  ctx.fillStyle = "#eafbe9";
  ctx.font = "700 150px Bangers, Trebuchet MS, sans-serif";
  ctx.fillText(String(n), W / 2, H / 2 + 60);
  ctx.textAlign = "left";
}

function render(ts) {
  drawBackground();
  drawPath();

  const ordered = obstacles.slice().sort((a, b) => a.t - b.t);
  for (const p of props.slice().sort((a, b) => a.t - b.t)) if (p.t <= PLAYER_T) drawProp(p);
  for (const o of ordered) if (o.t <= PLAYER_T) drawObstacle(o);

  if (phase === "dead") drawTrex();
  if (phase === "playing") drawChaser();
  if (phase !== "calibrating") drawRunner();

  for (const o of ordered) if (o.t > PLAYER_T) drawObstacle(o);
  for (const p of props) if (p.t > PLAYER_T) drawProp(p);

  if (phase === "playing" && speed > 0.011) {
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 2;
    for (let i = 0; i < 5; i++) {
      const lx = 30 + i * 120;
      const ph = (worldScroll * 0.4 + i * 90) % 240;
      ctx.beginPath();
      ctx.moveTo(lx, ph);
      ctx.lineTo(lx + 8, ph + 60);
      ctx.stroke();
    }
  }

  if (phase === "calibrating") drawCountdown(ts);

  updateHud();
}

function updateHud() {
  elScore.textContent = String(score);
  elTime.textContent = elapsed.toFixed(1);
  elAction.textContent = phase === "playing" ? action : phase === "calibrating" ? "READY" : action;
  if (usingKb) { elStatus.textContent = "keyboard"; elStatus.style.color = "#38bdf8"; }
  else if (!wsReady) { elStatus.textContent = "link…"; elStatus.style.color = "#8fae93"; }
  else if (!camReady) { elStatus.textContent = "no cam"; elStatus.style.color = "#f4b03c"; }
  else if (bodyPresent) { elStatus.textContent = "tracking"; elStatus.style.color = "#6fe06f"; }
  else { elStatus.textContent = "show body"; elStatus.style.color = "#f4b03c"; }
}

function drawPip() {
  pctx.save();
  pctx.translate(pip.width, 0);
  pctx.scale(-1, 1);
  if (camReady) {
    pctx.drawImage(cam, 0, 0, pip.width, pip.height);
    pctx.restore();
    if (bodyPresent) {
      const mx = steer * pip.width;
      pctx.strokeStyle = "rgba(255,255,255,.85)";
      pctx.lineWidth = 2;
      pctx.beginPath(); pctx.moveTo(mx, 0); pctx.lineTo(mx, pip.height); pctx.stroke();
      pctx.fillStyle = "#6fe06f";
      pctx.beginPath(); pctx.arc(mx, pip.height / 2, 10, 0, Math.PI * 2); pctx.fill();
      const dy = clamp((bodyY - baselineY) * 4 + 0.5, 0, 1) * pip.height;
      pctx.fillStyle = (baselineY - bodyYSmooth > JUMP_THRESH) ? "#38bdf8" : (bodyYSmooth - baselineY > DUCK_THRESH) ? "#f4b03c" : "rgba(255,255,255,.5)";
      pctx.fillRect(pip.width - 10, dy - 3, 8, 6);
    }
  } else {
    pctx.restore();
    pctx.fillStyle = "#05100a";
    pctx.fillRect(0, 0, pip.width, pip.height);
    pctx.fillStyle = "#8fae93";
    pctx.font = "14px Trebuchet MS, sans-serif";
    pctx.textAlign = "center";
    pctx.fillText(usingKb ? "keyboard mode" : "waiting for camera", pip.width / 2, pip.height / 2);
    pctx.textAlign = "left";
  }
}

function frame(ts) {
  maybeSend(ts);
  update(ts);
  render(ts);
  drawPip();
  requestAnimationFrame(frame);
}

startBtn.addEventListener("click", async () => {
  if (window.Jungle) Jungle.start();
  if (!camReady) await initCam();
  beginCalibration();
});

muteBtn.addEventListener("click", () => {
  if (!window.Jungle) return;
  Jungle.start();
  const m = Jungle.toggle();
  muteBtn.textContent = m ? "SOUND: OFF" : "SOUND: ON";
});

fsBtn.addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else if (fsTarget.requestFullscreen) fsTarget.requestFullscreen();
  else if (fsTarget.webkitRequestFullscreen) fsTarget.webkitRequestFullscreen();
});

document.addEventListener("fullscreenchange", () => {
  fsBtn.textContent = document.fullscreenElement ? "EXIT FULL SCREEN" : "FULL SCREEN";
});

speedRange.addEventListener("input", () => {
  speedLevel = clamp(parseInt(speedRange.value, 10) || 2, 1, 4);
  speedName.textContent = SPEED_NAMES[speedLevel];
});

window.addEventListener("keydown", (e) => {
  const k = e.key;
  if (k === "ArrowLeft") { KB.left = true; lastKbDown = performance.now(); }
  else if (k === "ArrowRight") { KB.right = true; lastKbDown = performance.now(); }
  else if (k === " " || k === "ArrowUp") { KB.jump = true; lastKbDown = performance.now(); e.preventDefault(); }
  else if (k === "ArrowDown") { KB.duck = true; lastKbDown = performance.now(); e.preventDefault(); }
  else if (k === "r" || k === "R") { if (phase === "playing" || phase === "dead") startRun(); }
  else if (k === "Enter") { if (phase === "ready") { if (window.Jungle) Jungle.start(); beginCalibration(); } }
});

window.addEventListener("keyup", (e) => {
  const k = e.key;
  if (k === "ArrowLeft") KB.left = false;
  else if (k === "ArrowRight") KB.right = false;
  else if (k === " " || k === "ArrowUp") KB.jump = false;
  else if (k === "ArrowDown") KB.duck = false;
});

connectWS();
initCam();
requestAnimationFrame(frame);
