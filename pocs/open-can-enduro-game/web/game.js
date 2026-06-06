const game = document.getElementById("game");
const ctx = game.getContext("2d");
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");

const elScore = document.getElementById("score");
const elSpeed = document.getElementById("speed");
const elStatus = document.getElementById("status");
const overlay = document.getElementById("overlay");
const overlayTitle = document.getElementById("overlay-title");
const overlayText = document.getElementById("overlay-text");
const startBtn = document.getElementById("start");

const W = game.width;
const H = game.height;
const MARGIN = 64;
const ROAD_L = MARGIN;
const ROAD_R = W - MARGIN;
const ROAD_W = ROAD_R - ROAD_L;
const ENEMY_COLORS = ["#ef4444", "#f59e0b", "#a855f7", "#22d3ee", "#84cc16"];

const car = { w: 46, h: 84, screenX: ROAD_L + (ROAD_W - 46) / 2, y: H - 130 };

let steer = 0.5;
let handPresent = false;
let camReady = false;
let wsReady = false;

let enemies = [];
let dash = 0;
let speed = 4;
let score = 0;
let running = false;
let crashed = false;
let spawnTimer = 60;

let ws = null;
let awaiting = false;
let lastSend = 0;

function targetScreenX() {
  const s = Math.min(1, Math.max(0, 0.5 + (steer - 0.5) * 1.7));
  return ROAD_L + s * (ROAD_W - car.w);
}

function connectWS() {
  try {
    ws = new WebSocket("ws://" + location.hostname + ":8765");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { wsReady = true; };
    ws.onmessage = (e) => {
      awaiting = false;
      try {
        const m = JSON.parse(e.data);
        handPresent = !!m.present;
        if (m.present && typeof m.x === "number") steer = m.x;
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

function spawnEnemy() {
  const w = 46;
  const x = ROAD_L + Math.random() * (ROAD_W - w);
  enemies.push({ x, y: -110, w, h: 84, color: ENEMY_COLORS[(Math.random() * ENEMY_COLORS.length) | 0], passed: false });
}

function hit(e) {
  return car.screenX < e.x + e.w && car.screenX + car.w > e.x && car.y < e.y + e.h && car.y + car.h > e.y;
}

function crash() {
  crashed = true;
  running = false;
  overlayTitle.textContent = "CRASH!";
  overlayText.textContent = "You passed " + score + " cars. Raise your hand and race again.";
  startBtn.textContent = "RACE AGAIN";
  overlay.classList.remove("hidden");
}

function startRun() {
  enemies = [];
  speed = 4;
  score = 0;
  spawnTimer = 60;
  crashed = false;
  running = true;
  car.screenX = ROAD_L + (ROAD_W - car.w) / 2;
  overlay.classList.add("hidden");
}

function update() {
  if (!running || crashed) return;
  speed += 0.0016;
  dash = (dash + speed) % 44;
  car.screenX += (targetScreenX() - car.screenX) * 0.2;
  spawnTimer -= 1;
  if (spawnTimer <= 0) {
    spawnEnemy();
    spawnTimer = 50 + Math.random() * 45;
  }
  for (const e of enemies) {
    e.y += speed;
    if (!e.passed && e.y > car.y + car.h) { e.passed = true; score += 1; }
    if (hit(e)) { crash(); break; }
  }
  enemies = enemies.filter((e) => e.y < H + 120);
}

function drawCar(x, y, w, h, body, glass) {
  ctx.fillStyle = body;
  roundRect(x, y, w, h, 9);
  ctx.fill();
  ctx.fillStyle = "rgba(0,0,0,.25)";
  roundRect(x + 6, y + 8, w - 12, 18, 5); ctx.fill();
  ctx.fillStyle = glass;
  roundRect(x + 8, y + 30, w - 16, 22, 5); ctx.fill();
  ctx.fillStyle = "#0c0f1a";
  ctx.fillRect(x - 4, y + 12, 6, 20);
  ctx.fillRect(x + w - 2, y + 12, 6, 20);
  ctx.fillRect(x - 4, y + h - 32, 6, 20);
  ctx.fillRect(x + w - 2, y + h - 32, 6, 20);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function render() {
  ctx.fillStyle = "#1f7a36";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#176b2d";
  for (let y = (dash % 44) - 44; y < H; y += 44) {
    ctx.fillRect(0, y, MARGIN, 22);
    ctx.fillRect(W - MARGIN, y, MARGIN, 22);
  }
  ctx.fillStyle = "#2f3445";
  ctx.fillRect(ROAD_L, 0, ROAD_W, H);
  ctx.fillStyle = "#e7e9f2";
  ctx.fillRect(ROAD_L - 6, 0, 6, H);
  ctx.fillRect(ROAD_R, 0, 6, H);
  ctx.fillStyle = "rgba(255,255,255,.7)";
  const midA = ROAD_L + ROAD_W / 3 - 3;
  const midB = ROAD_L + (2 * ROAD_W) / 3 - 3;
  for (let y = (dash % 44) - 44; y < H; y += 44) {
    ctx.fillRect(midA, y, 6, 24);
    ctx.fillRect(midB, y, 6, 24);
  }
  for (const e of enemies) drawCar(e.x, e.y, e.w, e.h, e.color, "#0c0f1a");
  drawCar(car.screenX, car.y, car.w, car.h, "#34d399", "#cdfbe7");

  elScore.textContent = String(score);
  elSpeed.textContent = String(Math.round(speed * 16));
  if (!wsReady) { elStatus.textContent = "link…"; elStatus.style.color = "#8b93b8"; }
  else if (!camReady) { elStatus.textContent = "no cam"; elStatus.style.color = "#fbbf24"; }
  else if (handPresent) { elStatus.textContent = "tracking"; elStatus.style.color = "#34d399"; }
  else { elStatus.textContent = "show hand"; elStatus.style.color = "#fbbf24"; }
}

function drawPip() {
  pctx.save();
  pctx.translate(pip.width, 0);
  pctx.scale(-1, 1);
  if (camReady) {
    pctx.drawImage(cam, 0, 0, pip.width, pip.height);
    pctx.restore();
    if (handPresent) {
      const mx = steer * pip.width;
      pctx.fillStyle = "#34d399";
      pctx.beginPath();
      pctx.arc(mx, pip.height / 2, 11, 0, Math.PI * 2);
      pctx.fill();
      pctx.strokeStyle = "rgba(255,255,255,.9)";
      pctx.lineWidth = 2;
      pctx.beginPath();
      pctx.moveTo(mx, 0);
      pctx.lineTo(mx, pip.height);
      pctx.stroke();
    }
  } else {
    pctx.restore();
    pctx.fillStyle = "#05070f";
    pctx.fillRect(0, 0, pip.width, pip.height);
    pctx.fillStyle = "#8b93b8";
    pctx.font = "14px Trebuchet MS, sans-serif";
    pctx.textAlign = "center";
    pctx.fillText("waiting for camera", pip.width / 2, pip.height / 2);
  }
}

function frame(ts) {
  maybeSend(ts);
  update();
  render();
  drawPip();
  requestAnimationFrame(frame);
}

startBtn.addEventListener("click", async () => {
  if (!camReady) await initCam();
  startRun();
});

window.addEventListener("keydown", (e) => {
  if (e.key === "r" || e.key === "R") startRun();
});

connectWS();
initCam();
requestAnimationFrame(frame);
