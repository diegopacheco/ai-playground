const cv = document.getElementById("c");
const ctx = cv.getContext("2d");
const scoreEl = document.getElementById("score");
const bestEl = document.getElementById("best");
const speedFill = document.getElementById("speedfill");
const screenEl = document.getElementById("screen");
const overlayEl = document.getElementById("overlay");
const startBtn = document.getElementById("startbtn");

let W = 0, H = 0, DPR = 1, groundY = 0;
let skyA = [], skyB = [], symbols = [];
let floorOffset = 0;

const H_STAND = 62, H_SLIDE = 32;
const GRAV = 2600, JUMP_V = -1020;
const BASE_SPEED = 380, MAX_SPEED = 820;

const GROUND_DET = [
  { t: "TEMP=0", c: "#ff3b3b" },
  { t: "LINT", c: "#19f6ff" },
  { t: "ASSERT", c: "#9dff4d" },
  { t: "SEED", c: "#ffe14d" },
  { t: "CI GATE", c: "#ff7a3d" },
  { t: "TYPECHECK", c: "#b78bff" }
];
const AIR_DET = [
  { t: "DETERMINISM", c: "#ff3b3b" },
  { t: "SPEC", c: "#19f6ff" },
  { t: "STYLE GUIDE", c: "#ff2bd6" },
  { t: "CODE REVIEW", c: "#b78bff" },
  { t: "REQUIREMENTS", c: "#ffe14d" }
];
const SLOP = [
  { t: "PR", c: "#ff2bd6" },
  { t: "EMAIL", c: "#19f6ff" },
  { t: "CODE", c: "#9dff4d" },
  { t: "TEST", c: "#ffe14d" },
  { t: "REVIEW", c: "#ff7a3d" },
  { t: "COMMIT", c: "#b78bff" },
  { t: "DIFF", c: "#19f6ff" }
];
const VERDICTS = [
  "determinism landed. the agent compiled exactly as asked. tragic.",
  "TEMP=0 hit you. every run identical from now on. game over.",
  "a code review pinned you down. you stopped improvising.",
  "the spec caught you mid-air. requirements satisfied. game over.",
  "the style guide clipped your run. you got predictable.",
  "a CI gate slammed shut. the slop stopped flowing."
];

const rand = (a, b) => a + Math.random() * (b - a);
const pick = a => a[(Math.random() * a.length) | 0];

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth;
  H = window.innerHeight;
  cv.width = W * DPR;
  cv.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  groundY = H - Math.max(96, H * 0.16);
  buildSkylines();
  buildSymbols();
}
window.addEventListener("resize", resize);

function buildLayer(span, hMin, hMax, color) {
  const arr = [];
  let x = 0;
  while (x < span) {
    const w = rand(46, 104);
    const h = rand(hMin, hMax);
    const win = [];
    const cols = Math.max(1, (w / 16) | 0);
    const rows = Math.max(1, (h / 18) | 0);
    for (let i = 0; i < cols * rows; i++) win.push(Math.random() < 0.45);
    arr.push({ x, w, h, color, cols, rows, win });
    x += w + rand(14, 40);
  }
  return { items: arr, span: x };
}

function buildSkylines() {
  const span = Math.max(W * 1.6, 1400);
  skyA = buildLayer(span, H * 0.18, H * 0.42, "#1a0f3d");
  skyB = buildLayer(span, H * 0.10, H * 0.26, "#2b1466");
}

function buildSymbols() {
  symbols = [];
  const glyphs = ["</>", "{ }", "01", "fn()", "git", "AI", "#!", "λ", "==>", "0x"];
  for (let i = 0; i < 16; i++) {
    symbols.push({
      x: rand(0, W), y: rand(30, groundY - 60),
      g: pick(glyphs), s: rand(11, 22),
      a: rand(0.04, 0.16), f: rand(0.15, 0.5)
    });
  }
}

let state = "start";
let player, obstacles, orbs, particles, slopBits;
let distance, speed, score, best, spawnDist, orbDist, slopTimer, shake;

best = parseInt(localStorage.getItem("megaslop_best") || "0", 10);

function reset() {
  player = { x: Math.max(120, W * 0.16), y: groundY - H_STAND, w: 46, h: H_STAND, vy: 0, grounded: true, run: 0 };
  obstacles = [];
  orbs = [];
  particles = [];
  slopBits = [];
  distance = 0;
  speed = BASE_SPEED;
  score = 0;
  spawnDist = 480;
  orbDist = 900;
  slopTimer = 0;
  shake = 0;
}

function emitSlop() {
  const s = pick(SLOP);
  const front = player.grounded && player.h <= H_SLIDE;
  slopBits.push({
    x: player.x + player.w + 4,
    y: player.y + player.h * 0.35,
    vx: rand(120, 260),
    vy: front ? rand(-120, -40) : rand(-360, -180),
    rot: 0,
    vr: rand(-8, 8),
    life: rand(0.7, 1.2), max: 1.2,
    c: s.c, t: s.t
  });
  if (slopBits.length > 40) slopBits.shift();
}

function start() {
  reset();
  state = "play";
  screenEl.classList.add("hidden");
  overlayEl.innerHTML = "";
}

function jump() {
  if (player.grounded) {
    player.vy = JUMP_V;
    player.grounded = false;
    for (let i = 0; i < 8; i++) burst(player.x + player.w / 2, groundY, "#19f6ff", 0.5);
  }
}

let slideHeld = false;

function gameOver() {
  state = "over";
  shake = 14;
  for (let i = 0; i < 36; i++) burst(player.x + player.w / 2, player.y + player.h / 2, pick(["#ff2bd6", "#19f6ff", "#ff3b3b"]), 1.4);
  if (score > best) {
    best = score;
    localStorage.setItem("megaslop_best", String(best));
  }
  overlayEl.innerHTML =
    '<div class="gameover">' +
    "<h1>SLOPPED OUT</h1>" +
    '<p class="verdict">' + pick(VERDICTS) + "</p>" +
    '<div class="stats">' +
    '<div><span class="n">' + score + '</span><span class="l">DISTANCE</span></div>' +
    '<div><span class="n" style="color:var(--magenta);text-shadow:0 0 14px var(--magenta)">' + best + '</span><span class="l">BEST</span></div>' +
    "</div>" +
    '<button id="againbtn">REBOOT AGENT</button>' +
    '<p class="tag dim small">SPACE to run again</p>' +
    "</div>";
  document.getElementById("againbtn").onclick = start;
}

function spawnObstacle() {
  const air = Math.random() < 0.42;
  if (air) {
    const s = pick(AIR_DET);
    const h = 44;
    obstacles.push({ x: W + 60, y: groundY - 100, w: s.t.length > 8 ? 150 : 110, h, c: s.c, t: s.t, air: true });
  } else {
    const s = pick(GROUND_DET);
    const h = rand(50, 90);
    obstacles.push({ x: W + 60, y: groundY - h, w: Math.max(44, s.t.length * 9), h, c: s.c, t: s.t, air: false });
  }
}

function spawnOrb() {
  orbs.push({ x: W + 60, y: groundY - rand(90, 170), r: 13, got: false, ph: rand(0, 6.28) });
}

function burst(x, y, c, scale) {
  particles.push({
    x, y,
    vx: rand(-220, 220) * scale,
    vy: rand(-320, 40) * scale,
    life: rand(0.3, 0.8), max: 0.8, c, r: rand(2, 4.5)
  });
}

function hit(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function update(dt) {
  speed = Math.min(MAX_SPEED, BASE_SPEED + distance * 0.05);
  distance += speed * dt * 0.1;
  score = Math.floor(distance);
  floorOffset = (floorOffset + speed * dt) % 60;

  for (const L of [skyA, skyB]) {
    const f = L === skyA ? 0.12 : 0.28;
    for (const b of L.items) {
      b.x -= speed * dt * f;
      if (b.x + b.w < 0) b.x += L.span;
    }
  }
  for (const s of symbols) {
    s.x -= speed * dt * s.f;
    if (s.x < -40) { s.x = W + rand(10, 80); s.y = rand(30, groundY - 60); }
  }

  if (!player.grounded) {
    player.vy += GRAV * dt;
    player.y += player.vy * dt;
    player.h = H_STAND;
    if (player.y + H_STAND >= groundY) {
      player.y = groundY - H_STAND;
      player.vy = 0;
      player.grounded = true;
    }
  } else {
    player.h = slideHeld ? H_SLIDE : H_STAND;
    player.y = groundY - player.h;
    player.run += speed * dt;
  }

  spawnDist -= speed * dt;
  if (spawnDist <= 0) {
    spawnObstacle();
    spawnDist = rand(300, 560);
  }
  orbDist -= speed * dt;
  if (orbDist <= 0) {
    spawnOrb();
    orbDist = rand(700, 1300);
  }

  slopTimer -= dt;
  if (slopTimer <= 0) {
    emitSlop();
    slopTimer = rand(0.12, 0.26);
  }

  for (let i = obstacles.length - 1; i >= 0; i--) {
    const o = obstacles[i];
    o.x -= speed * dt;
    if (o.x + o.w < -20) { obstacles.splice(i, 1); continue; }
    if (hit(player, o)) { gameOver(); return; }
  }

  for (let i = orbs.length - 1; i >= 0; i--) {
    const o = orbs[i];
    o.x -= speed * dt;
    o.ph += dt * 4;
    if (o.x + o.r < -20) { orbs.splice(i, 1); continue; }
    const cx = player.x + player.w / 2, cy = player.y + player.h / 2;
    if (!o.got && Math.hypot(cx - o.x, cy - o.y) < o.r + 26) {
      o.got = true;
      distance += 30;
      for (let k = 0; k < 12; k++) burst(o.x, o.y, "#ffe14d", 0.9);
      orbs.splice(i, 1);
    }
  }

  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life -= dt;
    if (p.life <= 0) { particles.splice(i, 1); continue; }
    p.vy += 1200 * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
  }

  for (let i = slopBits.length - 1; i >= 0; i--) {
    const b = slopBits[i];
    b.life -= dt;
    if (b.life <= 0 || b.x > W + 60) { slopBits.splice(i, 1); continue; }
    b.vy += 900 * dt;
    b.x += (b.vx - speed * 0.35) * dt;
    b.y += b.vy * dt;
    b.rot += b.vr * dt;
  }

  if (shake > 0) shake = Math.max(0, shake - dt * 40);
}

function drawBuilding(b, baseY) {
  ctx.fillStyle = b.color;
  ctx.fillRect(b.x, baseY - b.h, b.w, b.h);
  ctx.strokeStyle = "rgba(123,44,255,.55)";
  ctx.lineWidth = 1;
  ctx.strokeRect(b.x + 0.5, baseY - b.h + 0.5, b.w - 1, b.h - 1);
  const cw = b.w / b.cols, ch = b.h / b.rows;
  for (let r = 0; r < b.rows; r++) {
    for (let c = 0; c < b.cols; c++) {
      if (!b.win[r * b.cols + c]) continue;
      ctx.fillStyle = (r + c) % 2 ? "rgba(25,246,255,.5)" : "rgba(255,43,214,.45)";
      ctx.fillRect(b.x + c * cw + 2, baseY - b.h + r * ch + 2, cw - 4, ch - 4);
    }
  }
}

function drawBackground() {
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, "#0c0524");
  g.addColorStop(0.5, "#160a38");
  g.addColorStop(1, "#070312");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.fillStyle = "rgba(255,43,214,.10)";
  ctx.beginPath();
  ctx.arc(W * 0.72, groundY * 0.55, Math.min(W, H) * 0.22, 0, 6.2832);
  ctx.fill();
  ctx.restore();

  for (const s of symbols) {
    ctx.globalAlpha = s.a;
    ctx.fillStyle = "#19f6ff";
    ctx.font = s.s + "px monospace";
    ctx.fillText(s.g, s.x, s.y);
  }
  ctx.globalAlpha = 1;

  for (const b of skyA.items) drawBuilding(b, groundY);
  for (const b of skyB.items) drawBuilding(b, groundY);
}

function drawFloor() {
  ctx.fillStyle = "#05010f";
  ctx.fillRect(0, groundY, W, H - groundY);

  ctx.save();
  ctx.strokeStyle = "rgba(25,246,255,.18)";
  ctx.lineWidth = 1;
  for (let x = -floorOffset; x < W; x += 60) {
    ctx.beginPath();
    ctx.moveTo(x, groundY);
    ctx.lineTo((x - W / 2) * 3 + W / 2, H);
    ctx.stroke();
  }
  for (let i = 1; i < 7; i++) {
    const y = groundY + (H - groundY) * (i / 7) * (i / 7);
    ctx.globalAlpha = 0.18 - i * 0.02;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  ctx.restore();

  ctx.save();
  ctx.shadowColor = "#19f6ff";
  ctx.shadowBlur = 22;
  ctx.strokeStyle = "#19f6ff";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, groundY);
  ctx.lineTo(W, groundY);
  ctx.stroke();
  ctx.restore();
}

function drawAgent() {
  const x = player.x, y = player.y, w = player.w, h = player.h;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,.4)";
  ctx.beginPath();
  ctx.ellipse(x + w / 2, groundY + 6, w * 0.6, 8, 0, 0, 6.2832);
  ctx.fill();

  ctx.shadowColor = "#19f6ff";
  ctx.shadowBlur = 20;

  if (player.grounded && h > H_SLIDE) {
    const phase = Math.sin(player.run * 0.05);
    ctx.strokeStyle = "#ff2bd6";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x + w * 0.35, y + h - 4);
    ctx.lineTo(x + w * 0.30 + phase * 9, groundY);
    ctx.moveTo(x + w * 0.65, y + h - 4);
    ctx.lineTo(x + w * 0.70 - phase * 9, groundY);
    ctx.stroke();
  }

  const bg = ctx.createLinearGradient(x, y, x + w, y + h);
  bg.addColorStop(0, "#0bd6e6");
  bg.addColorStop(1, "#1577ff");
  ctx.fillStyle = bg;
  roundRect(x, y, w, h, 9);
  ctx.fill();

  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255,255,255,.12)";
  roundRect(x + 4, y + 4, w - 8, h * 0.34, 6);
  ctx.fill();

  const visorY = y + h * 0.22;
  ctx.fillStyle = "#08121f";
  roundRect(x + 6, visorY, w - 12, h * 0.30, 5);
  ctx.fill();
  ctx.fillStyle = "#19f6ff";
  ctx.shadowColor = "#19f6ff";
  ctx.shadowBlur = 12;
  const ey = visorY + h * 0.13;
  ctx.fillRect(x + 12, ey, 7, 5);
  ctx.fillRect(x + w - 19, ey, 7, 5);
  ctx.shadowBlur = 0;

  ctx.strokeStyle = "#ffe14d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x + w / 2, y);
  ctx.lineTo(x + w / 2, y - 12);
  ctx.stroke();
  ctx.fillStyle = "#ffe14d";
  ctx.shadowColor = "#ffe14d";
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(x + w / 2, y - 14, 3.4, 0, 6.2832);
  ctx.fill();
  ctx.restore();
}

function drawObstacle(o) {
  ctx.save();
  ctx.shadowColor = o.c;
  ctx.shadowBlur = 16;
  ctx.fillStyle = "rgba(10,4,24,.85)";
  roundRect(o.x, o.y, o.w, o.h, 6);
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = o.c;
  roundRect(o.x, o.y, o.w, o.h, 6);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = o.c;
  ctx.font = "bold 12px 'Courier New',monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.save();
  if (o.air) {
    ctx.fillText(o.t, o.x + o.w / 2, o.y + o.h / 2);
  } else {
    ctx.translate(o.x + o.w / 2, o.y + o.h / 2);
    if (o.h > 40 && o.t.length * 8 > o.h) ctx.rotate(-Math.PI / 2);
    ctx.fillText(o.t, 0, 0);
  }
  ctx.restore();
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
  if (o.air) {
    ctx.strokeStyle = o.c;
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    for (let i = 0; i < 3; i++) {
      const lx = o.x + o.w * (0.3 + i * 0.2);
      ctx.beginPath();
      ctx.moveTo(lx, o.y + o.h);
      ctx.lineTo(lx, o.y + o.h + 10);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

function drawOrb(o) {
  const pulse = 1 + Math.sin(o.ph) * 0.12;
  ctx.save();
  ctx.shadowColor = "#ffe14d";
  ctx.shadowBlur = 18;
  ctx.fillStyle = "#ffe14d";
  ctx.beginPath();
  ctx.arc(o.x, o.y, o.r * pulse, 0, 6.2832);
  ctx.fill();
  ctx.fillStyle = "#070312";
  ctx.font = "bold 13px 'Courier New',monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("{}", o.x, o.y + 1);
  ctx.restore();
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function drawSlopBits() {
  for (const b of slopBits) {
    ctx.save();
    ctx.globalAlpha = Math.min(1, b.life / b.max);
    ctx.translate(b.x, b.y);
    ctx.rotate(b.rot);
    ctx.shadowColor = b.c;
    ctx.shadowBlur = 10;
    ctx.fillStyle = "rgba(8,4,20,.9)";
    const w = b.t.length * 8 + 10, h = 18;
    roundRect(-w / 2, -h / 2, w, h, 4);
    ctx.fill();
    ctx.strokeStyle = b.c;
    ctx.lineWidth = 1.5;
    roundRect(-w / 2, -h / 2, w, h, 4);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = b.c;
    ctx.font = "bold 11px 'Courier New',monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(b.t, 0, 1);
    ctx.restore();
  }
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function drawParticles() {
  for (const p of particles) {
    ctx.globalAlpha = Math.max(0, p.life / p.max);
    ctx.fillStyle = p.c;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, 6.2832);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function roundRect(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function render() {
  ctx.save();
  if (shake > 0) ctx.translate(rand(-shake, shake), rand(-shake, shake));
  drawBackground();
  drawFloor();
  for (const o of orbs) drawOrb(o);
  for (const o of obstacles) drawObstacle(o);
  drawAgent();
  drawSlopBits();
  drawParticles();
  ctx.restore();

  scoreEl.textContent = score;
  bestEl.textContent = best;
  speedFill.style.width = (10 + ((speed - BASE_SPEED) / (MAX_SPEED - BASE_SPEED)) * 90) + "%";
}

let last = performance.now();
function loop(now) {
  const dt = Math.min(0.045, (now - last) / 1000);
  last = now;
  if (state === "play") update(dt);
  render();
  requestAnimationFrame(loop);
}

function onJumpKey() {
  if (state === "start") start();
  else if (state === "over") start();
  else jump();
}

window.addEventListener("keydown", e => {
  if (e.code === "Space" || e.code === "ArrowUp" || e.code === "KeyW") {
    e.preventDefault();
    onJumpKey();
  } else if (e.code === "ArrowDown" || e.code === "KeyS") {
    e.preventDefault();
    slideHeld = true;
  }
});
window.addEventListener("keyup", e => {
  if (e.code === "ArrowDown" || e.code === "KeyS") slideHeld = false;
});

cv.addEventListener("pointerdown", e => {
  if (state !== "play") return;
  if (e.clientY > window.innerHeight * 0.6) slideHeld = true;
  else jump();
});
window.addEventListener("pointerup", () => { slideHeld = false; });

startBtn.onclick = start;

resize();
reset();
requestAnimationFrame(loop);
