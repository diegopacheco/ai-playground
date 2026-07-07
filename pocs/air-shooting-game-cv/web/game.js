const cv = document.getElementById("game");
const ctx = cv.getContext("2d");
const W = cv.width, H = cv.height;
const cam = document.getElementById("cam");
const grab = document.getElementById("grab");
const gctx = grab.getContext("2d");
const pip = document.getElementById("pip");
const pctx = pip.getContext("2d");
const elStatus = document.getElementById("status");

const BW = 64, BH = 96;
const FIRE_T = 0.55, COCK_T = 0.78;
const SHIRTS = ["#e05555", "#7bc47f", "#b08bd0", "#6fa8dc"];
const BANDANAS = ["#f4d03f", "#e07f9c", "#5dd0c0", "#f2a65a"];

const SPOTS = [
  { type: "window", x: 95, y: 215, w: 70, h: 85 },
  { type: "window", x: 215, y: 215, w: 70, h: 85 },
  { type: "window", x: 675, y: 235, w: 70, h: 85 },
  { type: "window", x: 795, y: 235, w: 70, h: 85 },
  { type: "ground", cx: 380, coverTop: 436, cover: "barrel" },
  { type: "ground", cx: 480, coverTop: 452, cover: "crate" },
  { type: "ground", cx: 580, coverTop: 436, cover: "barrel" },
];

let phase = "menu";
let frame = 0;
let score = 0, hearts = 3, wave = 0;
let banditsLeft = 0, spawnCd = 0, bannerT = 0, overT = 0;
let bandits = [], bags = [], fx = [];
let shake = 0, redFlash = 0, invuln = 0, bagCd = 500;

let wsReady = false, camReady = false, awaiting = false;
let ws = null;
let hand = { seen: 0, gun: false, thumb: 1, x: 0.5, y: 0.5 };
let aim = { x: W / 2, y: H / 2 };
let hammer = true, fireCd = 0;
let mouse = { x: W / 2, y: H / 2 };

let ac = null;
function audio() {
  if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)();
  if (ac.state === "suspended") ac.resume();
  return ac;
}
function sfxShot() {
  const a = audio(), t = a.currentTime, len = 0.18;
  const buf = a.createBuffer(1, a.sampleRate * len, a.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2);
  const src = a.createBufferSource();
  src.buffer = buf;
  const f = a.createBiquadFilter();
  f.type = "lowpass";
  f.frequency.value = 900;
  const g = a.createGain();
  g.gain.setValueAtTime(0.8, t);
  g.gain.exponentialRampToValueAtTime(0.01, t + len);
  src.connect(f); f.connect(g); g.connect(a.destination);
  src.start(t);
}
function tone(type, f0, f1, len, vol) {
  const a = audio(), t = a.currentTime;
  const o = a.createOscillator(), g = a.createGain();
  o.type = type;
  o.frequency.setValueAtTime(f0, t);
  o.frequency.exponentialRampToValueAtTime(f1, t + len);
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.01, t + len);
  o.connect(g); g.connect(a.destination);
  o.start(t); o.stop(t + len);
}
function sfxHit() { tone("square", 700, 180, 0.16, 0.35); }
function sfxHurt() { tone("sawtooth", 160, 55, 0.35, 0.5); }
function sfxDing() { tone("sine", 1250, 1800, 0.2, 0.35); }

function spawnBandit() {
  const free = SPOTS.filter(s => !bandits.some(b => b.spot === s));
  if (!free.length) return;
  const spot = free[(Math.random() * free.length) | 0];
  const ci = (Math.random() * SHIRTS.length) | 0;
  const aimT = Math.max(48, 130 - wave * 10) + (Math.random() * 30 | 0);
  bandits.push({ spot, state: "rise", t: 0, aimT, shirt: SHIRTS[ci], bandana: BANDANAS[ci], flip: Math.random() < 0.5 });
  banditsLeft--;
}

function banditGeom(b) {
  const s = b.spot;
  if (s.type === "window") return { cx: s.x + s.w / 2, by: s.y + s.h, clip: [s.x, s.y, s.w, s.h] };
  return { cx: s.cx, by: s.coverTop + 24, clip: [s.cx - 54, 0, 108, s.coverTop + 20] };
}

function banditFrac(b) {
  if (b.state === "rise") return b.t / 18;
  if (b.state === "hide") return 1 - b.t / 12;
  return 1;
}

function startGame() {
  score = 0; hearts = 3; wave = 1;
  bandits = []; bags = []; fx = [];
  banditsLeft = 6 + wave * 2;
  spawnCd = 40; bagCd = 400; invuln = 0;
  bannerT = 90; phase = "wave";
}

function nextWave() {
  wave++;
  banditsLeft = 6 + wave * 2;
  spawnCd = 40;
  bannerT = 90; phase = "wave";
}

function addPop(x, y, text, color) {
  fx.push({ kind: "pop", x, y, t: 0, ttl: 50, text, color });
}
function addPuffs(x, y, n, color) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2, sp = 0.6 + Math.random() * 2;
    fx.push({ kind: "puff", x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 0.6, t: 0, ttl: 26 + Math.random() * 14, r: 4 + Math.random() * 6, color });
  }
}

function fire() {
  if (fireCd > 0) return;
  fireCd = 10;
  sfxShot();
  shake = Math.max(shake, 4);
  fx.push({ kind: "flash", x: aim.x, y: aim.y, t: 0, ttl: 7 });
  if (phase === "menu") { startGame(); return; }
  if (phase === "over") { if (overT > 45) startGame(); return; }
  if (phase !== "play" && phase !== "wave") return;
  let hit = false;
  for (let i = bags.length - 1; i >= 0; i--) {
    const g = bags[i];
    if (Math.hypot(g.x - aim.x, g.y - aim.y) < 32) {
      bags.splice(i, 1);
      score += 300;
      sfxDing();
      addPop(g.x, g.y, "+300", "#ffd93b");
      addPuffs(g.x, g.y, 8, "#ffd93b");
      hit = true;
      break;
    }
  }
  if (!hit) {
    for (const b of bandits) {
      if (b.state === "dead" || b.state === "hide") continue;
      const gm = banditGeom(b);
      const f = banditFrac(b);
      const top = gm.by - BH * f;
      if (aim.x > gm.cx - 36 && aim.x < gm.cx + 36 && aim.y > top - 26 && aim.y < gm.by) {
        b.state = "dead"; b.t = 0;
        score += 100;
        sfxHit();
        addPop(gm.cx, top, "+100", "#9dff8a");
        addPuffs(aim.x, aim.y, 6, "#ffe8c2");
        hit = true;
        break;
      }
    }
  }
  if (!hit) {
    fx.push({ kind: "impact", x: aim.x, y: aim.y, t: 0, ttl: 14 });
    addPuffs(aim.x, aim.y, 4, "#d8b98c");
  }
}

function playerHit() {
  if (invuln > 0) return;
  hearts--;
  invuln = 80;
  redFlash = 26;
  shake = Math.max(shake, 12);
  sfxHurt();
  if (hearts <= 0) { phase = "over"; overT = 0; }
}

function update() {
  frame++;
  if (fireCd > 0) fireCd--;
  if (shake > 0) shake *= 0.85;
  if (redFlash > 0) redFlash--;
  if (invuln > 0) invuln--;

  const usingHand = hand.seen > 0;
  if (usingHand) hand.seen--;
  const tx = usingHand ? hand.x * W : mouse.x;
  const ty = usingHand ? hand.y * H : mouse.y;
  aim.x += (tx - aim.x) * 0.35;
  aim.y += (ty - aim.y) * 0.35;

  if (usingHand && hand.gun) {
    if (hammer && hand.thumb < FIRE_T) { hammer = false; fire(); }
    else if (!hammer && hand.thumb > COCK_T) hammer = true;
  }

  if (phase === "wave") {
    bannerT--;
    if (bannerT <= 0) phase = "play";
  }
  if (phase === "over") overT++;

  if (phase === "play") {
    const maxActive = Math.min(4, 1 + Math.floor(wave / 2));
    if (banditsLeft > 0) {
      spawnCd--;
      if (spawnCd <= 0 && bandits.filter(b => b.state !== "dead").length < maxActive) {
        spawnBandit();
        spawnCd = Math.max(26, 80 - wave * 6) + (Math.random() * 30 | 0);
      }
    } else if (!bandits.length) {
      nextWave();
    }
    bagCd--;
    if (bagCd <= 0) {
      const dir = Math.random() < 0.5 ? 1 : -1;
      bags.push({ x: dir > 0 ? -40 : W + 40, dir, y0: 90 + Math.random() * 70, t: 0, y: 0 });
      bagCd = 480 + (Math.random() * 240 | 0);
    }
  }

  for (let i = bandits.length - 1; i >= 0; i--) {
    const b = bandits[i];
    b.t++;
    if (b.state === "rise" && b.t >= 18) { b.state = "aim"; b.t = 0; }
    else if (b.state === "aim" && b.t >= b.aimT) {
      b.state = "shoot"; b.t = 0;
      const gm = banditGeom(b);
      fx.push({ kind: "bang", x: gm.cx + (b.flip ? -26 : 26), y: gm.by - 52, t: 0, ttl: 12 });
      playerHit();
    }
    else if (b.state === "shoot" && b.t >= 14) { b.state = "hide"; b.t = 0; }
    else if (b.state === "hide" && b.t >= 12) bandits.splice(i, 1);
    else if (b.state === "dead" && b.t >= 30) bandits.splice(i, 1);
  }

  for (let i = bags.length - 1; i >= 0; i--) {
    const g = bags[i];
    g.t++;
    g.x += g.dir * 2.4;
    g.y = g.y0 + Math.sin(g.t * 0.08) * 16;
    if (g.x < -60 || g.x > W + 60) bags.splice(i, 1);
  }

  for (let i = fx.length - 1; i >= 0; i--) {
    const e = fx[i];
    e.t++;
    if (e.kind === "puff") { e.x += e.vx; e.y += e.vy; e.vy -= 0.02; }
    if (e.kind === "pop") e.y -= 0.9;
    if (e.t >= e.ttl) fx.splice(i, 1);
  }
}

function rrect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawSky() {
  const g = ctx.createLinearGradient(0, 0, 0, 340);
  g.addColorStop(0, "#ffcf6b");
  g.addColorStop(0.5, "#ff9e5e");
  g.addColorStop(1, "#e2637a");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, 340);
  ctx.fillStyle = "#ffe9a8";
  ctx.beginPath();
  ctx.arc(480, 210, 74, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffb347";
  ctx.lineWidth = 6;
  ctx.stroke();
  ctx.fillStyle = "rgba(255,240,210,.75)";
  const cx1 = 160 + (frame * 0.12) % (W + 300) - 150;
  const cx2 = 700 - (frame * 0.08) % (W + 300);
  for (const cx0 of [cx1, cx2]) {
    ctx.beginPath();
    ctx.ellipse(cx0, 80, 52, 18, 0, 0, Math.PI * 2);
    ctx.ellipse(cx0 + 38, 88, 38, 14, 0, 0, Math.PI * 2);
    ctx.ellipse(cx0 - 40, 90, 34, 13, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#b95f6e";
  ctx.beginPath();
  ctx.moveTo(0, 340);
  ctx.lineTo(0, 290); ctx.lineTo(70, 250); ctx.lineTo(120, 250); ctx.lineTo(160, 300);
  ctx.lineTo(300, 305); ctx.lineTo(420, 270); ctx.lineTo(520, 270); ctx.lineTo(600, 310);
  ctx.lineTo(760, 300); ctx.lineTo(830, 255); ctx.lineTo(890, 255); ctx.lineTo(960, 300);
  ctx.lineTo(960, 340);
  ctx.closePath();
  ctx.fill();
}

function drawGround() {
  const g = ctx.createLinearGradient(0, 330, 0, H);
  g.addColorStop(0, "#e8b878");
  g.addColorStop(1, "#c98d4f");
  ctx.fillStyle = g;
  ctx.fillRect(0, 330, W, H - 330);
  ctx.strokeStyle = "rgba(140,90,45,.35)";
  ctx.lineWidth = 3;
  for (let i = 0; i < 6; i++) {
    ctx.beginPath();
    const y = 380 + i * 28;
    ctx.moveTo(20 + (i % 2) * 60, y);
    ctx.quadraticCurveTo(W / 2, y + 8, W - 30 - (i % 2) * 80, y);
    ctx.stroke();
  }
}

function drawCactus(x, y, s) {
  ctx.fillStyle = "#5f9e62";
  ctx.strokeStyle = "#3c6b3e";
  ctx.lineWidth = 3;
  rrect(x - 9 * s, y - 60 * s, 18 * s, 60 * s, 9 * s);
  ctx.fill(); ctx.stroke();
  rrect(x - 30 * s, y - 48 * s, 12 * s, 26 * s, 6 * s);
  ctx.fill(); ctx.stroke();
  rrect(x + 18 * s, y - 40 * s, 12 * s, 22 * s, 6 * s);
  ctx.fill(); ctx.stroke();
}

function drawBuilding(x, w, top, sign, base, trim, windows) {
  ctx.fillStyle = base;
  ctx.strokeStyle = "#5a3620";
  ctx.lineWidth = 5;
  ctx.fillRect(x, top, w, 400 - top);
  ctx.strokeRect(x, top, w, 400 - top);
  ctx.fillStyle = trim;
  ctx.fillRect(x - 10, top - 18, w + 20, 26);
  ctx.strokeRect(x - 10, top - 18, w + 20, 26);
  ctx.fillStyle = "#7a4a28";
  ctx.fillRect(x + 14, top + 24, w - 28, 46);
  ctx.strokeRect(x + 14, top + 24, w - 28, 46);
  ctx.fillStyle = "#ffe8c2";
  ctx.font = "bold 26px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.textAlign = "center";
  ctx.fillText(sign, x + w / 2, top + 57);
  for (const win of windows) {
    ctx.fillStyle = "#2c2036";
    ctx.fillRect(win.x, win.y, win.w, win.h);
    ctx.strokeStyle = "#5a3620";
    ctx.lineWidth = 5;
    ctx.strokeRect(win.x, win.y, win.w, win.h);
    ctx.fillStyle = trim;
    ctx.fillRect(win.x - 8, win.y - 12, win.w + 16, 10);
  }
  ctx.fillStyle = "#8a5a32";
  ctx.fillRect(x + w / 2 - 26, 400 - 76, 52, 76);
  ctx.strokeStyle = "#5a3620";
  ctx.strokeRect(x + w / 2 - 26, 400 - 76, 52, 76);
}

function drawScene() {
  drawSky();
  drawGround();
  drawBuilding(50, 280, 140, "SALOON", "#d98e5f", "#a5544a",
    [{ x: 95, y: 215, w: 70, h: 85 }, { x: 215, y: 215, w: 70, h: 85 }]);
  drawBuilding(630, 280, 160, "GOLD & CO", "#c9b26a", "#7d8a5a",
    [{ x: 675, y: 235, w: 70, h: 85 }, { x: 795, y: 235, w: 70, h: 85 }]);
  drawCactus(30, 520, 1.1);
  drawCactus(930, 505, 0.9);
}

function drawBarrel(cx, top) {
  ctx.fillStyle = "#a5683a";
  ctx.strokeStyle = "#5a3620";
  ctx.lineWidth = 4;
  rrect(cx - 32, top, 64, 76, 12);
  ctx.fill(); ctx.stroke();
  ctx.strokeStyle = "#3d2415";
  ctx.beginPath(); ctx.moveTo(cx - 32, top + 20); ctx.lineTo(cx + 32, top + 20); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx - 32, top + 56); ctx.lineTo(cx + 32, top + 56); ctx.stroke();
  ctx.strokeStyle = "rgba(90,54,32,.5)";
  for (const dx of [-14, 0, 14]) {
    ctx.beginPath(); ctx.moveTo(cx + dx, top + 4); ctx.lineTo(cx + dx, top + 72); ctx.stroke();
  }
}

function drawCrate(cx, top) {
  ctx.fillStyle = "#c99a56";
  ctx.strokeStyle = "#5a3620";
  ctx.lineWidth = 4;
  ctx.fillRect(cx - 42, top, 84, 62);
  ctx.strokeRect(cx - 42, top, 84, 62);
  ctx.beginPath();
  ctx.moveTo(cx - 42, top); ctx.lineTo(cx + 42, top + 62);
  ctx.moveTo(cx + 42, top); ctx.lineTo(cx - 42, top + 62);
  ctx.stroke();
}

function drawCovers() {
  for (const s of SPOTS) {
    if (s.type !== "ground") continue;
    if (s.cover === "barrel") drawBarrel(s.cx, s.coverTop);
    else drawCrate(s.cx, s.coverTop);
  }
}

function drawBandit(b) {
  const gm = banditGeom(b);
  const f = banditFrac(b);
  ctx.save();
  ctx.beginPath();
  ctx.rect(gm.clip[0], gm.clip[1], gm.clip[2], gm.clip[3]);
  ctx.clip();
  ctx.translate(gm.cx, gm.by + BH * (1 - f));
  if (b.state === "dead") {
    const k = b.t / 30;
    ctx.translate(0, k * k * 60);
    ctx.rotate((b.flip ? -1 : 1) * k * 0.9);
    ctx.globalAlpha = 1 - k * 0.6;
  }
  const dir = b.flip ? -1 : 1;
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#2c2036";
  ctx.fillStyle = b.shirt;
  rrect(-22, -46, 44, 46, 10);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = b.bandana;
  ctx.fillRect(-22, -46, 44, 10);
  const armY = b.state === "aim" || b.state === "shoot" ? -38 : -22;
  ctx.fillStyle = b.shirt;
  rrect(dir * 14, armY, dir * 26, 12, 6);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#444";
  ctx.fillRect(dir * 36 - (dir < 0 ? 14 : 0), armY - 4, 14, 9);
  ctx.fillStyle = "#f0c8a0";
  ctx.beginPath();
  ctx.arc(0, -62, 17, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = b.bandana;
  ctx.beginPath();
  ctx.arc(0, -58, 15, 0, Math.PI);
  ctx.fill();
  ctx.fillStyle = "#2c2036";
  ctx.beginPath();
  ctx.arc(-6, -66, 2.6, 0, Math.PI * 2);
  ctx.arc(6, -66, 2.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#6b4226";
  ctx.beginPath();
  ctx.ellipse(0, -76, 26, 7, 0, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  rrect(-13, -94, 26, 20, 8);
  ctx.fill(); ctx.stroke();
  if (b.state === "aim" && b.t > b.aimT - 34 && Math.floor(b.t / 5) % 2 === 0) {
    ctx.fillStyle = "#ff4d4d";
    ctx.font = "bold 30px 'Marker Felt', 'Comic Sans MS', cursive";
    ctx.textAlign = "center";
    ctx.fillText("!", 0, -102);
  }
  ctx.restore();
}

function drawBag(g) {
  ctx.save();
  ctx.translate(g.x, g.y);
  ctx.rotate(Math.sin(g.t * 0.08) * 0.2);
  ctx.fillStyle = "#d9b168";
  ctx.strokeStyle = "#5a3620";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(0, 6, 20, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#b98b45";
  rrect(-8, -18, 16, 12, 5);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#5a3620";
  ctx.font = "bold 20px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.textAlign = "center";
  ctx.fillText("$", 0, 13);
  ctx.restore();
}

function drawFx() {
  for (const e of fx) {
    const k = e.t / e.ttl;
    if (e.kind === "flash" || e.kind === "bang") {
      const r = e.kind === "bang" ? 20 : 26;
      ctx.save();
      ctx.translate(e.x, e.y);
      ctx.rotate(e.t * 0.4);
      ctx.fillStyle = k < 0.5 ? "#fff3b0" : "#ffb347";
      ctx.beginPath();
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        const rr = i % 2 ? r * (1 - k) : r * 0.4;
        ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    } else if (e.kind === "impact") {
      ctx.strokeStyle = `rgba(90,54,32,${1 - k})`;
      ctx.lineWidth = 3;
      for (let i = 0; i < 6; i++) {
        const a = (i / 6) * Math.PI * 2 + 0.4;
        ctx.beginPath();
        ctx.moveTo(e.x + Math.cos(a) * 6, e.y + Math.sin(a) * 6);
        ctx.lineTo(e.x + Math.cos(a) * (10 + k * 12), e.y + Math.sin(a) * (10 + k * 12));
        ctx.stroke();
      }
    } else if (e.kind === "puff") {
      ctx.fillStyle = e.color;
      ctx.globalAlpha = 1 - k;
      ctx.beginPath();
      ctx.arc(e.x, e.y, e.r * (1 - k * 0.5), 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    } else if (e.kind === "pop") {
      ctx.fillStyle = e.color;
      ctx.globalAlpha = 1 - k;
      ctx.font = "bold 26px 'Marker Felt', 'Comic Sans MS', cursive";
      ctx.textAlign = "center";
      ctx.strokeStyle = "#2c2036";
      ctx.lineWidth = 4;
      ctx.strokeText(e.text, e.x, e.y);
      ctx.fillText(e.text, e.x, e.y);
      ctx.globalAlpha = 1;
    }
  }
}

function drawHeart(x, y, on) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = on ? "#ff4d5e" : "rgba(60,40,50,.5)";
  ctx.strokeStyle = "#2c2036";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.bezierCurveTo(-14, -8, -6, -18, 0, -8);
  ctx.bezierCurveTo(6, -18, 14, -8, 0, 6);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.restore();
}

function drawHud() {
  ctx.fillStyle = "rgba(44,32,54,.75)";
  rrect(14, 12, 210, 40, 10);
  ctx.fill();
  ctx.fillStyle = "#ffe8c2";
  ctx.font = "bold 24px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.textAlign = "left";
  ctx.fillText("SCORE " + String(score).padStart(5, "0"), 28, 40);
  ctx.fillStyle = "rgba(44,32,54,.75)";
  rrect(W / 2 - 62, 12, 124, 40, 10);
  ctx.fill();
  ctx.fillStyle = "#ffd93b";
  ctx.textAlign = "center";
  ctx.fillText("WAVE " + wave, W / 2, 40);
  ctx.fillStyle = "rgba(44,32,54,.75)";
  rrect(W - 150, 12, 136, 40, 10);
  ctx.fill();
  for (let i = 0; i < 3; i++) drawHeart(W - 118 + i * 40, 33, i < hearts);
  const usingHand = hand.seen > 0;
  let msg;
  if (!camReady) msg = "MOUSE MODE - CLICK TO SHOOT";
  else if (!usingHand) msg = "SHOW YOUR HAND TO THE CAMERA";
  else if (!hand.gun) msg = "MAKE A FINGER GUN";
  else msg = hammer ? "COCKED - DROP THUMB TO FIRE" : "RAISE THUMB TO RE-COCK";
  ctx.fillStyle = "rgba(44,32,54,.75)";
  rrect(14, H - 46, 380, 34, 10);
  ctx.fill();
  ctx.fillStyle = usingHand && hand.gun ? "#9dff8a" : "#ffd93b";
  ctx.font = "bold 17px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.textAlign = "left";
  ctx.fillText(msg, 28, H - 23);
}

function drawCrosshair() {
  ctx.save();
  ctx.translate(aim.x, aim.y);
  ctx.rotate(Math.sin(frame * 0.05) * 0.08);
  const cocked = hand.seen > 0 ? hammer : true;
  ctx.strokeStyle = cocked ? "#ff5533" : "#ffb347";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(0, 0, 18, 0, Math.PI * 2);
  ctx.stroke();
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(Math.cos(a) * 10, Math.sin(a) * 10);
    ctx.lineTo(Math.cos(a) * 26, Math.sin(a) * 26);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5533";
  ctx.beginPath();
  ctx.arc(0, 0, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawBoard(title, lines) {
  ctx.fillStyle = "rgba(28,21,34,.72)";
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.translate(W / 2, H / 2);
  ctx.rotate(-0.02);
  ctx.fillStyle = "#8a5a32";
  ctx.strokeStyle = "#3d2415";
  ctx.lineWidth = 8;
  rrect(-310, -150, 620, 300, 20);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#ffd93b";
  ctx.font = "bold 56px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.textAlign = "center";
  ctx.strokeStyle = "#5a2020";
  ctx.lineWidth = 8;
  ctx.strokeText(title, 0, -70);
  ctx.fillText(title, 0, -70);
  ctx.font = "bold 24px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.fillStyle = "#ffe8c2";
  lines.forEach((ln, i) => ctx.fillText(ln, 0, -18 + i * 40));
  ctx.restore();
}

function drawBanner() {
  const k = bannerT / 90;
  ctx.save();
  ctx.globalAlpha = Math.min(1, k * 3);
  ctx.translate(W / 2, 220);
  ctx.rotate(-0.03);
  ctx.fillStyle = "#8a5a32";
  ctx.strokeStyle = "#3d2415";
  ctx.lineWidth = 7;
  rrect(-190, -55, 380, 110, 16);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#ffd93b";
  ctx.font = "bold 60px 'Marker Felt', 'Comic Sans MS', cursive";
  ctx.textAlign = "center";
  ctx.strokeStyle = "#5a2020";
  ctx.lineWidth = 8;
  ctx.strokeText("WAVE " + wave, 0, 20);
  ctx.fillText("WAVE " + wave, 0, 20);
  ctx.restore();
}

function draw() {
  ctx.save();
  if (shake > 0.5) ctx.translate((Math.random() - 0.5) * shake, (Math.random() - 0.5) * shake);
  drawScene();
  for (const b of bandits) drawBandit(b);
  drawCovers();
  for (const g of bags) drawBag(g);
  drawFx();
  ctx.restore();
  if (redFlash > 0) {
    ctx.fillStyle = `rgba(255,40,40,${redFlash / 60})`;
    ctx.fillRect(0, 0, W, H);
  }
  if (phase === "menu") {
    drawBoard("AIR SHOOTOUT", [
      "AIM: point your index finger at the screen",
      "FIRE: drop your thumb like a gun hammer",
      "don't get shot, partner",
      "SHOOT TO START",
    ]);
  } else if (phase === "over") {
    drawBoard("GAME OVER", [
      "final score: " + score,
      "you made it to wave " + wave,
      "",
      "SHOOT TO TRY AGAIN",
    ]);
  } else {
    drawHud();
    if (phase === "wave") drawBanner();
  }
  drawCrosshair();
  if (camReady) pctx.drawImage(cam, 0, 0, pip.width, pip.height);
}

const STEP = 1000 / 60;
let last = performance.now(), acc = 0;
function loop(now) {
  acc = Math.min(acc + now - last, 100);
  last = now;
  while (acc >= STEP) { update(); acc -= STEP; }
  draw();
  sendFrame();
  requestAnimationFrame(loop);
}

function setStatus() {
  if (!wsReady) elStatus.textContent = "connecting to hand tracker...";
  else if (!camReady) elStatus.textContent = "no camera - mouse mode (click to shoot)";
  else elStatus.textContent = "point your finger gun at the screen and drop the thumb to fire";
}

function sendFrame() {
  if (!wsReady || !camReady || awaiting || !ws || ws.readyState !== 1) return;
  awaiting = true;
  gctx.drawImage(cam, 0, 0, grab.width, grab.height);
  grab.toBlob((blob) => {
    if (!blob) { awaiting = false; return; }
    blob.arrayBuffer().then((b) => {
      if (ws.readyState === 1) ws.send(b);
      else awaiting = false;
    });
  }, "image/jpeg", 0.7);
}

function connect() {
  ws = new WebSocket("ws://" + location.hostname + ":8765");
  ws.onopen = () => { wsReady = true; setStatus(); };
  ws.onclose = () => { wsReady = false; awaiting = false; setStatus(); setTimeout(connect, 1200); };
  ws.onmessage = (ev) => {
    awaiting = false;
    const data = JSON.parse(ev.data);
    if (data.hands && data.hands.length) {
      const h = data.hands[0];
      hand.seen = 45;
      hand.x = h.x;
      hand.y = h.y;
      hand.gun = h.gun;
      hand.thumb = h.thumb;
    }
  };
}

async function initCam() {
  if (location.search.includes("nocam")) { setStatus(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
    cam.srcObject = stream;
    await cam.play();
    camReady = true;
  } catch (e) {
    camReady = false;
  }
  setStatus();
}

cv.addEventListener("mousemove", (e) => {
  const r = cv.getBoundingClientRect();
  mouse.x = (e.clientX - r.left) * (W / r.width);
  mouse.y = (e.clientY - r.top) * (H / r.height);
});
cv.addEventListener("mousedown", () => { if (hand.seen <= 0) fire(); });
window.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); fire(); } });

connect();
initCam();
setStatus();
requestAnimationFrame(loop);
