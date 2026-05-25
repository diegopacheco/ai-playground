const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const W = canvas.width;
const H = canvas.height;
const SP = 2;

const overlay = document.getElementById("overlay");
const bigEl = document.getElementById("big");
const subEl = document.getElementById("sub");
const startEl = document.getElementById("start");
const scoreEl = document.getElementById("score");
const hpEl = document.getElementById("hpbar");

const palette = {
  G: "#3fb950", D: "#1f7a34", W: "#ffffff", B: "#0a0a14", M: "#0d4a1e",
  H: "#3a2e2a", F: "#ffcc99", E: "#dfeaff", P: "#1a1a2a", S: "#5b8cff",
  A: "#ffcc99", T: "#2b2b3a", K: "#101018", Y: "#ffd23f"
};

const heroArt = [
  "  GGGGGG  ",
  " GDDDDDDG ",
  " GWWDDWWG ",
  " GBWDDWBG ",
  " GDDDDDDG ",
  "  GMMMMG  ",
  " GGGGGGGG ",
  "GG GGGG GG",
  "GG GGGG GG",
  " G GGGG G ",
  "   GGGG   ",
  "  GG  GG  ",
  "  GG  GG  ",
  " DD    DD "
];

const devArt = [
  "   HHHH   ",
  "  HHHHHH  ",
  "  HFFFFH  ",
  "  EEFFEE  ",
  "  PFFFFP  ",
  "  FFFFFF  ",
  "   FFFF   ",
  "  SSSSSS  ",
  " SSSSSSSS ",
  " ASSSSSSA ",
  "  SSSSSS  ",
  "  TT  TT  ",
  "  TT  TT  ",
  "  KK  KK  "
];

function drawSprite(art, x, y, flip) {
  for (let r = 0; r < art.length; r++) {
    const row = art[r];
    for (let c = 0; c < row.length; c++) {
      const k = row[c];
      if (k === " ") continue;
      const col = palette[k];
      const cc = flip ? row.length - 1 - c : c;
      ctx.fillStyle = col;
      ctx.fillRect(Math.round(x + cc * SP), Math.round(y + r * SP), SP, SP);
    }
  }
}

const platforms = [
  { x: 0, y: 332, w: W, h: 28 },
  { x: 70, y: 250, w: 150, h: 12 },
  { x: 420, y: 250, w: 150, h: 12 },
  { x: 245, y: 175, w: 150, h: 12 }
];

const HERO_W = heroArt[0].length * SP;
const HERO_H = heroArt.length * SP;
const DEV_W = devArt[0].length * SP;
const DEV_H = devArt.length * SP;

const keys = {};
let state = "menu";
let player, slops, devs, particles, popups;
let score, hp, spawnTimer, fireTimer, hitFlash, elapsed;

function reset() {
  player = { x: W / 2 - HERO_W / 2, y: 332 - HERO_H, vx: 0, vy: 0, dir: 1, onGround: true, inv: 0, muzzle: 0 };
  slops = [];
  devs = [];
  particles = [];
  popups = [];
  score = 0;
  hp = 5;
  spawnTimer = 60;
  fireTimer = 0;
  hitFlash = 0;
  elapsed = 0;
  updateHud();
}

function updateHud() {
  scoreEl.textContent = "SCORE " + String(score).padStart(6, "0");
  hpEl.textContent = "█".repeat(hp) + "·".repeat(5 - hp);
}

function spawnDev() {
  const surfaces = platforms;
  const surf = surfaces[Math.floor(Math.random() * surfaces.length)];
  const fromLeft = Math.random() < 0.5;
  const speed = 0.7 + Math.min(1.6, elapsed / 2400);
  const x = fromLeft ? surf.x - DEV_W : surf.x + surf.w;
  if (surf.w < DEV_W + 8) return;
  devs.push({
    x: x,
    y: surf.y - DEV_H,
    vx: fromLeft ? speed : -speed,
    surf: surf,
    wob: Math.random() * 6
  });
}

function fire() {
  if (fireTimer > 0) return;
  fireTimer = 14;
  player.muzzle = 6;
  const mx = player.dir === 1 ? player.x + HERO_W : player.x - 8;
  slops.push({
    x: mx,
    y: player.y + 14,
    vx: player.dir * 6,
    vy: -0.6,
    life: 90,
    seed: Math.random() * 100
  });
}

function rectHit(ax, ay, aw, ah, bx, by, bw, bh) {
  return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

function burst(x, y, color, n) {
  for (let i = 0; i < n; i++) {
    particles.push({
      x: x,
      y: y,
      vx: (Math.random() - 0.5) * 5,
      vy: -Math.random() * 4 - 1,
      life: 24 + Math.random() * 16,
      color: color,
      s: SP + Math.floor(Math.random() * 2) * SP
    });
  }
}

function update() {
  elapsed++;

  player.vx = 0;
  if (keys["ArrowLeft"]) { player.vx = -2.4; player.dir = -1; }
  if (keys["ArrowRight"]) { player.vx = 2.4; player.dir = 1; }
  if (keys["z"] && player.onGround) { player.vy = -10; player.onGround = false; }

  player.vy += 0.6;
  if (player.vy > 12) player.vy = 12;
  player.x += player.vx;
  player.y += player.vy;

  if (player.x < 0) player.x = 0;
  if (player.x + HERO_W > W) player.x = W - HERO_W;

  player.onGround = false;
  for (const p of platforms) {
    if (player.vy >= 0 &&
        player.x + HERO_W > p.x + 2 && player.x < p.x + p.w - 2 &&
        player.y + HERO_H > p.y && player.y + HERO_H - player.vy <= p.y + 1) {
      player.y = p.y - HERO_H;
      player.vy = 0;
      player.onGround = true;
    }
  }
  if (player.y > H) { player.y = 0; player.vy = 0; }

  if (fireTimer > 0) fireTimer--;
  if (player.muzzle > 0) player.muzzle--;
  if (player.inv > 0) player.inv--;
  if (hitFlash > 0) hitFlash--;

  for (const s of slops) {
    s.x += s.vx;
    s.vy += 0.05;
    s.y += s.vy;
    s.life--;
  }

  spawnTimer--;
  const spawnRate = Math.max(28, 80 - Math.floor(elapsed / 60));
  if (spawnTimer <= 0) {
    spawnDev();
    spawnTimer = spawnRate;
  }

  for (const d of devs) {
    d.x += d.vx;
    d.wob += 0.3;
    d.y = d.surf.y - DEV_H + Math.sin(d.wob) * 1.5;
  }

  for (const s of slops) {
    if (s.life <= 0) continue;
    for (const d of devs) {
      if (d.dead) continue;
      if (rectHit(s.x - 5, s.y - 5, 12, 12, d.x + 4, d.y + 4, DEV_W - 8, DEV_H - 8)) {
        d.dead = true;
        s.life = 0;
        score += 100;
        burst(d.x + DEV_W / 2, d.y + DEV_H / 2, "#3fb950", 14);
        burst(d.x + DEV_W / 2, d.y + DEV_H / 2, "#6cff9e", 8);
        popups.push({ x: d.x + 4, y: d.y, t: "+100", life: 40 });
        updateHud();
      }
    }
  }

  if (player.inv <= 0) {
    for (const d of devs) {
      if (d.dead) continue;
      if (rectHit(player.x + 3, player.y + 3, HERO_W - 6, HERO_H - 6, d.x + 4, d.y + 2, DEV_W - 8, DEV_H - 4)) {
        hp--;
        player.inv = 70;
        hitFlash = 8;
        player.vy = -6;
        player.vx = d.vx > 0 ? 6 : -6;
        player.x += player.vx * 2;
        burst(player.x + HERO_W / 2, player.y + HERO_H / 2, "#ff5c8a", 12);
        updateHud();
        if (hp <= 0) gameOver();
        break;
      }
    }
  }

  for (const p of particles) {
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.25;
    p.life--;
  }
  for (const pu of popups) pu.life--;

  slops = slops.filter(s => s.life > 0 && s.x > -20 && s.x < W + 20);
  devs = devs.filter(d => !d.dead && d.x > -DEV_W - 40 && d.x < W + DEV_W + 40);
  particles = particles.filter(p => p.life > 0);
  popups = popups.filter(p => p.life > 0);
}

function drawBg() {
  ctx.fillStyle = "#1a1030";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#241546";
  for (let i = 0; i < 6; i++) {
    const bx = (i * 130 + 20) % W;
    ctx.fillRect(bx, 120, 70, 220);
    ctx.fillStyle = "#2e1b58";
    for (let wy = 130; wy < 320; wy += 22) {
      for (let wx = bx + 8; wx < bx + 64; wx += 18) {
        ctx.fillRect(wx, wy, 8, 10);
      }
    }
    ctx.fillStyle = "#241546";
  }
  ctx.fillStyle = "#ffd23f33";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#1a1030";
  ctx.fillRect(0, 0, W, H - 0.0001);
  ctx.fillStyle = "#1a1030";
}

function drawPlatforms() {
  for (const p of platforms) {
    ctx.fillStyle = "#3a2b66";
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.fillStyle = "#6c4dd6";
    ctx.fillRect(p.x, p.y, p.w, 4);
    ctx.fillStyle = "#241546";
    for (let gx = p.x + 6; gx < p.x + p.w - 4; gx += 16) {
      ctx.fillRect(gx, p.y + 6, 4, p.h - 8);
    }
  }
}

function drawSlop(s) {
  const wob = Math.sin((s.life + s.seed) * 0.5) * 1.5;
  ctx.fillStyle = "#1f7a34";
  ctx.fillRect(s.x - 6, s.y - 4 + wob, 12, 10);
  ctx.fillStyle = "#3fb950";
  ctx.fillRect(s.x - 5, s.y - 5, 10, 10);
  ctx.fillStyle = "#6cff9e";
  ctx.fillRect(s.x - 3, s.y - 3, 4, 4);
  ctx.fillStyle = "#3fb950";
  ctx.fillRect(s.x - s.vx, s.y + wob, 4, 4);
}

function draw() {
  drawBg();
  drawPlatforms();

  for (const s of slops) drawSlop(s);

  for (const d of devs) {
    drawSprite(devArt, d.x, d.y, d.vx > 0);
  }

  if (!(player.inv > 0 && Math.floor(player.inv / 4) % 2)) {
    drawSprite(heroArt, player.x, player.y, player.dir === -1);
    if (player.muzzle > 0) {
      const mx = player.dir === 1 ? player.x + HERO_W - 2 : player.x - 10;
      ctx.fillStyle = "#ffd23f";
      ctx.fillRect(mx, player.y + 12, 10, 8);
      ctx.fillStyle = "#fff";
      ctx.fillRect(mx + 2, player.y + 14, 4, 4);
    }
  }

  for (const p of particles) {
    ctx.globalAlpha = Math.min(1, p.life / 16);
    ctx.fillStyle = p.color;
    ctx.fillRect(p.x, p.y, p.s, p.s);
    ctx.globalAlpha = 1;
  }

  ctx.font = "12px 'Courier New', monospace";
  ctx.fillStyle = "#ffd23f";
  for (const pu of popups) {
    ctx.globalAlpha = Math.min(1, pu.life / 20);
    ctx.fillText(pu.t, pu.x, pu.y - (40 - pu.life) * 0.5);
    ctx.globalAlpha = 1;
  }

  if (hitFlash > 0) {
    ctx.fillStyle = "#ff5c8a44";
    ctx.fillRect(0, 0, W, H);
  }
}

function loop() {
  if (state === "play") {
    update();
    draw();
  }
  requestAnimationFrame(loop);
}

function startGame() {
  reset();
  state = "play";
  overlay.classList.add("hidden");
}

function gameOver() {
  state = "over";
  bigEl.textContent = "GAME OVER";
  subEl.textContent = "final score " + score;
  startEl.textContent = "PRESS ENTER";
  overlay.classList.remove("hidden");
}

document.addEventListener("keydown", e => {
  keys[e.key.toLowerCase()] = true;
  keys[e.key] = true;
  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " ", "z", "x"].includes(e.key) ||
      ["z", "x"].includes(e.key.toLowerCase())) {
    e.preventDefault();
  }
  if (e.key === "Enter" && (state === "menu" || state === "over")) startGame();
  if ((e.key === "x" || e.key === "X") && state === "play") fire();
});

document.addEventListener("keyup", e => {
  keys[e.key.toLowerCase()] = false;
  keys[e.key] = false;
});

reset();
state = "menu";
overlay.classList.remove("hidden");
loop();
