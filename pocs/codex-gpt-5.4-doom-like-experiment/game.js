const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const armorEl = document.getElementById("armor");
const armorBarEl = document.getElementById("armor-bar");
const healthEl = document.getElementById("health");
const healthBarEl = document.getElementById("health-bar");
const killsEl = document.getElementById("kills");
const fpsEl = document.getElementById("fps");
const statusEl = document.getElementById("status");
const flashEl = document.getElementById("flash");
const messageEl = document.getElementById("message");
const menuTextEl = document.getElementById("menu-text");
const startButton = document.getElementById("start");
const musicButton = document.getElementById("music");
const quitButton = document.getElementById("quit");
const reticleEl = document.getElementById("reticle");
const bgmEl = document.getElementById("bgm");
const touchButtons = [...document.querySelectorAll("[data-control]")];

const map = [
  "1111111111111111",
  "1000000000000001",
  "1011110111110101",
  "1010000100010101",
  "1010111101010101",
  "1000100001010001",
  "1110101111011101",
  "1000101000010001",
  "1011101011110101",
  "1000001000000101",
  "1011111011110101",
  "1000000010000001",
  "1011111010111101",
  "1000000000100001",
  "1000111111100001",
  "1111111111111111"
];

const enemySpawnPoints = [
  { x: 4.5, y: 1.5 },
  { x: 7.5, y: 1.5 },
  { x: 10.5, y: 1.5 }
];

const propSpawns = [
  { x: 3.5, y: 1.5, kind: "crate" },
  { x: 4.5, y: 3.5, kind: "barrel" },
  { x: 8.5, y: 3.5, kind: "torch" },
  { x: 11.5, y: 4.5, kind: "barrel" },
  { x: 5.5, y: 7.5, kind: "crate" },
  { x: 10.5, y: 8.5, kind: "torch" },
  { x: 3.5, y: 12.5, kind: "crate" },
  { x: 8.5, y: 12.5, kind: "barrel" },
  { x: 12.5, y: 13.5, kind: "torch" }
];

function buildEnemies() {
  return enemySpawnPoints.map((enemy, index) => ({ ...enemy, id: `enemy-${index}`, alive: true, cooldown: 0 }));
}

function buildProps() {
  return propSpawns.map((prop, index) => ({ ...prop, id: `prop-${index}`, active: true }));
}

const state = {
  active: false,
  keys: {},
  touch: {
    forward: false,
    back: false,
    left: false,
    right: false,
    turnLeft: false,
    turnRight: false
  },
  player: {
    x: 1.5,
    y: 1.5,
    angle: 0.3,
    pitch: 0,
    moveSpeed: 2.9,
    turnSpeed: 2.1,
    radius: 0.2,
    health: 100,
    armor: 100
  },
  enemies: buildEnemies(),
  props: buildProps(),
  items: [],
  door: { x: 13.5, y: 14.5, kind: "exit", active: true },
  soundEnabled: true,
  projectFlash: 0,
  fps: 0,
  lastTime: 0,
  dragging: false,
  shotKick: 0,
  hitPulse: 0
};

const fov = Math.PI / 3;
const maxDepth = 20;
const rayStep = 0.02;
const spriteCanvas = document.createElement("canvas");
spriteCanvas.width = 64;
spriteCanvas.height = 64;
const spriteCtx = spriteCanvas.getContext("2d");
let audioContext;
let musicLoopId = 0;

function isWall(x, y) {
  const mx = Math.floor(x);
  const my = Math.floor(y);
  if (my < 0 || my >= map.length || mx < 0 || mx >= map[0].length) return true;
  return map[my][mx] === "1";
}

function castRay(angle) {
  const sin = Math.sin(angle);
  const cos = Math.cos(angle);
  let depth = 0;

  while (depth < maxDepth) {
    const x = state.player.x + cos * depth;
    const y = state.player.y + sin * depth;
    if (isWall(x, y)) {
      const hitX = x - Math.floor(x);
      const hitY = y - Math.floor(y);
      const edge = Math.min(hitX, 1 - hitX, hitY, 1 - hitY);
      return { depth, edge };
    }
    depth += rayStep;
  }

  return { depth: maxDepth, edge: 0.5 };
}

function ensureAudio() {
  if (!audioContext) {
    const AudioCtor = window.AudioContext || window.webkitAudioContext;
    if (AudioCtor) audioContext = new AudioCtor();
  }
  if (audioContext?.state === "suspended") audioContext.resume();
}

function playSound(type) {
  if (!state.soundEnabled) return;
  ensureAudio();
  if (!audioContext) return;

  const now = audioContext.currentTime;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  const filter = audioContext.createBiquadFilter();

  oscillator.connect(filter);
  filter.connect(gain);
  gain.connect(audioContext.destination);

  if (type === "shot") {
    oscillator.type = "square";
    oscillator.frequency.setValueAtTime(180, now);
    oscillator.frequency.exponentialRampToValueAtTime(70, now + 0.08);
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(1400, now);
    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(0.12, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
    oscillator.start(now);
    oscillator.stop(now + 0.11);
    return;
  }

  if (type === "kill") {
    oscillator.type = "sawtooth";
    oscillator.frequency.setValueAtTime(190, now);
    oscillator.frequency.exponentialRampToValueAtTime(55, now + 0.22);
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(700, now);
    gain.gain.setValueAtTime(0.001, now);
    gain.gain.exponentialRampToValueAtTime(0.09, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.24);
    oscillator.start(now);
    oscillator.stop(now + 0.25);
    return;
  }

  oscillator.type = "triangle";
  oscillator.frequency.setValueAtTime(120, now);
  oscillator.frequency.exponentialRampToValueAtTime(60, now + 0.18);
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(900, now);
  gain.gain.setValueAtTime(0.001, now);
  gain.gain.exponentialRampToValueAtTime(0.08, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
  oscillator.start(now);
  oscillator.stop(now + 0.19);
}

function playMusicNote(frequency, startAt, duration, gainValue, type) {
  if (!audioContext || !state.soundEnabled) return;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  const filter = audioContext.createBiquadFilter();
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, startAt);
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(900, startAt);
  oscillator.connect(filter);
  filter.connect(gain);
  gain.connect(audioContext.destination);
  gain.gain.setValueAtTime(0.0001, startAt);
  gain.gain.exponentialRampToValueAtTime(gainValue, startAt + 0.03);
  gain.gain.exponentialRampToValueAtTime(0.0001, startAt + duration);
  oscillator.start(startAt);
  oscillator.stop(startAt + duration + 0.02);
}

function scheduleFallbackMusic() {
  if (!state.soundEnabled) return;
  ensureAudio();
  if (!audioContext) return;
  const now = audioContext.currentTime;
  const bass = [65.41, 73.42, 82.41, 73.42, 61.74, 65.41, 82.41, 98.0];
  const lead = [196, 220, 246.94, 220, 174.61, 196, 220, 246.94];
  bass.forEach((note, index) => {
    const t = now + index * 0.4;
    playMusicNote(note, t, 0.34, 0.025, "triangle");
    playMusicNote(lead[index], t + 0.08, 0.18, 0.012, "square");
  });
  musicLoopId = window.setTimeout(scheduleFallbackMusic, 3200);
}

function startMusic() {
  if (!state.soundEnabled) return;
  stopMusic();
  if (bgmEl && typeof bgmEl.play === "function") {
    try {
      bgmEl.play();
    } catch (_error) {
      scheduleFallbackMusic();
    }
    window.setTimeout(() => {
      if (state.soundEnabled && (!bgmEl || bgmEl.playing === false)) {
        scheduleFallbackMusic();
      }
    }, 600);
    return;
  }
  scheduleFallbackMusic();
}

function stopMusic() {
  window.clearTimeout(musicLoopId);
  musicLoopId = 0;
  if (bgmEl && typeof bgmEl.stop === "function") bgmEl.stop();
}

function syncMusicButton() {
  if (musicButton) musicButton.textContent = `Sound: ${state.soundEnabled ? "On" : "Off"}`;
}

function toggleSound() {
  state.soundEnabled = !state.soundEnabled;
  syncMusicButton();
  if (state.soundEnabled) {
    startMusic();
  } else {
    stopMusic();
  }
}

function createEnemySprite(frame = 0) {
  spriteCtx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
  spriteCtx.imageSmoothingEnabled = false;

  const paint = (x, y, w, h, color) => {
    spriteCtx.fillStyle = color;
    spriteCtx.fillRect(x, y, w, h);
  };

  paint(22, 4, 20, 8, "#2f3e21");
  paint(18, 10, 28, 10, "#6f8760");
  paint(20, 12, 6, 4, "#ff5442");
  paint(38, 12, 6, 4, "#ff5442");
  paint(26, 16, 12, 3, "#191d14");
  paint(18, 20, 28, 18, "#4e6142");
  paint(14, 22 + frame, 8, 22, "#384830");
  paint(42, 22 - frame, 8, 22, "#384830");
  paint(22, 24, 20, 16, "#242b21");
  paint(24, 28, 16, 5, "#98b083");
  paint(10, 30 + frame, 8, 10, "#90a676");
  paint(46, 28 - frame, 8, 10, "#90a676");
  paint(18, 40, 12, 18, "#313a2c");
  paint(34, 40, 12, 18, "#313a2c");
  paint(14, 52, 12, 8, "#201e15");
  paint(38, 52, 12, 8, "#201e15");
  paint(8, 26, 8, 14, "#8b9780");
  paint(48, 24, 8, 16, "#8b9780");
  paint(6, 24, 4, 18, "#c9d3b2");
  paint(54, 22, 4, 18, "#c9d3b2");
  paint(0, 58, 64, 4, "rgba(0,0,0,0.2)");

  return spriteCanvas;
}

function createBarrelSprite() {
  spriteCtx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
  spriteCtx.imageSmoothingEnabled = false;

  const paint = (x, y, w, h, color) => {
    spriteCtx.fillStyle = color;
    spriteCtx.fillRect(x, y, w, h);
  };

  paint(18, 18, 28, 34, "#6e341d");
  paint(16, 22, 32, 6, "#b86d34");
  paint(16, 34, 32, 6, "#b86d34");
  paint(16, 46, 32, 6, "#b86d34");
  paint(22, 20, 4, 30, "#c78647");
  paint(38, 20, 4, 30, "#4b1f0e");
  return spriteCanvas;
}

function createCrateSprite() {
  spriteCtx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
  spriteCtx.imageSmoothingEnabled = false;

  const paint = (x, y, w, h, color) => {
    spriteCtx.fillStyle = color;
    spriteCtx.fillRect(x, y, w, h);
  };

  paint(14, 20, 36, 30, "#7c5a30");
  paint(18, 24, 28, 22, "#936b39");
  paint(30, 20, 4, 30, "#c79752");
  paint(14, 32, 36, 4, "#c79752");
  paint(18, 24, 4, 22, "#4f3516");
  paint(42, 24, 4, 22, "#4f3516");
  return spriteCanvas;
}

function createTorchSprite(frame = 0) {
  spriteCtx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
  spriteCtx.imageSmoothingEnabled = false;

  const paint = (x, y, w, h, color) => {
    spriteCtx.fillStyle = color;
    spriteCtx.fillRect(x, y, w, h);
  };

  paint(29, 18, 6, 34, "#1a1612");
  paint(26, 14, 12, 8, "#3c2b1a");
  paint(22, 4 + frame, 20, 14, "rgba(255,170,70,0.85)");
  paint(26, 0 + frame, 12, 14, "rgba(255,230,160,0.92)");
  paint(28, 8 + frame, 8, 6, "rgba(255,90,30,0.95)");
  return spriteCanvas;
}

function createDoorSprite() {
  spriteCtx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
  spriteCtx.imageSmoothingEnabled = false;
  const paint = (x, y, w, h, color) => {
    spriteCtx.fillStyle = color;
    spriteCtx.fillRect(x, y, w, h);
  };
  paint(14, 6, 36, 52, "#4c5f58");
  paint(18, 10, 28, 44, "#79978c");
  paint(24, 18, 16, 22, "#98f0bf");
  paint(24, 44, 16, 6, "#203028");
  paint(42, 30, 4, 4, "#d7e0db");
  return spriteCanvas;
}

function createMedkitSprite() {
  spriteCtx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
  spriteCtx.imageSmoothingEnabled = false;
  const paint = (x, y, w, h, color) => {
    spriteCtx.fillStyle = color;
    spriteCtx.fillRect(x, y, w, h);
  };
  paint(18, 22, 28, 24, "#daddd8");
  paint(22, 26, 20, 16, "#f7faf5");
  paint(28, 28, 8, 12, "#d94141");
  paint(24, 32, 16, 4, "#d94141");
  paint(24, 18, 16, 6, "#6e7a74");
  return spriteCanvas;
}

function hasLineOfSight(fromX, fromY, toX, toY, padding = 0.28) {
  const dx = toX - fromX;
  const dy = toY - fromY;
  const distance = Math.hypot(dx, dy);
  const steps = Math.max(1, Math.ceil(distance / 0.04));

  for (let i = 1; i < steps; i += 1) {
    const t = i / steps;
    const x = fromX + dx * t;
    const y = fromY + dy * t;
    if (Math.hypot(x - toX, y - toY) <= padding) {
      return true;
    }
    if (isWall(x, y)) {
      return false;
    }
  }

  return true;
}

function hitsEnemy(x, y, padding = 0.42) {
  for (const enemy of state.enemies) {
    if (!enemy.alive) continue;
    if (Math.hypot(enemy.x - x, enemy.y - y) < padding) return true;
  }
  return false;
}

function hitsCrate(x, y, padding = 0.45) {
  for (const prop of state.props) {
    if (!prop.active || prop.kind !== "crate") continue;
    if (Math.hypot(prop.x - x, prop.y - y) < padding) return true;
  }
  return false;
}

function movePlayer(dt) {
  const sprint = state.keys.shift ? 1.55 : 1;
  const speed = state.player.moveSpeed * sprint * dt;
  const turnSpeed = state.player.turnSpeed * dt;

  if (state.keys.arrowleft || state.touch.turnLeft) state.player.angle -= turnSpeed;
  if (state.keys.arrowright || state.touch.turnRight) state.player.angle += turnSpeed;

  let moveX = 0;
  let moveY = 0;
  const forwardX = Math.cos(state.player.angle);
  const forwardY = Math.sin(state.player.angle);
  const sideX = Math.cos(state.player.angle + Math.PI / 2);
  const sideY = Math.sin(state.player.angle + Math.PI / 2);

  if (state.keys.w || state.touch.forward) {
    moveX += forwardX * speed;
    moveY += forwardY * speed;
  }
  if (state.keys.s || state.touch.back) {
    moveX -= forwardX * speed;
    moveY -= forwardY * speed;
  }
  if (state.keys.a || state.touch.left || state.keys.q) {
    moveX -= sideX * speed;
    moveY -= sideY * speed;
  }
  if (state.keys.d || state.touch.right || state.keys.e) {
    moveX += sideX * speed;
    moveY += sideY * speed;
  }

  const nextX = state.player.x + moveX;
  const nextY = state.player.y + moveY;

  if (
    !isWall(nextX + Math.sign(moveX) * state.player.radius, state.player.y) &&
    !isWall(nextX, state.player.y) &&
    !hitsEnemy(nextX, state.player.y) &&
    !hitsCrate(nextX, state.player.y)
  ) {
    state.player.x = nextX;
  }
  if (
    !isWall(state.player.x, nextY + Math.sign(moveY) * state.player.radius) &&
    !isWall(state.player.x, nextY) &&
    !hitsEnemy(state.player.x, nextY) &&
    !hitsCrate(state.player.x, nextY)
  ) {
    state.player.y = nextY;
  }
}

function damagePlayer(amount) {
  let remaining = amount;
  if (state.player.armor > 0) {
    const absorbed = Math.min(state.player.armor, remaining * 0.7);
    state.player.armor -= absorbed;
    remaining -= absorbed;
  }
  state.player.health = Math.max(0, state.player.health - remaining);
  playSound("hurt");
  flashEl.classList.add("active");
  setTimeout(() => flashEl.classList.remove("active"), 80);
}

function updateEnemies(dt) {
  let living = 0;
  for (const enemy of state.enemies) {
    if (!enemy.alive) continue;
    living += 1;
    enemy.cooldown = Math.max(0, enemy.cooldown - dt);
    const dx = enemy.x - state.player.x;
    const dy = enemy.y - state.player.y;
    const distance = Math.hypot(dx, dy);
    const angleToPlayer = Math.atan2(dy, dx);
    const facing = normalizeAngle(angleToPlayer - state.player.angle);

    if (distance > 1.2 && distance < 8 && Math.abs(facing) < 0.85) {
      enemy.x -= (dx / distance) * dt * 0.45;
      enemy.y -= (dy / distance) * dt * 0.45;
      if (isWall(enemy.x, enemy.y)) {
        enemy.x += (dx / distance) * dt * 0.45;
        enemy.y += (dy / distance) * dt * 0.45;
      }
    }

    const los = castRay(angleToPlayer);
    if (distance < los.depth + 0.2 && distance < 7 && enemy.cooldown === 0) {
      enemy.cooldown = 1.1;
      damagePlayer(8);
    }
  }

  killsEl.textContent = `🎯 ${state.enemies.length - living} / ${state.enemies.length}`;
  if (state.player.health <= 0) {
    endGame("Game Over");
  } else if (living === 0) {
    endGame("Jungle secure");
  }
}

function normalizeAngle(angle) {
  let value = angle;
  while (value < -Math.PI) value += Math.PI * 2;
  while (value > Math.PI) value -= Math.PI * 2;
  return value;
}

function fire() {
  if (!state.active) return;
  let bestTarget = null;
  let bestScore = Infinity;
  state.shotKick = 1;
  playSound("shot");
  const aimY = canvas.height / 2 + state.player.pitch * canvas.height * 0.32;
  const centerRayDepth = castRay(state.player.angle).depth;

  for (const enemy of state.enemies) {
    if (!enemy.alive) continue;
    const projection = projectSprite(enemy.x, enemy.y);
    const size = (canvas.height / projection.distance) * 1.45;
    const centerDx = Math.abs(projection.screenX - canvas.width / 2);
    const centerDy = Math.abs(projection.screenY - aimY);
    if (centerDx > size * 0.35) continue;
    if (centerDy > size * 0.45) continue;
    if (!hasLineOfSight(state.player.x, state.player.y, enemy.x, enemy.y)) continue;
    const score = centerDx + centerDy + projection.distance * 8;
    if (score < bestScore) {
      bestScore = score;
      bestTarget = enemy;
    }
  }

  for (const prop of state.props) {
    if (!prop.active || prop.kind !== "crate") continue;
    const projection = projectSprite(prop.x, prop.y);
    const size = canvas.height / projection.distance;
    const centerDx = Math.abs(projection.screenX - canvas.width / 2);
    const centerDy = Math.abs(projection.screenY - aimY);
    if (centerDx > size * 0.75) continue;
    if (centerDy > size * 0.55) continue;
    if (projection.distance > centerRayDepth + 0.7) continue;
    if (!hasLineOfSight(state.player.x, state.player.y, prop.x, prop.y, 0.55)) continue;
    const score = centerDx + centerDy + projection.distance * 6;
    if (score < bestScore) {
      bestScore = score;
      bestTarget = prop;
    }
  }

  flashEl.classList.add("active");
  setTimeout(() => flashEl.classList.remove("active"), 50);
  if (bestTarget) {
    if ("alive" in bestTarget) {
      bestTarget.alive = false;
      state.hitPulse = 1;
      playSound("kill");
      statusEl.textContent = "target dropped";
    } else {
      bestTarget.active = false;
      state.items.push({ x: bestTarget.x, y: bestTarget.y, kind: "medkit", active: true });
      playSound("kill");
      statusEl.textContent = "life drop";
    }
  } else {
    statusEl.textContent = "miss";
  }
}

function drawBackground() {
  const sky = ctx.createLinearGradient(0, 0, 0, canvas.height * 0.5);
  sky.addColorStop(0, "#7f8d86");
  sky.addColorStop(0.45, "#57655f");
  sky.addColorStop(1, "#2a312f");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, canvas.width, canvas.height / 2);

  const floor = ctx.createLinearGradient(0, canvas.height / 2, 0, canvas.height);
  floor.addColorStop(0, "#3b3d3d");
  floor.addColorStop(1, "#161616");
  ctx.fillStyle = floor;
  ctx.fillRect(0, canvas.height / 2, canvas.width, canvas.height / 2);

  for (let i = 0; i < 9; i += 1) {
    const towerX = i * 122 - 6;
    const towerHeight = 82 + (i % 4) * 36;
    ctx.fillStyle = "rgba(39, 44, 42, 0.9)";
    ctx.fillRect(towerX, canvas.height * 0.5 - towerHeight, 98, towerHeight);
    ctx.fillStyle = "rgba(58, 66, 63, 0.76)";
    ctx.fillRect(towerX + 8, canvas.height * 0.5 - towerHeight + 14, 82, towerHeight - 18);
    ctx.fillStyle = "rgba(123, 143, 133, 0.22)";
    ctx.fillRect(towerX + 16, canvas.height * 0.5 - towerHeight + 18, 10, towerHeight - 28);
    ctx.fillRect(towerX + 38, canvas.height * 0.5 - towerHeight + 26, 10, towerHeight - 38);
    ctx.fillRect(towerX + 62, canvas.height * 0.5 - towerHeight + 20, 10, towerHeight - 30);
  }

  for (let i = 0; i < 12; i += 1) {
    const shackX = i * 88 + (i % 2) * 10;
    const shackY = canvas.height * 0.5 - 22 - (i % 3) * 6;
    ctx.fillStyle = "rgba(55, 59, 57, 0.96)";
    ctx.fillRect(shackX, shackY, 70, 26);
    ctx.fillStyle = "rgba(82, 90, 86, 0.96)";
    ctx.beginPath();
    ctx.moveTo(shackX - 4, shackY);
    ctx.lineTo(shackX + 35, shackY - 12);
    ctx.lineTo(shackX + 74, shackY);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "rgba(154, 210, 190, 0.16)";
    ctx.fillRect(shackX + 10, shackY + 8, 10, 8);
    ctx.fillRect(shackX + 28, shackY + 10, 12, 6);
    ctx.fillRect(shackX + 48, shackY + 7, 8, 10);
  }

  ctx.fillStyle = "rgba(54, 78, 48, 0.55)";
  for (let i = 0; i < 16; i += 1) {
    const x = i * 70 + (i % 2) * 18;
    const h = 28 + (i % 4) * 18;
    ctx.beginPath();
    ctx.moveTo(x, canvas.height * 0.5);
    ctx.lineTo(x + 10, canvas.height * 0.5 - h);
    ctx.lineTo(x + 22, canvas.height * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x + 20, canvas.height * 0.5);
    ctx.lineTo(x + 30, canvas.height * 0.5 - h * 1.1);
    ctx.lineTo(x + 42, canvas.height * 0.5);
    ctx.closePath();
    ctx.fill();
  }

  ctx.strokeStyle = "rgba(102, 190, 158, 0.22)";
  ctx.lineWidth = 2;
  for (let i = 0; i < 7; i += 1) {
    const x = 40 + i * 150;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.bezierCurveTo(x + 20, 80, x - 30, 170, x + 12, canvas.height * 0.5);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(210, 220, 216, 0.18)";
  for (let i = 0; i < 8; i += 1) {
    const y = 80 + i * 28;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y + (i % 2 === 0 ? 6 : -6));
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(140, 150, 142, 0.16)";
  ctx.lineWidth = 2;
  for (let i = 0; i < 9; i += 1) {
    const y = canvas.height * 0.56 + i * 24;
    const inset = i * 38;
    ctx.beginPath();
    ctx.moveTo(inset, y);
    ctx.lineTo(canvas.width - inset, y);
    ctx.stroke();
  }

  for (let i = 0; i < 17; i += 1) {
    const x = i * (canvas.width / 16);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.028)";
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, canvas.height * 0.56);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(70, 112, 84, 0.18)";
  for (let i = 0; i < 10; i += 1) {
    const x = i * 110;
    ctx.fillRect(x, canvas.height * 0.12, 3, canvas.height * 0.24);
  }

  ctx.fillStyle = "#243122";
  ctx.fillRect(0, canvas.height * 0.74, canvas.width, canvas.height * 0.26);
  ctx.fillStyle = "#2f472a";
  for (let i = 0; i < 24; i += 1) {
    const x = i * 42;
    const h = 18 + (i % 4) * 10;
    ctx.fillRect(x, canvas.height * 0.78 - h, 28, h);
  }
  ctx.fillStyle = "#3f6a39";
  for (let i = 0; i < 28; i += 1) {
    const x = i * 34 + (i % 2) * 4;
    const h = 10 + (i % 5) * 8;
    ctx.fillRect(x, canvas.height * 0.82 - h, 14, h);
  }
  ctx.fillStyle = "#1b2218";
  for (let i = 0; i < 16; i += 1) {
    const x = i * 62;
    ctx.fillRect(x, canvas.height * 0.86, 20, canvas.height * 0.14);
  }
}

function drawWorld() {
  drawBackground();
  const depthBuffer = new Array(canvas.width);

  for (let x = 0; x < canvas.width; x += 1) {
    const cameraX = (2 * x) / canvas.width - 1;
    const angle = state.player.angle + cameraX * (fov / 2);
    const ray = castRay(angle);
    const correctedDepth = ray.depth * Math.cos(cameraX * (fov / 2));
    depthBuffer[x] = correctedDepth;
    const wallHeight = Math.min(canvas.height, canvas.height / Math.max(correctedDepth, 0.0001));
    const shade = Math.max(0.18, 1 - correctedDepth / maxDepth);
    const edgeGlow = Math.max(0, 0.2 - ray.edge) * 4.4;
    const panelBand = ((Math.floor(ray.depth * 3.4) + Math.floor(ray.edge * 16)) % 2) * 14;
    const wetGlow = Math.sin(ray.depth * 4 + ray.edge * 20) * 7 + 7;
    const red = Math.floor((74 + edgeGlow * 22 + panelBand * 0.2) * shade);
    const green = Math.floor((78 + edgeGlow * 24 + panelBand * 0.25 + wetGlow) * shade);
    const blue = Math.floor((82 + edgeGlow * 26 + panelBand * 0.3 + wetGlow) * shade);
    ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
    ctx.fillRect(x, (canvas.height - wallHeight) / 2, 1, wallHeight);
    if (x % 24 === 0) {
      ctx.fillStyle = `rgba(188, 198, 194, ${Math.max(0.06, 0.25 - correctedDepth * 0.03)})`;
      ctx.fillRect(x, (canvas.height - wallHeight) / 2, 1, wallHeight);
    }
    if (wallHeight > 120 && x % 37 < 3) {
      ctx.fillStyle = `rgba(30, 34, 32, ${Math.max(0.16, 0.7 - correctedDepth * 0.08)})`;
      ctx.fillRect(x, (canvas.height - wallHeight) / 2 + wallHeight * 0.15, 1, wallHeight * 0.7);
    }
  }

  const sprites = [
    ...state.props.filter((prop) => prop.active).map((prop) => ({ ...projectSprite(prop.x, prop.y), kind: "prop", prop })),
    ...state.items.filter((item) => item.active).map((item) => ({ ...projectSprite(item.x, item.y), kind: "item", item })),
    ...(state.door.active ? [{ ...projectSprite(state.door.x, state.door.y), kind: "door", door: state.door }] : []),
    ...state.enemies
      .filter((enemy) => enemy.alive)
      .map((enemy) => ({ ...projectSprite(enemy.x, enemy.y), kind: "enemy", enemy }))
  ]
    .filter((sprite) => sprite && Math.abs(sprite.angle) < fov * 0.7)
    .sort((a, b) => b.distance - a.distance);

  for (const sprite of sprites) {
    renderSprite(sprite, depthBuffer);
  }

  drawWeapon();
}

function projectSprite(x, y) {
  const dx = x - state.player.x;
  const dy = y - state.player.y;
  const distance = Math.hypot(dx, dy);
  const angle = normalizeAngle(Math.atan2(dy, dx) - state.player.angle);
  const screenX = ((angle + fov / 2) / fov) * canvas.width;
  const screenY = canvas.height / 2;
  return { x, y, distance, angle, screenX, screenY };
}

function renderSprite(sprite, depthBuffer) {
  const sizeScale = sprite.kind === "enemy" ? 1.45 : sprite.kind === "door" ? 1.2 : sprite.kind === "item" ? 0.7 : 1.05;
  const size = (canvas.height / sprite.distance) * sizeScale;
  const screenX = sprite.screenX;
  const half = size / 2;
  const startX = Math.floor(screenX - half);
  const endX = Math.floor(screenX + half);
  const top = sprite.screenY - half * 0.85;

  const texture = getSpriteTexture(sprite);
  const stripeWidth = Math.max(1, Math.ceil((endX - startX) / texture.width));

  for (let x = startX; x < endX; x += 1) {
    if (x < 0 || x >= canvas.width || depthBuffer[x] < sprite.distance) continue;
    const srcX = Math.floor(((x - startX) / Math.max(endX - startX, 1)) * texture.width);
    ctx.drawImage(texture, srcX, 0, 1, texture.height, x, top, stripeWidth, size);
  }
}

function getSpriteTexture(sprite) {
  if (sprite.kind === "enemy") {
    const frame = Math.sin(state.lastTime * 0.012 + sprite.enemy.x) > 0 ? 0 : 2;
    return createEnemySprite(frame);
  }
  if (sprite.kind === "door") return createDoorSprite();
  if (sprite.kind === "item") return createMedkitSprite();
  if (sprite.prop.kind === "barrel") return createBarrelSprite();
  if (sprite.prop.kind === "crate") return createCrateSprite();
  const frame = Math.sin(state.lastTime * 0.02 + sprite.prop.x) > 0 ? 0 : 1;
  return createTorchSprite(frame);
}

function drawWeapon() {
  const bob = Math.sin(state.lastTime * 0.008) * 8;
  const recoil = state.shotKick * 22;
  const baseY = canvas.height * 0.72 + bob + recoil;

  ctx.fillStyle = "rgba(0, 0, 0, 0.28)";
  ctx.beginPath();
  ctx.ellipse(canvas.width * 0.5, canvas.height * 0.94, canvas.width * 0.14, canvas.height * 0.04, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#11161b";
  ctx.beginPath();
  ctx.moveTo(canvas.width * 0.38, canvas.height);
  ctx.lineTo(canvas.width * 0.43, baseY + canvas.height * 0.06);
  ctx.lineTo(canvas.width * 0.57, baseY + canvas.height * 0.06);
  ctx.lineTo(canvas.width * 0.62, canvas.height);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#1f2a33";
  ctx.fillRect(canvas.width * 0.41, baseY + canvas.height * 0.015, canvas.width * 0.18, canvas.height * 0.11);
  ctx.fillStyle = "#0b0f13";
  ctx.fillRect(canvas.width * 0.445, baseY + canvas.height * 0.03, canvas.width * 0.11, canvas.height * 0.12);

  ctx.fillStyle = "#697783";
  ctx.fillRect(canvas.width * 0.468, baseY - canvas.height * 0.035, canvas.width * 0.064, canvas.height * 0.13);
  ctx.fillStyle = "#c9d0d6";
  ctx.fillRect(canvas.width * 0.478, baseY - canvas.height * 0.027, canvas.width * 0.044, canvas.height * 0.09);

  ctx.fillStyle = "#29343d";
  ctx.fillRect(canvas.width * 0.485, baseY - canvas.height * 0.11, canvas.width * 0.03, canvas.height * 0.08);
  ctx.fillStyle = "#8fd1df";
  ctx.fillRect(canvas.width * 0.492, baseY - canvas.height * 0.12, canvas.width * 0.016, canvas.height * 0.028);

  ctx.fillStyle = "#7b420e";
  ctx.beginPath();
  ctx.moveTo(canvas.width * 0.425, baseY + canvas.height * 0.08);
  ctx.lineTo(canvas.width * 0.46, canvas.height);
  ctx.lineTo(canvas.width * 0.49, canvas.height);
  ctx.lineTo(canvas.width * 0.47, baseY + canvas.height * 0.05);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(canvas.width * 0.575, baseY + canvas.height * 0.08);
  ctx.lineTo(canvas.width * 0.54, canvas.height);
  ctx.lineTo(canvas.width * 0.51, canvas.height);
  ctx.lineTo(canvas.width * 0.53, baseY + canvas.height * 0.05);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#7e8d95";
  ctx.fillRect(canvas.width * 0.446, baseY + canvas.height * 0.05, canvas.width * 0.108, canvas.height * 0.016);

  if (flashEl.classList.contains("active") && state.active) {
    ctx.fillStyle = "rgba(255, 225, 170, 0.92)";
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.5, baseY - canvas.height * 0.13);
    ctx.lineTo(canvas.width * 0.47, baseY - canvas.height * 0.19);
    ctx.lineTo(canvas.width * 0.53, baseY - canvas.height * 0.19);
    ctx.closePath();
    ctx.fill();
  }
}

function updateHud() {
  armorEl.textContent = Math.round(state.player.armor);
  healthEl.textContent = Math.round(state.player.health);
  armorBarEl.style.width = `${Math.max(0, Math.min(100, state.player.armor))}%`;
  healthBarEl.style.width = `${Math.max(0, Math.min(100, state.player.health))}%`;
  fpsEl.textContent = `${state.fps} FPS`;
  const remaining = state.enemies.filter((enemy) => enemy.alive).length;
  statusEl.textContent = remaining === 0 ? "Sector clean" : `${remaining} signatures left`;
}

function collectItems() {
  for (const item of state.items) {
    if (!item.active) continue;
    if (Math.hypot(item.x - state.player.x, item.y - state.player.y) < 0.8) {
      item.active = false;
      state.player.health = Math.min(100, state.player.health + 25);
      playSound("hurt");
      statusEl.textContent = "life restored";
    }
  }
}

function checkExit() {
  if (state.door.active && Math.hypot(state.door.x - state.player.x, state.door.y - state.player.y) < 0.9) {
    endGame("Jungle secure");
  }
}

function updateEffects(dt) {
  state.shotKick = Math.max(0, state.shotKick - dt * 6);
  state.hitPulse = Math.max(0, state.hitPulse - dt * 3);
  reticleEl.style.transform = `translate(-50%, calc(-50% + ${Math.round(state.player.pitch * 120)}px))`;
}

function endGame(text) {
  state.active = false;
  document.exitPointerLock?.();
  openMenu(text, "Press Enter or Start to run again.");
}

function resetGame() {
  state.player.x = 1.5;
  state.player.y = 1.5;
  state.player.angle = 0.3;
  state.player.pitch = 0;
  state.player.health = 100;
  state.player.armor = 100;
  state.enemies = buildEnemies();
  state.props = buildProps();
  state.items = [];
  state.door = { x: 13.5, y: 14.5, kind: "exit", active: true };
  state.shotKick = 0;
  state.hitPulse = 0;
  state.active = true;
  messageEl.classList.add("hidden");
  statusEl.textContent = "Sweep the hall";
  startButton.textContent = "Start";
  menuTextEl.textContent = "WASD to move, mouse to turn, click to fire, Shift to sprint. Use the Sound button or press M to toggle audio.";
}

function frame(time) {
  if (!state.lastTime) state.lastTime = time;
  const dt = Math.min((time - state.lastTime) / 1000, 0.04);
  state.lastTime = time;
  state.fps = Math.round(1 / Math.max(dt, 0.0001));
  updateEffects(dt);

  if (state.active) {
    movePlayer(dt);
    updateEnemies(dt);
    collectItems();
    checkExit();
  }

  drawWorld();
  updateHud();
  requestAnimationFrame(frame);
}

document.addEventListener("keydown", (event) => {
  state.keys[event.key.toLowerCase()] = true;
  if (event.key === " ") fire();
  if (event.key.toLowerCase() === "m") toggleSound();
  if (event.key === "Enter") {
    if (state.active) {
      openMenu("Paused", "Press Enter or Start to continue.");
    } else {
      startGame();
    }
  }
  if (event.key === "Escape" && state.active) {
    openMenu("Paused", "Press Enter or Start to continue.");
  }
});

document.addEventListener("keyup", (event) => {
  state.keys[event.key.toLowerCase()] = false;
});

document.addEventListener("mousemove", (event) => {
  if (document.pointerLockElement === canvas && state.active) {
    state.player.angle += event.movementX * 0.0027;
    state.player.pitch = Math.max(-0.35, Math.min(0.35, state.player.pitch + event.movementY * 0.0013));
  }
});

canvas.addEventListener("mousedown", () => {
  state.dragging = true;
});

document.addEventListener("mouseup", () => {
  state.dragging = false;
});

canvas.addEventListener("mouseleave", () => {
  state.dragging = false;
});

canvas.addEventListener("mousemove", (event) => {
  if (document.pointerLockElement !== canvas && state.dragging && state.active) {
    state.player.angle += event.movementX * 0.01;
    state.player.pitch = Math.max(-0.35, Math.min(0.35, state.player.pitch + event.movementY * 0.0022));
  }
});

canvas.addEventListener("click", () => {
  if (!state.active) return;
  if (document.pointerLockElement !== canvas && !("ontouchstart" in window)) {
    canvas.requestPointerLock();
  }
  fire();
});

startButton.addEventListener("click", () => {
  startGame();
});

musicButton.addEventListener("click", () => {
  toggleSound();
});

quitButton.addEventListener("click", () => {
  quitGame();
});

function setControl(name, pressed) {
  if (name === "fire") {
    if (pressed) fire();
    return;
  }
  if (name === "turn-left") state.touch.turnLeft = pressed;
  if (name === "turn-right") state.touch.turnRight = pressed;
  if (name === "forward") state.touch.forward = pressed;
  if (name === "back") state.touch.back = pressed;
  if (name === "left") state.touch.left = pressed;
  if (name === "right") state.touch.right = pressed;
}

for (const button of touchButtons) {
  const control = button.dataset.control;
  const press = (event) => {
    event.preventDefault();
    setControl(control, true);
    button.classList.add("active");
  };
  const release = (event) => {
    event.preventDefault();
    setControl(control, false);
    button.classList.remove("active");
  };
  button.addEventListener("pointerdown", press);
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("pointerleave", release);
}

function openMenu(title, text) {
  state.active = false;
  document.exitPointerLock?.();
  messageEl.classList.remove("hidden");
  messageEl.querySelector("h2").textContent = title;
  menuTextEl.textContent = `${text} Sound button or M toggles audio.`;
}

function startGame() {
  startMusic();
  if (messageEl.querySelector("h2").textContent === "Paused") {
    state.active = true;
    messageEl.classList.add("hidden");
    canvas.requestPointerLock?.();
    return;
  }
  resetGame();
  canvas.requestPointerLock?.();
}

function quitGame() {
  state.active = false;
  document.exitPointerLock?.();
  state.keys = {};
  state.touch.forward = false;
  state.touch.back = false;
  state.touch.left = false;
  state.touch.right = false;
  state.touch.turnLeft = false;
  state.touch.turnRight = false;
  stopMusic();
  messageEl.classList.remove("hidden");
  messageEl.querySelector("h2").textContent = "DoomLike";
  menuTextEl.textContent = "Press Enter or Start to begin. Use Enter or Escape to open the menu while playing.";
}

syncMusicButton();
requestAnimationFrame(frame);
