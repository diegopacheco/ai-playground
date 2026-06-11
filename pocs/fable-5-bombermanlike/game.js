const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const COLS = 15;
const ROWS = 13;
const TILE = 48;
const FLOOR = 0;
const WALL = 1;
const BRICK = 2;

const hud = {
  score: document.getElementById('score'),
  lives: document.getElementById('lives'),
  bombs: document.getElementById('bombs'),
  fire: document.getElementById('fire'),
  foes: document.getElementById('foes')
};

let state = 'title';
let grid = [];
let loot = {};
let bombs = [];
let blasts = [];
let sparks = [];
let enemies = [];
let score = 0;
let lives = 3;
let respawnTimer = 0;
let shake = 0;
let time = 0;

const keys = {};

const player = {
  x: 0, y: 0, size: 30, speed: 168,
  maxBombs: 1, range: 1, alive: true, face: 1, grace: 0
};

function cellKey(cx, cy) {
  return cx + ',' + cy;
}

function tileAt(cx, cy) {
  if (cx < 0 || cy < 0 || cx >= COLS || cy >= ROWS) return WALL;
  return grid[cy][cx];
}

function bombAt(cx, cy) {
  return bombs.find(b => b.cx === cx && b.cy === cy);
}

function solidFor(cx, cy, mover) {
  const t = tileAt(cx, cy);
  if (t !== FLOOR) return true;
  const b = bombAt(cx, cy);
  if (b && !(mover === player && b.passable)) return true;
  return false;
}

function buildLevel() {
  grid = [];
  loot = {};
  bombs = [];
  blasts = [];
  sparks = [];
  enemies = [];
  for (let y = 0; y < ROWS; y++) {
    const row = [];
    for (let x = 0; x < COLS; x++) {
      if (x === 0 || y === 0 || x === COLS - 1 || y === ROWS - 1) row.push(WALL);
      else if (x % 2 === 0 && y % 2 === 0) row.push(WALL);
      else if (Math.random() < 0.62) row.push(BRICK);
      else row.push(FLOOR);
    }
    grid.push(row);
  }
  const safe = [[1, 1], [2, 1], [1, 2], [1, 3], [3, 1]];
  for (const [x, y] of safe) grid[y][x] = FLOOR;
  const drops = ['bomb', 'bomb', 'fire', 'fire', 'speed'];
  const brickCells = [];
  for (let y = 1; y < ROWS - 1; y++)
    for (let x = 1; x < COLS - 1; x++)
      if (grid[y][x] === BRICK) brickCells.push([x, y]);
  for (const drop of drops) {
    if (!brickCells.length) break;
    const i = Math.floor(Math.random() * brickCells.length);
    const [x, y] = brickCells.splice(i, 1)[0];
    loot[cellKey(x, y)] = { type: drop, open: false };
  }
  const spots = [];
  for (let y = 1; y < ROWS - 1; y++)
    for (let x = 1; x < COLS - 1; x++)
      if (grid[y][x] === FLOOR && x + y > 9) spots.push([x, y]);
  for (let i = 0; i < 4 && spots.length; i++) {
    const j = Math.floor(Math.random() * spots.length);
    const [x, y] = spots.splice(j, 1)[0];
    enemies.push({
      x: x * TILE + TILE / 2, y: y * TILE + TILE / 2,
      size: 28, speed: 84 + i * 10, dir: Math.floor(Math.random() * 4),
      wob: Math.random() * Math.PI * 2
    });
  }
  placePlayer();
}

function placePlayer() {
  player.x = TILE * 1.5;
  player.y = TILE * 1.5;
  player.alive = true;
  player.grace = 2;
}

function startGame() {
  score = 0;
  lives = 3;
  player.maxBombs = 1;
  player.range = 1;
  player.speed = 168;
  buildLevel();
  state = 'play';
}

const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

function boxBlocked(x, y, half, mover) {
  const pts = [[x - half, y - half], [x + half, y - half], [x - half, y + half], [x + half, y + half]];
  return pts.some(([px, py]) => solidFor(Math.floor(px / TILE), Math.floor(py / TILE), mover));
}

function moveBox(ent, dx, dy, mover) {
  const half = ent.size / 2;
  if (dx !== 0 && !boxBlocked(ent.x + dx, ent.y, half, mover)) ent.x += dx;
  else if (dx !== 0 && mover === player) nudge(ent, 'y', dx, half);
  if (dy !== 0 && !boxBlocked(ent.x, ent.y + dy, half, mover)) ent.y += dy;
  else if (dy !== 0 && mover === player) nudge(ent, 'x', dy, half);
}

function nudge(ent, axis, amount, half) {
  const speed = Math.abs(amount);
  const center = ent[axis] / TILE;
  const target = (Math.floor(center) + 0.5) * TILE;
  const diff = target - ent[axis];
  if (Math.abs(diff) < 1) return;
  const step = Math.sign(diff) * Math.min(speed, Math.abs(diff));
  const nx = axis === 'x' ? ent.x + step : ent.x;
  const ny = axis === 'y' ? ent.y + step : ent.y;
  const fx = axis === 'x' ? nx : ent.x + Math.sign(amount) * 2;
  const fy = axis === 'y' ? ny : ent.y + Math.sign(amount) * 2;
  if (!boxBlocked(nx, ny, half, player) && !boxBlocked(fx, fy, half, player)) {
    ent.x = nx;
    ent.y = ny;
  }
}

function dropBomb() {
  if (!player.alive) return;
  const cx = Math.floor(player.x / TILE);
  const cy = Math.floor(player.y / TILE);
  if (bombAt(cx, cy)) return;
  if (bombs.length >= player.maxBombs) return;
  bombs.push({ cx, cy, t: 2.1, range: player.range, passable: true });
}

function detonate(bomb) {
  bombs.splice(bombs.indexOf(bomb), 1);
  shake = 0.3;
  const cells = [[bomb.cx, bomb.cy, 'core']];
  for (const [dx, dy] of DIRS) {
    for (let r = 1; r <= bomb.range; r++) {
      const cx = bomb.cx + dx * r;
      const cy = bomb.cy + dy * r;
      const t = tileAt(cx, cy);
      if (t === WALL) break;
      if (t === BRICK) {
        grid[cy][cx] = FLOOR;
        score += 10;
        const item = loot[cellKey(cx, cy)];
        if (item) item.open = true;
        cells.push([cx, cy, r === bomb.range ? 'tip' : 'arm']);
        break;
      }
      cells.push([cx, cy, r === bomb.range ? 'tip' : 'arm']);
      const other = bombAt(cx, cy);
      if (other) {
        detonate(other);
        break;
      }
    }
  }
  for (const [cx, cy, kind] of cells) {
    blasts.push({ cx, cy, kind, t: 0.45 });
    for (let i = 0; i < 5; i++) {
      sparks.push({
        x: cx * TILE + TILE / 2, y: cy * TILE + TILE / 2,
        vx: (Math.random() - 0.5) * 220, vy: (Math.random() - 0.5) * 220,
        t: 0.35 + Math.random() * 0.3
      });
    }
  }
}

function blastHits(x, y, half) {
  return blasts.some(b => {
    const bx = b.cx * TILE + TILE / 2;
    const by = b.cy * TILE + TILE / 2;
    return Math.abs(bx - x) < half + 18 && Math.abs(by - y) < half + 18;
  });
}

function killPlayer() {
  if (!player.alive || player.grace > 0) return;
  player.alive = false;
  lives--;
  shake = 0.5;
  respawnTimer = 1.4;
  if (lives <= 0) state = 'gameover';
}

function update(dt) {
  time += dt;
  if (shake > 0) shake -= dt;
  if (state !== 'play') return;

  if (player.alive) {
    if (player.grace > 0) player.grace -= dt;
    let dx = 0, dy = 0;
    if (keys['ArrowLeft'] || keys['a']) dx = -1;
    else if (keys['ArrowRight'] || keys['d']) dx = 1;
    if (keys['ArrowUp'] || keys['w']) dy = -1;
    else if (keys['ArrowDown'] || keys['s']) dy = 1;
    if (dx) player.face = dx;
    const mag = dx && dy ? Math.SQRT1_2 : 1;
    moveBox(player, dx * player.speed * dt * mag, 0, player);
    moveBox(player, 0, dy * player.speed * dt * mag, player);
  } else if (state === 'play') {
    respawnTimer -= dt;
    if (respawnTimer <= 0) placePlayer();
  }

  for (const b of bombs) {
    b.t -= dt;
    if (b.passable) {
      const half = player.size / 2;
      const overlapX = Math.abs(player.x - (b.cx * TILE + TILE / 2)) < TILE / 2 + half;
      const overlapY = Math.abs(player.y - (b.cy * TILE + TILE / 2)) < TILE / 2 + half;
      if (!(overlapX && overlapY)) b.passable = false;
    }
  }
  for (const b of [...bombs]) if (b.t <= 0 && bombs.includes(b)) detonate(b);

  for (const b of blasts) b.t -= dt;
  blasts = blasts.filter(b => b.t > 0);
  for (const s of sparks) {
    s.t -= dt;
    s.x += s.vx * dt;
    s.y += s.vy * dt;
  }
  sparks = sparks.filter(s => s.t > 0);

  for (const e of [...enemies]) {
    e.wob += dt * 6;
    const [dx, dy] = DIRS[e.dir];
    const before = e.x + e.y;
    moveBox(e, dx * e.speed * dt, dy * e.speed * dt, e);
    const cx = Math.floor(e.x / TILE);
    const cy = Math.floor(e.y / TILE);
    const centered = Math.abs(e.x - (cx * TILE + TILE / 2)) < 2 && Math.abs(e.y - (cy * TILE + TILE / 2)) < 2;
    const stuck = e.x + e.y === before;
    if (stuck || (centered && Math.random() < 0.04)) {
      const open = [];
      for (let d = 0; d < 4; d++) {
        const [ox, oy] = DIRS[d];
        if (!solidFor(cx + ox, cy + oy, e)) open.push(d);
      }
      if (open.length) e.dir = open[Math.floor(Math.random() * open.length)];
      if (stuck) {
        e.x = cx * TILE + TILE / 2;
        e.y = cy * TILE + TILE / 2;
      }
    }
    if (blastHits(e.x, e.y, e.size / 2)) {
      enemies.splice(enemies.indexOf(e), 1);
      score += 100;
      for (let i = 0; i < 10; i++) {
        sparks.push({
          x: e.x, y: e.y,
          vx: (Math.random() - 0.5) * 260, vy: (Math.random() - 0.5) * 260,
          t: 0.4 + Math.random() * 0.3
        });
      }
    } else if (player.alive && Math.abs(e.x - player.x) < 26 && Math.abs(e.y - player.y) < 26) {
      killPlayer();
    }
  }

  if (player.alive && blastHits(player.x, player.y, player.size / 2)) killPlayer();

  if (player.alive) {
    const ck = cellKey(Math.floor(player.x / TILE), Math.floor(player.y / TILE));
    const item = loot[ck];
    if (item && item.open) {
      if (item.type === 'bomb') player.maxBombs++;
      if (item.type === 'fire') player.range++;
      if (item.type === 'speed') player.speed += 26;
      score += 50;
      delete loot[ck];
    }
  }

  if (!enemies.length && state === 'play') state = 'win';

  hud.score.textContent = score;
  hud.lives.textContent = lives;
  hud.bombs.textContent = player.maxBombs;
  hud.fire.textContent = player.range;
  hud.foes.textContent = enemies.length;
}

function rr(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

function drawTile(x, y, t) {
  const px = x * TILE;
  const py = y * TILE;
  if (t === WALL) {
    ctx.fillStyle = '#3d4a37';
    ctx.fillRect(px, py, TILE, TILE);
    ctx.fillStyle = '#4d5c45';
    ctx.fillRect(px + 4, py + 4, TILE - 8, TILE - 12);
    ctx.fillStyle = '#2a3326';
    ctx.fillRect(px + 4, py + TILE - 10, TILE - 8, 6);
  } else if (t === BRICK) {
    ctx.fillStyle = '#a85b32';
    ctx.fillRect(px, py, TILE, TILE);
    ctx.fillStyle = '#c2703f';
    for (let r = 0; r < 3; r++) {
      const off = r % 2 ? 0 : 12;
      for (let c = -1; c < 3; c++) {
        ctx.fillRect(px + c * 22 + off + 2, py + r * 16 + 3, 18, 11);
      }
    }
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(px, py + TILE - 5, TILE, 5);
  } else {
    const even = (x + y) % 2 === 0;
    ctx.fillStyle = even ? '#15240f' : '#182a11';
    ctx.fillRect(px, py, TILE, TILE);
    ctx.fillStyle = 'rgba(157,255,60,0.05)';
    ctx.fillRect(px + 2, py + 2, 3, 3);
  }
}

function drawLoot(cx, cy, type) {
  const px = cx * TILE + TILE / 2;
  const py = cy * TILE + TILE / 2;
  const pulse = 5 + Math.sin(time * 5) * 2;
  ctx.fillStyle = '#0d1a08';
  rr(px - 16, py - 16, 32, 32, 6);
  ctx.fill();
  ctx.strokeStyle = '#9dff3c';
  ctx.lineWidth = 2;
  rr(px - 16 - pulse / 4, py - 16 - pulse / 4, 32 + pulse / 2, 32 + pulse / 2, 6);
  ctx.stroke();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = '18px monospace';
  if (type === 'bomb') {
    ctx.fillStyle = '#ffb938';
    ctx.beginPath();
    ctx.arc(px, py + 2, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffb938';
    ctx.beginPath();
    ctx.moveTo(px + 5, py - 6);
    ctx.lineTo(px + 11, py - 13);
    ctx.stroke();
  } else if (type === 'fire') {
    ctx.fillStyle = '#ff5d3c';
    ctx.beginPath();
    ctx.moveTo(px, py - 11);
    ctx.quadraticCurveTo(px + 11, py, px, py + 11);
    ctx.quadraticCurveTo(px - 11, py, px, py - 11);
    ctx.fill();
  } else {
    ctx.fillStyle = '#3cd9ff';
    ctx.beginPath();
    ctx.moveTo(px + 3, py - 11);
    ctx.lineTo(px - 7, py + 2);
    ctx.lineTo(px - 1, py + 2);
    ctx.lineTo(px - 3, py + 11);
    ctx.lineTo(px + 7, py - 2);
    ctx.lineTo(px + 1, py - 2);
    ctx.closePath();
    ctx.fill();
  }
}

function drawBomb(b) {
  const px = b.cx * TILE + TILE / 2;
  const py = b.cy * TILE + TILE / 2;
  const throb = 1 + Math.sin(time * (b.t < 0.7 ? 22 : 8)) * 0.08;
  ctx.fillStyle = b.t < 0.7 && Math.sin(time * 24) > 0 ? '#3a2f2c' : '#22201f';
  ctx.beginPath();
  ctx.arc(px, py + 3, 15 * throb, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.18)';
  ctx.beginPath();
  ctx.arc(px - 5, py - 2, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#caa46a';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(px + 4, py - 9);
  ctx.quadraticCurveTo(px + 10, py - 18, px + 16, py - 15);
  ctx.stroke();
  const fl = 3 + Math.sin(time * 18) * 2;
  ctx.fillStyle = '#ffb938';
  ctx.beginPath();
  ctx.arc(px + 17, py - 16, fl, 0, Math.PI * 2);
  ctx.fill();
}

function drawBlast(b) {
  const px = b.cx * TILE;
  const py = b.cy * TILE;
  const k = b.t / 0.45;
  ctx.fillStyle = `rgba(255,93,60,${0.85 * k})`;
  ctx.fillRect(px + 3, py + 3, TILE - 6, TILE - 6);
  ctx.fillStyle = `rgba(255,185,56,${0.9 * k})`;
  ctx.fillRect(px + 9, py + 9, TILE - 18, TILE - 18);
  ctx.fillStyle = `rgba(255,250,210,${0.95 * k})`;
  ctx.fillRect(px + 17, py + 17, TILE - 34, TILE - 34);
}

function drawPlayer() {
  if (!player.alive) return;
  if (player.grace > 0 && Math.sin(time * 20) > 0.2) return;
  const { x, y } = player;
  const bob = Math.sin(time * 10) * 1.5;
  ctx.fillStyle = '#f4f4ec';
  rr(x - 12, y - 8 + bob, 24, 20, 6);
  ctx.fill();
  ctx.fillStyle = '#e8e8df';
  ctx.beginPath();
  ctx.arc(x, y - 10 + bob, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ff5d3c';
  rr(x - 13, y - 17 + bob, 26, 7, 3);
  ctx.fill();
  ctx.fillStyle = '#0b0e0a';
  ctx.fillRect(x - 6 + player.face * 2, y - 11 + bob, 4, 5);
  ctx.fillRect(x + 2 + player.face * 2, y - 11 + bob, 4, 5);
  ctx.fillStyle = '#2c3526';
  rr(x - 11, y + 9 + bob, 9, 6, 2);
  ctx.fill();
  rr(x + 2, y + 9 + bob, 9, 6, 2);
  ctx.fill();
}

function drawEnemy(e) {
  const wob = Math.sin(e.wob) * 2;
  ctx.fillStyle = '#b03cff';
  ctx.beginPath();
  ctx.arc(e.x, e.y - 3 + wob, 14, Math.PI, 0);
  ctx.fill();
  ctx.fillRect(e.x - 14, e.y - 3 + wob, 28, 12);
  ctx.beginPath();
  for (let i = 0; i < 4; i++) {
    ctx.moveTo(e.x - 14 + i * 7, e.y + 9 + wob);
    ctx.lineTo(e.x - 14 + i * 7 + 3.5, e.y + 14 + wob - (i % 2) * 2);
    ctx.lineTo(e.x - 14 + i * 7 + 7, e.y + 9 + wob);
  }
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(e.x - 5, e.y - 4 + wob, 4.5, 0, Math.PI * 2);
  ctx.arc(e.x + 5, e.y - 4 + wob, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#0b0e0a';
  const [lx, ly] = DIRS[e.dir];
  ctx.beginPath();
  ctx.arc(e.x - 5 + lx * 2, e.y - 4 + wob + ly * 2, 2, 0, Math.PI * 2);
  ctx.arc(e.x + 5 + lx * 2, e.y - 4 + wob + ly * 2, 2, 0, Math.PI * 2);
  ctx.fill();
}

function drawOverlay(title, sub, color) {
  ctx.fillStyle = 'rgba(5,8,4,0.82)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.font = '38px "Press Start 2P"';
  ctx.shadowColor = color;
  ctx.shadowBlur = 24;
  ctx.fillText(title, canvas.width / 2, canvas.height / 2 - 40);
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#9dff3c';
  ctx.font = '13px "Press Start 2P"';
  if (Math.sin(time * 4) > -0.4) ctx.fillText(sub, canvas.width / 2, canvas.height / 2 + 36);
  if (state !== 'title') {
    ctx.fillStyle = '#ffb938';
    ctx.font = '15px "Press Start 2P"';
    ctx.fillText('SCORE ' + score, canvas.width / 2, canvas.height / 2 + 90);
  }
}

function render() {
  ctx.save();
  if (shake > 0) ctx.translate((Math.random() - 0.5) * 8, (Math.random() - 0.5) * 8);
  for (let y = 0; y < ROWS; y++)
    for (let x = 0; x < COLS; x++)
      drawTile(x, y, grid[y] ? grid[y][x] : WALL);
  for (const k in loot) {
    if (!loot[k].open) continue;
    const [cx, cy] = k.split(',').map(Number);
    drawLoot(cx, cy, loot[k].type);
  }
  for (const b of blasts) drawBlast(b);
  for (const b of bombs) drawBomb(b);
  for (const e of enemies) drawEnemy(e);
  drawPlayer();
  for (const s of sparks) {
    ctx.fillStyle = `rgba(255,185,56,${Math.min(1, s.t * 3)})`;
    ctx.fillRect(s.x - 2, s.y - 2, 4, 4);
  }
  ctx.restore();
  if (state === 'title') drawOverlay('BLASTGRID', 'PRESS ENTER TO START', '#ffb938');
  if (state === 'win') drawOverlay('YOU WIN', 'PRESS ENTER TO PLAY AGAIN', '#9dff3c');
  if (state === 'gameover') drawOverlay('GAME OVER', 'PRESS ENTER TO RETRY', '#ff5d3c');
}

let last = performance.now();
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  update(dt);
  render();
  requestAnimationFrame(loop);
}

window.addEventListener('keydown', e => {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) e.preventDefault();
  keys[e.key.length === 1 ? e.key.toLowerCase() : e.key] = true;
  if (e.key === ' ' && state === 'play') dropBomb();
  if (e.key === 'Enter' && state !== 'play') startGame();
});

window.addEventListener('keyup', e => {
  keys[e.key.length === 1 ? e.key.toLowerCase() : e.key] = false;
});

buildLevel();
requestAnimationFrame(loop);
