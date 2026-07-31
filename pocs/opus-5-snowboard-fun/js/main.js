import { buildScenery, COURSE_LENGTH, START_Z, height } from './terrain.js';
import { Rider } from './physics.js';
import { Renderer } from './render.js';
import { Input } from './input.js';
import { Sound } from './audio.js';

const STEP = 1 / 240;

const canvas = document.getElementById('view');
const renderer = new Renderer(canvas);
const scenery = buildScenery();
const rider = new Rider(scenery);
const input = new Input();
const sound = new Sound();

const el = {
  speed: document.getElementById('speed'),
  timer: document.getElementById('timer'),
  score: document.getElementById('score'),
  air: document.getElementById('air'),
  edge: document.getElementById('edgeFill'),
  progress: document.getElementById('progressFill'),
  toasts: document.getElementById('toasts'),
  menu: document.getElementById('menu'),
  results: document.getElementById('results'),
  stats: document.getElementById('stats'),
  paused: document.getElementById('paused'),
  grip: document.getElementById('grip')
};

const spray = [];
const flakes = [];
const trail = [];
let state = 'menu';
let last = 0;
let acc = 0;
let trailClock = 0;
let prevAir = false;
let prevCharge = 0;
let seenEvents = 0;

for (let i = 0; i < 260; i++) {
  flakes.push({
    x: (Math.random() - 0.5) * 70,
    y: Math.random() * 26,
    z: (Math.random() - 0.5) * 70,
    vx: 0,
    vy: 0,
    vz: 0,
    r: 0.035 + Math.random() * 0.05,
    c: '255,255,255',
    a: 0.55 + Math.random() * 0.4,
    life: 1,
    max: 1,
    ph: Math.random() * 6.28
  });
}

function spawn(x, y, z, vx, vy, vz, r, a, life, soft) {
  if (spray.length > 900) return;
  spray.push({ x, y, z, vx, vy, vz, r, c: '248,252,255', a, life, max: life, soft: soft !== false });
}

function emitSpray(dt) {
  if (rider.airborne || rider.crashTime > 0) return;
  const bite = Math.max(rider.skid, Math.abs(rider.edge) * 0.55);
  const power = bite * Math.min(1, rider.speed / 11);
  if (power < 0.05) return;
  const count = Math.min(16, Math.floor(power * rider.speed * dt * 42));
  const rx = Math.cos(rider.yaw);
  const rz = -Math.sin(rider.yaw);
  const side = -Math.sign(rider.edge || 1);
  for (let i = 0; i < count; i++) {
    const back = 0.1 + Math.random() * 1.1;
    const kick = 2 + power * 8;
    const grain = i % 3 === 0;
    const ox = rider.x - Math.sin(rider.yaw) * back + rx * side * 0.25;
    const oz = rider.z - Math.cos(rider.yaw) * back + rz * side * 0.25;
    const vx = rx * side * kick + (Math.random() - 0.5) * 2.4 + rider.vx * 0.16;
    const vz = rz * side * kick + (Math.random() - 0.5) * 2.4 + rider.vz * 0.16;
    if (grain) {
      spawn(
        ox,
        rider.y + 0.05,
        oz,
        vx * 1.4,
        2.2 + Math.random() * 4.5 * power,
        vz * 1.4,
        0.018 + Math.random() * 0.04,
        0.95,
        0.5 + Math.random() * 0.7,
        false
      );
    } else {
      spawn(
        ox,
        rider.y + 0.05 + Math.random() * 0.15,
        oz,
        vx,
        1.6 + Math.random() * 4.2 * power,
        vz,
        0.1 + Math.random() * 0.24,
        0.3 + power * 0.22,
        0.8 + Math.random() * 1.1
      );
    }
  }
}

function puff(strength) {
  const n = Math.min(60, 14 + Math.floor(strength * 4));
  for (let i = 0; i < n; i++) {
    const ang = Math.random() * Math.PI * 2;
    const sp = 1.4 + Math.random() * strength * 0.7;
    spawn(
      rider.x + Math.cos(ang) * 0.4,
      rider.y + 0.05,
      rider.z + Math.sin(ang) * 0.4,
      Math.cos(ang) * sp + rider.vx * 0.12,
      1.2 + Math.random() * 3.2,
      Math.sin(ang) * sp + rider.vz * 0.12,
      0.12 + Math.random() * 0.26,
      0.4,
      0.9 + Math.random() * 1.1
    );
    if (i % 3 === 0) {
      spawn(
        rider.x + Math.cos(ang) * 0.3,
        rider.y + 0.05,
        rider.z + Math.sin(ang) * 0.3,
        Math.cos(ang) * sp * 1.5,
        2 + Math.random() * 4,
        Math.sin(ang) * sp * 1.5,
        0.02 + Math.random() * 0.04,
        0.95,
        0.5 + Math.random() * 0.6,
        false
      );
    }
  }
}

function updateParticles(dt) {
  for (let i = spray.length - 1; i >= 0; i--) {
    const p = spray[i];
    p.life -= dt;
    if (p.life <= 0) {
      spray[i] = spray[spray.length - 1];
      spray.pop();
      continue;
    }
    p.vy -= 4.6 * dt;
    p.vx -= p.vx * 2.4 * dt;
    p.vy -= p.vy * 1.1 * dt;
    p.vz -= p.vz * 2.4 * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;
    p.r += dt * 0.85;
  }
  const cx = renderer.camX;
  const cy = renderer.camY;
  const cz = renderer.camZ;
  for (const f of flakes) {
    f.ph += dt;
    f.y -= (1.4 + Math.sin(f.ph) * 0.3) * dt;
    f.x += Math.sin(f.ph * 0.7) * 0.5 * dt;
    if (f.y < cy - 8) f.y = cy + 24;
    if (f.y > cy + 26) f.y = cy - 6;
    if (f.x < cx - 35) f.x += 70;
    if (f.x > cx + 35) f.x -= 70;
    if (f.z < cz - 35) f.z += 70;
    if (f.z > cz + 35) f.z -= 70;
  }
}

function pushTrail(dt) {
  trailClock += dt;
  if (trailClock < 0.04 || rider.airborne || rider.crashTime > 0) return;
  trailClock = 0;
  trail.push({
    x: rider.x,
    y: rider.y,
    z: rider.z,
    rx: Math.cos(rider.yaw),
    rz: -Math.sin(rider.yaw),
    skid: rider.skid
  });
  if (trail.length > 300) trail.shift();
}

function toast(text, kind) {
  const div = document.createElement('div');
  div.className = 'toast ' + kind;
  div.textContent = text;
  el.toasts.appendChild(div);
  setTimeout(() => div.remove(), 1700);
}

function drainEvents() {
  while (seenEvents < rider.events.length) {
    const e = rider.events[seenEvents++];
    toast(e.text, e.kind);
    if (e.kind === 'crash') {
      sound.burst('crash');
      renderer.shake = 1;
    }
  }
}

function fmt(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s < 10 ? '0' : ''}${s.toFixed(2)}`;
}

function updateHud() {
  el.speed.textContent = (rider.speed * 3.6).toFixed(0);
  el.timer.textContent = fmt(rider.time);
  el.score.textContent = rider.score.toFixed(0);
  el.air.textContent = rider.airborne ? rider.airTime.toFixed(1) + 's' : '--';
  el.air.className = rider.airborne ? 'value live' : 'value';
  const e = Math.abs(rider.edge);
  el.edge.style.width = (e * 100).toFixed(0) + '%';
  el.edge.style.background = rider.skid > 0.35 ? '#e8654a' : '#4fc3f7';
  el.grip.textContent = rider.airborne ? 'AIR' : rider.skid > 0.35 ? 'SKID' : 'CARVE';
  el.progress.style.width = (Math.max(0, (rider.z - START_Z) / (COURSE_LENGTH - START_Z)) * 100).toFixed(1) + '%';
}

function finish() {
  state = 'done';
  sound.quiet();
  el.stats.innerHTML = `
    <div><span>TIME</span><b>${fmt(rider.finishTime)}</b></div>
    <div><span>SCORE</span><b>${rider.score.toFixed(0)}</b></div>
    <div><span>TOP SPEED</span><b>${(rider.topSpeed * 3.6).toFixed(0)} km/h</b></div>
    <div><span>AIR TIME</span><b>${rider.totalAir.toFixed(1)} s</b></div>
    <div><span>TRICKS</span><b>${rider.tricks}</b></div>
    <div><span>WIPEOUTS</span><b>${rider.crashes}</b></div>`;
  el.results.classList.remove('hidden');
}

function restart() {
  rider.reset();
  spray.length = 0;
  trail.length = 0;
  seenEvents = 0;
  prevAir = false;
  prevCharge = 0;
  renderer.ready = false;
  renderer.camX = 0;
  renderer.camZ = START_Z - 8;
  renderer.camY = height(0, START_Z) + 3;
  el.toasts.innerHTML = '';
  el.results.classList.add('hidden');
  el.menu.classList.add('hidden');
  el.paused.classList.add('hidden');
  state = 'play';
  sound.start();
}

input.onKey = (code) => {
  if (code === 'Enter' || code === 'Space') {
    if (state === 'menu' || state === 'done') restart();
    return;
  }
  if (code === 'KeyR') {
    if (state !== 'menu') restart();
    return;
  }
  if (code === 'KeyP' || code === 'Escape') {
    if (state === 'play') {
      state = 'pause';
      el.paused.classList.remove('hidden');
      sound.quiet();
    } else if (state === 'pause') {
      state = 'play';
      el.paused.classList.add('hidden');
      sound.start();
    }
  }
};

window.addEventListener('resize', () => renderer.resize());
document.addEventListener('visibilitychange', () => {
  if (document.hidden) sound.quiet();
  else if (state === 'play') sound.start();
});
window.addEventListener('pagehide', () => sound.quiet());

function frame(ts) {
  requestAnimationFrame(frame);
  const now = ts / 1000;
  let dt = last ? Math.min(0.05, now - last) : 0;
  last = now;

  if (state === 'play') {
    const cmd = input.read();
    acc += dt;
    let guard = 0;
    while (acc >= STEP && guard++ < 24) {
      rider.step(STEP, cmd);
      acc -= STEP;
    }
    if (rider.airborne !== prevAir) {
      if (!rider.airborne) {
        if (rider.impact > 2.5) {
          puff(rider.impact);
          sound.burst('land');
          renderer.shake = Math.min(1, rider.impact / 14);
        }
      } else {
        sound.burst('pop');
      }
      prevAir = rider.airborne;
    }
    if (prevCharge > 0 && rider.charge === 0 && !rider.airborne) puff(3);
    prevCharge = rider.charge;
    emitSpray(dt);
    pushTrail(dt);
    drainEvents();
    updateHud();
    sound.update(rider);
    if (rider.finished) finish();
  } else {
    dt = Math.min(dt, 0.033);
  }

  updateParticles(state === 'play' ? dt : dt * 0.4);
  renderer.render(rider, scenery, spray, flakes, trail, dt);
}

renderer.camZ = START_Z - 8;
renderer.camY = height(0, START_Z) + 3;
requestAnimationFrame(frame);
