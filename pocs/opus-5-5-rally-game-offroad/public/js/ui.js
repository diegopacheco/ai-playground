import { TRACKS, buildTrack } from './core/tracks.js';
import { CARS } from './core/vehicle.js';
import { COLORS, FINISHES } from './render/cars.js';
import { formatTime, standings, progressOf } from './core/race.js';
import { nextNote } from './core/pacenotes.js';

const $ = (id) => document.getElementById(id);
const STEPS = ['Track', 'Vehicle', 'Paint', 'Conditions'];
const WEATHERS = [
  { id: 'clear', name: 'Clear', icon: 'SUN', text: 'Dry mud and dust clouds. Maximum grip, long shadows.' },
  { id: 'rain', name: 'Rain', icon: 'RAIN', text: 'Soaked ruts and deep puddles. Wet mud grips 16% less.' },
  { id: 'snow', name: 'Snow', icon: 'SNOW', text: 'Slush over frozen mud. Grip drops by a third, brake early.' },
];
export const CPU_NAMES = ['M. Rivera', 'K. Tanaka', 'J. Walker'];

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function trackThumb(def) {
  const c = el('canvas');
  c.width = 192;
  c.height = 144;
  const ctx = c.getContext('2d');
  const t = buildTrack(def);
  const g = ctx.createLinearGradient(0, 0, 0, 144);
  g.addColorStop(0, def.palette.grass);
  g.addColorStop(1, def.palette.dirt);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 192, 144);
  ctx.strokeStyle = '#1a120c';
  ctx.lineWidth = 9;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i <= t.count; i += 4) {
    const k = i % t.count;
    const x = 96 + (t.xs[k] - def.frame.cx) * 0.14;
    const y = 72 + (t.zs[k] - def.frame.cz) * 0.14;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.strokeStyle = '#ffb31a';
  ctx.lineWidth = 3;
  ctx.stroke();
  return c;
}

export function createMenu({ onChange, onStart }) {
  const choice = { track: 0, car: 0, color: COLORS[0].hex, finish: 'Gloss', weather: 'clear' };
  let step = 0;
  const thumbs = TRACKS.map(trackThumb);

  function renderSteps() {
    const nav = $('steps');
    nav.innerHTML = '';
    STEPS.forEach((name, k) => {
      const b = el('button', k === step ? 'active' : k < step ? 'done' : '', `${k + 1}. ${name.toUpperCase()}`);
      b.onclick = () => go(k);
      nav.append(b);
    });
    $('back').style.visibility = step === 0 ? 'hidden' : 'visible';
    $('next').textContent = step === STEPS.length - 1 ? 'START RACE' : 'Next';
    const car = CARS[choice.car];
    const colorName = COLORS.find((c) => c.hex === choice.color)?.name || choice.color;
    $('summary').innerHTML = `${TRACKS[choice.track].city} &middot; ${car.name}<br>${colorName} ${choice.finish} &middot; ${choice.weather} &middot; 3 laps vs 3 CPU`;
  }

  function trackPanel(panel) {
    panel.append(el('h3', '', 'CHOOSE YOUR STAGE'));
    TRACKS.forEach((def, k) => {
      const card = el('div', `card${k === choice.track ? ' selected' : ''}`);
      card.append(thumbs[k]);
      card.append(el('div', '', `<div class="city">${def.city}</div><div class="name">${def.name}</div><div class="blurb">${def.blurb}</div>`));
      card.onclick = () => set({ track: k });
      panel.append(card);
    });
  }

  function bar(label, v) {
    return `<span>${label}</span><div class="stat-bar"><i style="width:${Math.round(v * 100)}%"></i></div>`;
  }

  function carPanel(panel) {
    panel.append(el('h3', '', 'CHOOSE YOUR 4X4'));
    const maxT = Math.max(...CARS.map((c) => c.torque));
    const maxV = Math.max(...CARS.map((c) => c.topSpeed));
    CARS.forEach((spec, k) => {
      const card = el('div', `card car-card${k === choice.car ? ' selected' : ''}`);
      const ratio = spec.torque / spec.mass;
      card.append(el('div', '', `<div class="name">${spec.name}</div><div class="blurb">${spec.accent}</div><div class="stats">${bar('Torque', spec.torque / maxT)}${bar('Accel', Math.min(1, ratio / 0.42))}${bar('Grip', (spec.grip - 0.9) / 0.2)}${bar('Top speed', spec.topSpeed / maxV)}</div>`));
      card.append(el('div', 'car-num', String(k + 1).padStart(2, '0')));
      card.onclick = () => set({ car: k });
      panel.append(card);
    });
  }

  function paintPanel(panel) {
    panel.append(el('h3', '', 'PAINT COLOR'));
    const sw = el('div', 'swatches');
    COLORS.forEach((c) => {
      const s = el('div', `swatch${c.hex === choice.color ? ' selected' : ''}`);
      s.style.background = c.hex;
      s.title = c.name;
      s.onclick = () => set({ color: c.hex });
      sw.append(s);
    });
    panel.append(sw);
    const custom = el('label', 'custom-color', 'Custom color');
    const input = el('input');
    input.type = 'color';
    input.value = choice.color;
    input.oninput = () => set({ color: input.value }, false);
    custom.prepend(input);
    panel.append(custom);
    panel.append(el('h3', '', 'PAINT FINISH'));
    const chips = el('div', 'chips');
    FINISHES.forEach((f) => {
      const c = el('button', `chip${f === choice.finish ? ' selected' : ''}`, f.toUpperCase());
      c.onclick = () => set({ finish: f });
      chips.append(c);
    });
    panel.append(chips);
  }

  function weatherPanel(panel) {
    panel.append(el('h3', '', 'WEATHER'));
    WEATHERS.forEach((w) => {
      const card = el('div', `card weather-card${w.id === choice.weather ? ' selected' : ''}`);
      card.append(el('div', 'weather-icon', w.icon));
      card.append(el('div', '', `<div class="name">${w.name}</div><div class="blurb">${w.text}</div>`));
      card.onclick = () => set({ weather: w.id });
      panel.append(card);
    });
    panel.append(el('div', 'blurb', 'The track is always mud. Every stage is 3 laps against 3 CPU drivers.'));
  }

  function renderPanel() {
    const panel = $('panel');
    panel.innerHTML = '';
    [trackPanel, carPanel, paintPanel, weatherPanel][step](panel);
  }

  function set(patch, rerender = true) {
    Object.assign(choice, patch);
    onChange(choice);
    if (rerender) renderPanel();
    renderSteps();
  }

  function go(k) {
    step = Math.max(0, Math.min(STEPS.length - 1, k));
    renderPanel();
    renderSteps();
  }

  $('back').onclick = () => go(step - 1);
  $('next').onclick = () => (step === STEPS.length - 1 ? onStart({ ...choice }) : go(step + 1));
  addEventListener('keydown', (e) => {
    if ($('menu').classList.contains('hidden')) return;
    if (e.code === 'Enter') $('next').click();
  });

  return {
    choice,
    open() {
      $('menu').classList.remove('hidden');
      go(step);
      onChange(choice);
    },
    close() {
      $('menu').classList.add('hidden');
    },
  };
}

function drawGauge(ctx, car) {
  const W = 280;
  const cx = W / 2;
  const cy = W / 2;
  const r = 118;
  ctx.clearRect(0, 0, W, W);
  const start = Math.PI * 0.75;
  const sweep = Math.PI * 1.5;
  ctx.fillStyle = 'rgba(12,12,14,0.62)';
  ctx.beginPath();
  ctx.arc(cx, cy, r + 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 10;
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.beginPath();
  ctx.arc(cx, cy, r, start, start + sweep);
  ctx.stroke();
  const rpmT = Math.min(1, car.rpm / 7000);
  const grad = ctx.createLinearGradient(0, W, W, 0);
  grad.addColorStop(0, '#ffb31a');
  grad.addColorStop(1, '#e2581d');
  ctx.strokeStyle = car.rpm > 5900 ? '#ff3b2f' : grad;
  ctx.beginPath();
  ctx.arc(cx, cy, r, start, start + sweep * rpmT);
  ctx.stroke();
  ctx.fillStyle = '#b9b3a6';
  ctx.font = '600 15px "Barlow Condensed", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let k = 0; k <= 7; k++) {
    const a = start + (sweep * k) / 7;
    ctx.fillStyle = k >= 6 ? '#ff5a3d' : '#b9b3a6';
    ctx.fillText(String(k), cx + Math.cos(a) * (r - 22), cy + Math.sin(a) * (r - 22));
  }
  const a = start + sweep * rpmT;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx + Math.cos(a) * 30, cy + Math.sin(a) * 30);
  ctx.lineTo(cx + Math.cos(a) * (r - 6), cy + Math.sin(a) * (r - 6));
  ctx.stroke();
  const mph = Math.round(Math.abs(car.u) * 2.23694);
  ctx.fillStyle = '#f4f1ea';
  ctx.font = '800 64px "Barlow Condensed", sans-serif';
  ctx.fillText(String(mph), cx, cy + 4);
  ctx.font = '600 16px "Barlow Condensed", sans-serif';
  ctx.fillStyle = '#b9b3a6';
  ctx.fillText(`MPH  ${Math.round(Math.abs(car.u) * 3.6)} KM/H`, cx, cy + 44);
  ctx.font = '800 30px "Barlow Condensed", sans-serif';
  ctx.fillStyle = '#ffb31a';
  ctx.fillText(car.gear === -1 ? 'R' : `D${car.gear}`, cx, cy + 80);
  ctx.font = '600 12px "Barlow Condensed", sans-serif';
  ctx.fillStyle = '#b9b3a6';
  ctx.fillText('x1000 RPM', cx, cy - 44);
}

function drawMinimap(ctx, sim) {
  const { track } = sim.world;
  const W = 220;
  ctx.clearRect(0, 0, W, W);
  if (!track.bounds) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < track.count; i++) {
      minX = Math.min(minX, track.xs[i]);
      maxX = Math.max(maxX, track.xs[i]);
      minZ = Math.min(minZ, track.zs[i]);
      maxZ = Math.max(maxZ, track.zs[i]);
    }
    track.bounds = { cx: (minX + maxX) / 2, cz: (minZ + maxZ) / 2, span: Math.max(maxX - minX, maxZ - minZ) };
  }
  const { cx, cz, span } = track.bounds;
  const scale = (W * 0.8) / span;
  const tx = (x) => W / 2 + (x - cx) * scale;
  const tz = (z) => W / 2 + (z - cz) * scale;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i <= track.count; i += 3) {
    const k = i % track.count;
    if (i === 0) ctx.moveTo(tx(track.xs[k]), tz(track.zs[k]));
    else ctx.lineTo(tx(track.xs[k]), tz(track.zs[k]));
  }
  ctx.closePath();
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = 8;
  ctx.stroke();
  ctx.strokeStyle = '#d8cbb3';
  ctx.lineWidth = 3.5;
  ctx.stroke();
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(tx(track.xs[0]) - 3, tz(track.zs[0]) - 3, 6, 6);
  sim.cars.forEach((car, k) => {
    const me = k === sim.playerIndex;
    ctx.save();
    ctx.translate(tx(car.x), tz(car.z));
    ctx.rotate(-car.heading + Math.PI);
    ctx.fillStyle = me ? '#ffb31a' : '#ff4b3a';
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const s = me ? 8 : 6;
    ctx.moveTo(0, -s);
    ctx.lineTo(s * 0.7, s * 0.7);
    ctx.lineTo(-s * 0.7, s * 0.7);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  });
}

const WORDS = ['', 'one', 'two', 'three', 'four', 'five', 'six'];

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.25;
  u.pitch = 0.9;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function updatePaceNote(sim, called, voice) {
  const car = sim.cars[sim.playerIndex];
  const next = nextNote(sim.world.track, sim.notes, car.proj.i);
  const el = $('hud-note');
  if (!next || next.distance > 320) {
    el.classList.add('hidden');
    return;
  }
  const { note, distance } = next;
  const u = Math.max(car.u, 0);
  const brake = u > note.speed + 1.5 && distance < (u * u - note.speed * note.speed) / 12 + 40;
  el.classList.remove('hidden');
  el.classList.toggle('brake', brake);
  $('note-arrow').textContent = note.dir === 'LEFT' ? '\u25C0' : '\u25B6';
  $('note-text').textContent = `${note.dir} ${note.severity}`;
  $('note-dist').textContent = brake ? `BRAKE  ${Math.round(distance)} m` : `${Math.round(distance)} m  ${Math.round(note.speed * 2.23694)} mph`;
  const key = `${sim.race.entries[sim.playerIndex].lap}:${note.i}`;
  if (voice && !called.has(key) && distance < Math.max(70, u * 3.5)) {
    called.add(key);
    speak(`${note.dir.toLowerCase()} ${WORDS[note.severity]}${brake ? ', brake' : ''}`);
  }
}

export function createHud() {
  const gauge = $('gauge').getContext('2d');
  const minimap = $('minimap').getContext('2d');
  const called = new Set();
  let voice = true;
  let messageUntil = 0;
  let clock = 0;
  return {
    setVoice(on) {
      voice = on;
      if (!on && 'speechSynthesis' in window) speechSynthesis.cancel();
    },
    reset() {
      called.clear();
    },
    show() {
      $('hud').classList.remove('hidden');
    },
    hide() {
      $('hud').classList.add('hidden');
    },
    toggleControls() {
      $('controls').classList.toggle('hidden');
    },
    message(text, cls = '', seconds = 1.2) {
      const m = $('hud-message');
      m.textContent = text;
      m.className = `hud-center ${cls}`;
      messageUntil = clock + seconds;
    },
    setPerf(fps, quality, cam) {
      $('hud-fps').textContent = Math.round(fps);
      $('hud-quality').textContent = quality;
      $('hud-cam').textContent = cam;
    },
    update(dt, sim, names) {
      clock += dt;
      if (clock > messageUntil) $('hud-message').textContent = '';
      const race = sim.race;
      const me = race.entries[sim.playerIndex];
      const order = standings(race);
      $('hud-pos').textContent = order.indexOf(me) + 1;
      $('hud-lap').textContent = Math.min(race.laps, me.lap + 1);
      $('hud-time').textContent = formatTime(race.time);
      $('hud-last').textContent = me.lapTimes.length ? formatTime(me.lapTimes.at(-1)) : '-';
      $('hud-best').textContent = me.lapTimes.length ? formatTime(Math.min(...me.lapTimes)) : '-';
      const lead = progressOf(race, order[0]);
      $('hud-board').innerHTML = order.map((e, k) => {
        const gap = k === 0 ? 'LEADER' : `+${Math.round((lead - progressOf(race, e)) * sim.world.track.spacing)} m`;
        return `<li class="${e.id === sim.playerIndex ? 'me' : ''}"><span>${k + 1}. ${names[e.id]}</span><span class="gap">${e.finished ? formatTime(e.finishTime) : gap}</span></li>`;
      }).join('');
      if (sim.phase !== 'countdown') updatePaceNote(sim, called, voice);
      drawGauge(gauge, sim.cars[sim.playerIndex]);
      drawMinimap(minimap, sim);
    },
  };
}

export function showResults(sim, names, config) {
  const order = standings(sim.race);
  const me = order.findIndex((e) => e.id === sim.playerIndex) + 1;
  $('results-title').textContent = me === 1 ? 'VICTORY' : `FINISHED P${me}`;
  const rows = order.map((e, k) => {
    const best = e.lapTimes.length ? formatTime(Math.min(...e.lapTimes)) : '-';
    const time = e.finished ? formatTime(e.finishTime) : `${e.lap}/${sim.race.laps} laps`;
    return `<tr class="${e.id === sim.playerIndex ? 'me' : ''}"><td>${k + 1}</td><td>${names[e.id]}</td><td>${config.cars[e.id].name}</td><td>${time}</td><td>${best}</td></tr>`;
  });
  $('results-table').innerHTML = `<tr><th>POS</th><th>DRIVER</th><th>VEHICLE</th><th>TIME</th><th>BEST LAP</th></tr>${rows.join('')}`;
  $('results').classList.remove('hidden');
}
