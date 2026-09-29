import * as THREE from 'three';
import { CARS } from './core/vehicle.js';
import { LEVELS, createGovernor, tickGovernor } from './core/quality.js';
import { createShowroom } from './render/showroom.js';
import { createMenu, createHud, showResults, CPU_NAMES } from './ui.js';
import { createInput } from './input.js';
import { createAudio } from './audio.js';
import { createRaceScene, raceConfig } from './game.js';

const $ = (id) => document.getElementById(id);
const renderer = new THREE.WebGLRenderer({ canvas: $('view'), antialias: true, powerPreference: 'high-performance' });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const governor = createGovernor(1);
const quality = { ...LEVELS[governor.level] };
const showroom = createShowroom(renderer);
const hud = createHud();
let audio = null;
let race = null;
let config = null;
let mode = 'menu';
let finishedAt = 0;
let lastCount = 4;

function applyQuality() {
  Object.assign(quality, LEVELS[governor.level]);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2) * quality.scale);
  renderer.setSize(innerWidth, innerHeight, false);
  race?.setShadows(quality.shadows);
}

applyQuality();
addEventListener('resize', applyQuality);

const menu = createMenu({
  onChange: (c) => showroom.show(CARS[c.car], c.color, c.finish),
  onStart: (choice) => startRace(raceConfig(choice)),
});

const input = createInput((code) => {
  if (mode !== 'race' && mode !== 'paused') return;
  if (code === 'KeyP' || code === 'Escape') togglePause();
  if (mode !== 'race') return;
  if (code === 'KeyC') race.cycleCamera();
  if (code === 'KeyR') race.reset();
  if (code === 'KeyM') audio?.toggleMute();
  if (code === 'KeyH') hud.toggleControls();
});

function disposeScene(scene) {
  scene.traverse((o) => {
    o.geometry?.dispose();
    const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of mats) {
      for (const v of Object.values(m)) if (v && v.isTexture) v.dispose();
      m.dispose();
    }
  });
  scene.environment?.dispose();
}

function endRace() {
  audio?.stopRace();
  if (race) disposeScene(race.scene);
  race = null;
}

function setLoading(text, pct) {
  $('loading-text').textContent = text;
  $('loading-bar').style.width = `${pct}%`;
}

function startRace(cfg) {
  config = cfg;
  audio = audio || createAudio();
  audio.ctx.resume();
  endRace();
  menu.close();
  $('results').classList.add('hidden');
  $('pause').classList.add('hidden');
  $('loading').classList.remove('hidden');
  setLoading(`Building ${cfg.def.name}, ${cfg.def.city}`, 25);
  mode = 'loading';
  setTimeout(() => {
    setLoading('Planting forests, digging mud ruts, filling puddles', 70);
    setTimeout(() => {
      race = createRaceScene(renderer, cfg, quality);
      renderer.compile(race.scene, race.camera);
      audio.setupRace(race.sim.cars, race.sim.playerIndex, cfg.weather);
      renderer.toneMappingExposure = cfg.weather === 'clear' ? 0.95 : 1.15;
      $('loading').classList.add('hidden');
      input.clear();
      hud.show();
      mode = 'race';
      finishedAt = 0;
      lastCount = 4;
    }, 30);
  }, 30);
}

function togglePause() {
  if (mode === 'race') {
    mode = 'paused';
    $('pause').classList.remove('hidden');
    audio?.ctx.suspend();
  } else if (mode === 'paused') {
    mode = 'race';
    $('pause').classList.add('hidden');
    audio?.ctx.resume();
  }
}

function toMenu() {
  endRace();
  hud.hide();
  $('results').classList.add('hidden');
  $('pause').classList.add('hidden');
  renderer.toneMappingExposure = 0.95;
  mode = 'menu';
  menu.open();
}

$('resume').onclick = togglePause;
$('restart').onclick = () => startRace(config);
$('quit').onclick = toMenu;
$('again').onclick = () => startRace(config);
$('menu-btn').onclick = toMenu;

const names = [...CPU_NAMES, 'YOU'];

function raceTick(dt, aspect) {
  const sim = race.sim;
  const controls = input.read(dt, Math.abs(sim.cars[sim.playerIndex].u));
  race.frame(dt, controls, aspect, audio, hud);
  if (sim.phase === 'countdown') {
    const n = Math.ceil(sim.countdown);
    if (n !== lastCount && n > 0) {
      lastCount = n;
      hud.message(String(n), '', 0.9);
      audio?.beep(false);
    }
  } else if (lastCount !== 0) {
    lastCount = 0;
    hud.message('GO!', 'go', 1);
    audio?.beep(true);
  }
  if (sim.phase === 'finished' && !finishedAt) {
    finishedAt = sim.race.time;
    hud.message('FINISH', 'info', 3);
  }
  if (finishedAt && sim.race.time - finishedAt > 3 && $('results').classList.contains('hidden')) showResults(sim, names, config);
  hud.update(dt, sim, names);
  audio?.update(sim.cars, sim.playerIndex, race.camera);
  renderer.render(race.scene, race.camera);
}

let last = performance.now();
function loop(now = performance.now()) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  const aspect = innerWidth / innerHeight;
  tickGovernor(governor, dt);
  if (governor.changed) applyQuality();
  if (mode === 'menu') {
    showroom.update(dt, aspect);
    renderer.render(showroom.scene, showroom.camera);
  } else if (mode === 'race' && race) {
    raceTick(dt, aspect);
  } else if (mode === 'paused' && race) {
    renderer.render(race.scene, race.camera);
  }
  if (race) hud.setPerf(governor.fps, LEVELS[governor.level].name, race.cameraName());
  requestAnimationFrame(loop);
}

menu.open();
loop();
