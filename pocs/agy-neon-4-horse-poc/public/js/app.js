import * as THREE from 'three';
import { CAMERAS, COATS, GAITS, LIGHTING_MOODS } from './presets.js';
import { evaluateGaitPose } from './gait.js';
import { createHorseRig } from './horse-builder.js';
import { createColoradoWorld } from './colorado-world.js';
import { getTerrainHeight, getTerrainNormal, getTrailCenterX, getTrailSlopeX } from './terrain-math.js';
import { createSoundscape } from './audio.js';

const canvas = document.getElementById('scene-canvas');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance'
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2('#b5d2ea', 0.0024);

const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 680);

const sunLight = new THREE.DirectionalLight('#fff2d6', 3.1);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.near = 2;
sunLight.shadow.camera.far = 95;
sunLight.shadow.camera.left = -22;
sunLight.shadow.camera.right = 22;
sunLight.shadow.camera.top = 22;
sunLight.shadow.camera.bottom = -22;
sunLight.shadow.bias = -0.00045;
scene.add(sunLight);
scene.add(sunLight.target);

const hemiLight = new THREE.HemisphereLight('#7ab6ef', '#4c6633', 1.05);
scene.add(hemiLight);

const rimLight = new THREE.DirectionalLight('#ffe6c2', 0.55);
rimLight.position.set(-14, 10, -18);
scene.add(rimLight);

const world = createColoradoWorld(scene);
const horse = createHorseRig('bay');
scene.add(horse.root);

const soundscape = createSoundscape();

const state = {
  gaitId: 'gallop',
  coatId: 'bay',
  lightingId: 'morning',
  cameraId: 'hero',
  speedMultiplier: 1.0,
  paused: false,
  cycle: 0.14,
  worldZ: 18,
  timeSec: 0,
  orbitYaw: 0.72,
  orbitPitch: 0.24,
  orbitDist: 5.2,
  dragging: false,
  lastPointerX: 0,
  lastPointerY: 0,
  prevImpact: [false, false, false, false]
};

const sunDirection = new THREE.Vector3();
const targetCamPos = new THREE.Vector3();
const targetLookAt = new THREE.Vector3();
const smoothCamPos = new THREE.Vector3(3.6, 1.8, 4.2);
const smoothLookAt = new THREE.Vector3(0, 1.3, 0);

function applyLighting(lightingId) {
  const mood = LIGHTING_MOODS[lightingId] || LIGHTING_MOODS.morning;
  state.lightingId = mood.id;

  const elev = THREE.MathUtils.degToRad(mood.sunElevation);
  const azim = THREE.MathUtils.degToRad(mood.sunAzimuth);
  sunDirection.set(
    Math.cos(elev) * Math.sin(azim),
    Math.sin(elev),
    Math.cos(elev) * Math.cos(azim)
  ).normalize();

  sunLight.color.set(mood.sunColor);
  sunLight.intensity = mood.sunIntensity;
  hemiLight.color.set(mood.hemiSky);
  hemiLight.groundColor.set(mood.hemiGround);
  hemiLight.intensity = mood.hemiIntensity;
  scene.fog.color.set(mood.fogColor);
  scene.fog.density = mood.fogDensity;
  renderer.toneMappingExposure = mood.exposure;

  world.applyLightingMood(mood, sunDirection);
  syncButtons();
}

function setGait(gaitId) {
  if (!GAITS[gaitId]) return;
  state.gaitId = gaitId;
  syncButtons();
}

function setCoat(coatId) {
  if (!COATS[coatId]) return;
  state.coatId = coatId;
  horse.applyCoat(coatId);
  syncButtons();
}

function setCamera(cameraId, snap = false) {
  if (!CAMERAS[cameraId]) return;
  state.cameraId = cameraId;
  const cfg = CAMERAS[cameraId];
  camera.fov = cfg.fov;
  camera.updateProjectionMatrix();
  if (snap) {
    const baseY = horse.root.position.y;
    smoothCamPos.set(cfg.offset[0], baseY + cfg.offset[1], cfg.offset[2]);
    smoothLookAt.set(cfg.lookAt[0], baseY + cfg.lookAt[1], cfg.lookAt[2]);
    camera.position.copy(smoothCamPos);
    camera.lookAt(smoothLookAt);
  }
  syncButtons();
}

function syncButtons() {
  document.querySelectorAll('[data-gait]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.gait === state.gaitId);
  });
  document.querySelectorAll('[data-coat]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.coat === state.coatId);
  });
  document.querySelectorAll('[data-lighting]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.lighting === state.lightingId);
  });
  document.querySelectorAll('[data-camera]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.camera === state.cameraId);
  });
  const pauseBtn = document.getElementById('btn-pause');
  if (pauseBtn) {
    pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
  }
  const soundBtn = document.getElementById('btn-sound');
  if (soundBtn) {
    soundBtn.textContent = soundscape.isEnabled() ? 'Sound: On' : 'Sound: Off';
    soundBtn.classList.toggle('active', soundscape.isEnabled());
  }
}

function bindUI() {
  document.querySelectorAll('[data-gait]').forEach((btn) => {
    btn.addEventListener('click', () => setGait(btn.dataset.gait));
  });
  document.querySelectorAll('[data-coat]').forEach((btn) => {
    btn.addEventListener('click', () => setCoat(btn.dataset.coat));
  });
  document.querySelectorAll('[data-lighting]').forEach((btn) => {
    btn.addEventListener('click', () => applyLighting(btn.dataset.lighting));
  });
  document.querySelectorAll('[data-camera]').forEach((btn) => {
    btn.addEventListener('click', () => setCamera(btn.dataset.camera));
  });

  const speedRange = document.getElementById('speed-range');
  const speedReadout = document.getElementById('speed-mult');
  if (speedRange) {
    speedRange.addEventListener('input', () => {
      state.speedMultiplier = Number(speedRange.value);
      if (speedReadout) speedReadout.textContent = `${state.speedMultiplier.toFixed(2)}x`;
    });
  }

  const pauseBtn = document.getElementById('btn-pause');
  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      state.paused = !state.paused;
      syncButtons();
    });
  }

  const soundBtn = document.getElementById('btn-sound');
  if (soundBtn) {
    soundBtn.addEventListener('click', () => {
      soundscape.setEnabled(!soundscape.isEnabled());
      syncButtons();
    });
  }

  const toggleHudBtn = document.getElementById('btn-hud');
  const controlPanel = document.getElementById('controls-panel');
  if (toggleHudBtn && controlPanel) {
    toggleHudBtn.addEventListener('click', () => {
      const hidden = controlPanel.classList.toggle('collapsed');
      toggleHudBtn.textContent = hidden ? 'Show Controls' : 'Hide Controls';
    });
  }

  canvas.addEventListener('pointerdown', (e) => {
    state.dragging = true;
    state.lastPointerX = e.clientX;
    state.lastPointerY = e.clientY;
    if (state.cameraId !== 'orbit') {
      state.cameraId = 'orbit';
      syncButtons();
    }
  });

  window.addEventListener('pointermove', (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lastPointerX;
    const dy = e.clientY - state.lastPointerY;
    state.lastPointerX = e.clientX;
    state.lastPointerY = e.clientY;
    state.orbitYaw -= dx * 0.007;
    state.orbitPitch = THREE.MathUtils.clamp(state.orbitPitch + dy * 0.005, -0.08, 1.05);
  });

  window.addEventListener('pointerup', () => {
    state.dragging = false;
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (state.cameraId !== 'orbit') {
      state.cameraId = 'orbit';
      syncButtons();
    }
    state.orbitDist = THREE.MathUtils.clamp(state.orbitDist + e.deltaY * 0.004, 2.7, 13.5);
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      state.paused = !state.paused;
      syncButtons();
    } else if (e.key === '1') {
      setCamera('hero');
    } else if (e.key === '2') {
      setCamera('side');
    } else if (e.key === '3') {
      setCamera('low');
    } else if (e.key === '4') {
      setCamera('chase');
    } else if (e.key === '5') {
      setCamera('orbit');
    }
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

const hudSpeed = document.getElementById('hud-speed');
const hudPhase = document.getElementById('hud-phase');
const hudElevation = document.getElementById('hud-elevation');

function stepSimulation(dt) {
  const gait = GAITS[state.gaitId];
  const activeSpeed = gait.speedMps * state.speedMultiplier;

  if (!state.paused) {
    state.timeSec += dt;
    state.cycle = (state.cycle + dt * gait.strideHz * state.speedMultiplier) % 1;
    state.worldZ += activeSpeed * dt;
  }

  const trailX = getTrailCenterX(state.worldZ);
  const groundY = getTerrainHeight(trailX, state.worldZ);
  const normal = getTerrainNormal(trailX, state.worldZ);
  const trailSlopeX = getTrailSlopeX(state.worldZ);

  const pose = evaluateGaitPose(state.cycle, state.gaitId);
  const groundPitch = Math.atan2(normal.z, normal.y);
  const turnYaw = Math.atan(trailSlopeX) * 0.65;
  const turnBank = -trailSlopeX * 0.14;

  horse.root.position.set(0, groundY, 0);
  horse.root.rotation.set(groundPitch, turnYaw, turnBank);
  horse.applyPose(pose, state.timeSec, gait.strideScale * state.speedMultiplier);

  for (let i = 0; i < 4; i += 1) {
    const impact = pose.legs[i].groundImpact;
    if (impact && !state.prevImpact[i] && !state.paused) {
      world.spawnHoofDust(horse.hoofWorldPositions[i], gait.dustIntensity);
      soundscape.triggerHoofbeat(0.55 + gait.dustIntensity * 0.45, i);
    }
    state.prevImpact[i] = impact;
  }

  world.update(state.worldZ, state.paused ? 0 : activeSpeed, dt, state.timeSec);

  sunLight.position.copy(sunDirection).multiplyScalar(38).add(horse.root.position);
  sunLight.target.position.copy(horse.root.position);
  sunLight.target.updateMatrixWorld();

  if (state.cameraId === 'orbit') {
    const cx = Math.sin(state.orbitYaw) * Math.cos(state.orbitPitch) * state.orbitDist;
    const cy = Math.sin(state.orbitPitch) * state.orbitDist;
    const cz = Math.cos(state.orbitYaw) * Math.cos(state.orbitPitch) * state.orbitDist;
    targetCamPos.set(cx, groundY + 1.25 + cy, cz);
    targetLookAt.set(0, groundY + 1.25, 0);
  } else {
    const cfg = CAMERAS[state.cameraId];
    const gentleSway = Math.sin(state.timeSec * 0.55) * 0.18;
    targetCamPos.set(
      cfg.offset[0] + gentleSway,
      groundY + cfg.offset[1] + pose.bodyY * 0.18,
      cfg.offset[2]
    );
    targetLookAt.set(
      cfg.lookAt[0],
      groundY + cfg.lookAt[1],
      cfg.lookAt[2]
    );
  }

  const damp = 1 - Math.exp(-dt * 6.5);
  smoothCamPos.lerp(targetCamPos, damp);
  smoothLookAt.lerp(targetLookAt, damp);
  camera.position.copy(smoothCamPos);
  camera.lookAt(smoothLookAt);

  if (hudSpeed) {
    const mph = Math.round(gait.speedMph * state.speedMultiplier);
    hudSpeed.textContent = `${mph} mph`;
  }
  if (hudPhase) {
    hudPhase.textContent = pose.suspended ? 'Airborne' : gait.label;
  }
  if (hudElevation) {
    const ft = Math.round(9420 + groundY * 3.28);
    hudElevation.textContent = `${ft.toLocaleString()} ft`;
  }
}

let lastTime = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000));
  lastTime = now;
  stepSimulation(dt);
  renderer.render(scene, camera);
}

bindUI();
applyLighting('morning');
setCamera('hero', true);
stepSimulation(0.016);
renderer.render(scene, camera);
requestAnimationFrame(animate);

window.__HORSE_APP__ = {
  setState(options = {}) {
    if (options.gaitId) setGait(options.gaitId);
    if (options.coatId) setCoat(options.coatId);
    if (options.lightingId) applyLighting(options.lightingId);
    if (options.cameraId) setCamera(options.cameraId, true);
    if (typeof options.cycle === 'number') state.cycle = options.cycle;
    if (typeof options.worldZ === 'number') state.worldZ = options.worldZ;
    if (typeof options.paused === 'boolean') state.paused = options.paused;
    stepSimulation(0.016);
    if (options.cameraId) setCamera(options.cameraId, true);
    renderer.render(scene, camera);
    syncButtons();
  },
  getState() {
    return {
      gaitId: state.gaitId,
      coatId: state.coatId,
      lightingId: state.lightingId,
      cameraId: state.cameraId,
      cycle: state.cycle,
      worldZ: state.worldZ,
      paused: state.paused
    };
  }
};
