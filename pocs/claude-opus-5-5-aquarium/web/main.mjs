import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RACK_COLORS, MATERIALS, FISH, TANK, MAX_FOOD, byId } from './catalog.mjs';
import * as S from './state.mjs';
import { swimBox, createSwimmer, step, heading, patrol, nearestFood, canEat, sinkFood } from './swim.mjs';
import { sandHeight } from './patterns.mjs';
import { buildRoom, buildRack, buildTank, buildLights } from './tank.mjs';
import { buildFish } from './fish.mjs';
import { buildDecor } from './decor.mjs';
import { applyRack } from './materials.mjs';
import { createSound } from './sound.mjs';
import { buildUI } from './ui.mjs';

const KEY = 'aquarium-state';
const app = document.getElementById('app');
const panel = document.getElementById('panel');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b0d0e');
scene.fog = new THREE.Fog('#0b0d0e', 4, 9);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;

const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.enablePan = false;
controls.minDistance = 0.7;
controls.maxDistance = 4.5;
controls.maxPolarAngle = 1.62;
controls.target.set(0, -0.05, 0);
const HOME = new THREE.Vector3(0.75, 0.38, 2.45);
const home = HOME.clone();
const intro = { t: 0, from: new THREE.Vector3(-1.9, 1.1, 3.4) };
camera.position.copy(intro.from);
const stopIntro = () => { intro.t = Infinity; };
renderer.domElement.addEventListener('pointerdown', stopIntro, { once: true });
renderer.domElement.addEventListener('wheel', stopIntro, { once: true, passive: true });

const sound = createSound();
buildRoom(scene);
buildLights(scene);
const rackMaterial = buildRack(scene);
const tank = buildTank(scene, () => { if (Math.random() < 0.18) sound.bubble(); });
const decor = buildDecor(scene);
const box = swimBox();

const school = [];
function syncFish(list) {
  for (const f of FISH) {
    const want = list.filter(id => id === f.id).length;
    const have = school.filter(s => s.def.id === f.id);
    for (let i = have.length; i < want; i++) {
      const model = buildFish(f);
      const swimmer = createSwimmer(Math.random, box, f.band, f.speed * (0.85 + Math.random() * 0.3));
      if (applied) swimmer.pos.y = box.maxY;
      scene.add(model.root);
      school.push({ def: f, model, swimmer });
    }
    for (let i = want; i < have.length; i++) {
      const gone = have[i];
      scene.remove(gone.model.root);
      school.splice(school.indexOf(gone), 1);
    }
  }
}

const shark = { model: buildFish('shark'), t: 0 };

const flakeGeo = new THREE.BoxGeometry(0.009, 0.0015, 0.007);
const flakeColors = ['#d9822b', '#c9452a', '#e3b341', '#7a9a3a'].map(c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.8, transparent: true }));
const foods = [];

function feed() {
  const x = (Math.random() - 0.5) * (TANK.width * 0.5);
  const z = (Math.random() - 0.5) * (TANK.depth * 0.4);
  for (let i = 0; i < 8 && foods.length < MAX_FOOD; i++) {
    const f = { x: x + (Math.random() - 0.5) * 0.08, y: TANK.water - 0.004, z: z + (Math.random() - 0.5) * 0.06, phase: Math.random() * 6, rest: 0 };
    f.floor = sandHeight(f.x, f.z) + 0.002;
    f.mesh = new THREE.Mesh(flakeGeo, flakeColors[i % flakeColors.length].clone());
    f.mesh.rotation.set(Math.random(), Math.random() * 6, Math.random());
    scene.add(f.mesh);
    foods.push(f);
  }
  sound.splash();
}

function dropFood(f) {
  scene.remove(f.mesh);
  f.mesh.material.dispose();
  foods.splice(foods.indexOf(f), 1);
}

function updateFood(dt) {
  for (const f of [...foods]) {
    sinkFood(f, dt, f.floor);
    f.mesh.position.set(f.x, f.y, f.z);
    f.mesh.rotation.y += dt * 1.5;
    if (f.rest > 4) f.mesh.material.opacity = Math.max(0, 1 - (f.rest - 4) / 3);
    if (f.rest > 7) dropFood(f);
  }
}
scene.add(shark.model.root);

let state = S.restore(load());
let applied = null;

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY));
  } catch {
    return null;
  }
}

function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {}
}

function apply() {
  if (!applied || applied.material !== state.material || applied.rackColor !== state.rackColor) {
    applyRack(rackMaterial, byId(MATERIALS, state.material), byId(RACK_COLORS, state.rackColor).hex);
  }
  syncFish(state.fish);
  tank.setSubstrate(state.substrate);
  decor.show(state);
  shark.model.root.visible = state.shark;
  ui.render(state);
  applied = state;
  save();
}

function dispatch(next, effect) {
  if (next === state) return;
  state = next;
  apply();
  if (effect) effect();
}

const ui = buildUI(panel, {
  rackColor: id => dispatch(S.setRackColor(state, id), sound.click),
  material: id => dispatch(S.setMaterial(state, id), sound.click),
  addFish: id => dispatch(S.addFish(state, id), sound.splash),
  removeFish: id => dispatch(S.removeFish(state, id), sound.scoop),
  clearFish: () => dispatch(S.clearFish(state), sound.scoop),
  toggleDecor: id => dispatch(S.toggleDecor(state, id), sound.thunk),
  substrate: id => dispatch(S.setSubstrate(state, id), sound.scoop),
  toggleScape: id => dispatch(S.toggleScape(state, id), sound.thunk),
  grass: delta => dispatch(S.setGrass(state, state.grass + delta), delta > 0 ? sound.thunk : sound.scoop),
  feed,
  toggleShark: () => dispatch(S.toggleShark(state), () => { if (state.shark) sound.jaws(); else sound.scoop(); }),
  toggleSound: async () => {
    dispatch(S.toggleSound(state));
    if (state.sound) {
      await sound.enable();
      sound.bubble();
    } else {
      await sound.disable();
    }
  },
  volume: v => sound.setVolume(v)
});
apply();

document.getElementById('toggle').addEventListener('click', () => document.body.classList.toggle('panel-open'));

function resize() {
  const w = app.clientWidth;
  const h = app.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  const fit = Math.max(1, 0.9 / camera.aspect);
  home.copy(HOME).multiplyScalar(fit);
  controls.maxDistance = 4.5 * fit;
  const side = w > 820 ? panel.offsetWidth + 24 : 0;
  if (side) camera.setViewOffset(w, h, -side / 2, 0, w, h);
  else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(app);
resize();

let time = 0;
let last = performance.now();
function place(root, pos, vel) {
  const h = heading(vel);
  root.position.set(pos.x, pos.y, pos.z);
  root.rotation.set(0, h.yaw, h.pitch);
}

renderer.setAnimationLoop(now => {
  const dt = Math.max(0, Math.min((now - last) / 1000, 1 / 20));
  last = now;
  time += dt;
  if (intro.t <= 1) {
    intro.t += dt / 3.5;
    const k = Math.min(1, intro.t);
    camera.position.lerpVectors(intro.from, home, 1 - Math.pow(1 - k, 3));
  }
  let threat = null;
  if (state.shark) {
    shark.t += dt;
    const pose = patrol(shark.t, swimBox(0.13));
    place(shark.model.root, pose.pos, pose.vel);
    shark.model.animate(dt, 0.05);
    threat = pose.pos;
  }
  updateFood(dt);
  const reachable = foods.filter(f => f.y >= box.minY && f.y <= box.maxY + 0.02);
  for (const f of school) {
    const food = nearestFood(f.swimmer.pos, reachable);
    step(f.swimmer, dt, Math.random, box, threat, food);
    if (food && canEat(f.swimmer.pos, food) && foods.includes(food)) {
      dropFood(food);
      reachable.splice(reachable.indexOf(food), 1);
      sound.munch();
    }
    place(f.model.root, f.swimmer.pos, f.swimmer.vel);
    f.model.animate(dt, Math.hypot(f.swimmer.vel.x, f.swimmer.vel.y, f.swimmer.vel.z));
  }
  tank.update(dt, time);
  decor.update(time);
  controls.update(dt);
  renderer.render(scene, camera);
});

window.__AQUARIUM = { get state() { return state; }, fishCount: () => school.length,
  foodCount: () => foods.length,
  view(pos, target) {
    stopIntro();
    camera.position.set(...pos);
    controls.target.set(...target);
  }
};
