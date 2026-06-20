import * as THREE from "./vendor/three.module.js";

const W = 960, H = 540;

let renderer, scene, camera, raycaster, sun;
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const ndc = new THREE.Vector2();
const hitPoint = new THREE.Vector3();

const units = new Map();
const tcEntries = { me: null, ai: null };
const mineMeshes = [];
let rallyRing;

const MAT = {
  blue: new THREE.MeshStandardMaterial({ color: 0x2f5fb0, roughness: 0.6, metalness: 0.1 }),
  red: new THREE.MeshStandardMaterial({ color: 0xb33a2a, roughness: 0.6, metalness: 0.1 }),
  blueRoof: new THREE.MeshStandardMaterial({ color: 0x3a6fc4, roughness: 0.7 }),
  redRoof: new THREE.MeshStandardMaterial({ color: 0xc8503c, roughness: 0.7 }),
  stone: new THREE.MeshStandardMaterial({ color: 0xcdbb93, roughness: 0.9 }),
  wood: new THREE.MeshStandardMaterial({ color: 0x6b4a2a, roughness: 0.9 }),
  skin: new THREE.MeshStandardMaterial({ color: 0xe2b48c, roughness: 0.8 }),
  gold: new THREE.MeshStandardMaterial({ color: 0xe8c84e, roughness: 0.25, metalness: 0.5, emissive: 0x6b5510, emissiveIntensity: 0.45 }),
  goldDead: new THREE.MeshStandardMaterial({ color: 0x8c8a7a, roughness: 0.95 }),
  trunk: new THREE.MeshStandardMaterial({ color: 0x5b3a1e, roughness: 0.95 }),
  leaf: new THREE.MeshStandardMaterial({ color: 0x4f8a3a, roughness: 0.95 }),
  leafDark: new THREE.MeshStandardMaterial({ color: 0x3c7330, roughness: 0.95 }),
  rock: new THREE.MeshStandardMaterial({ color: 0x9a9488, roughness: 1 }),
  sand: new THREE.MeshStandardMaterial({ color: 0xdcc38f, roughness: 1 }),
  water: new THREE.MeshStandardMaterial({ color: 0x2f6dab, roughness: 0.18, metalness: 0.35, emissive: 0x12314f, emissiveIntensity: 0.25, transparent: true, opacity: 0.92 }),
  grassPatch: new THREE.MeshStandardMaterial({ color: 0x6fae4f, roughness: 0.98 }),
  flower: new THREE.MeshStandardMaterial({ color: 0xf0d24a, roughness: 0.8, emissive: 0x6b5a10, emissiveIntensity: 0.3 }),
};

const GEO = {
  villBody: new THREE.CylinderGeometry(3.4, 4.6, 9, 8),
  villHead: new THREE.SphereGeometry(3, 10, 8),
  solBody: new THREE.CylinderGeometry(3.8, 5, 10, 8),
  solHead: new THREE.SphereGeometry(3.2, 10, 8),
  solHelmet: new THREE.ConeGeometry(3.5, 3.6, 8),
  spear: new THREE.CylinderGeometry(0.5, 0.5, 20, 6),
  spearTip: new THREE.ConeGeometry(1.2, 3, 6),
  crystal: new THREE.OctahedronGeometry(1),
  flower: new THREE.SphereGeometry(1.5, 6, 5),
};

function ownerMat(owner) { return owner === "me" ? MAT.blue : MAT.red; }
function ownerRoof(owner) { return owner === "me" ? MAT.blueRoof : MAT.redRoof; }

function rng(s) { const x = Math.sin(s * 99.13) * 10000; return x - Math.floor(x); }

function buildTree(x, z, s) {
  const g = new THREE.Group();
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 2.3, 10, 7), MAT.trunk);
  trunk.position.y = 5; trunk.castShadow = true; g.add(trunk);
  const c1 = new THREE.Mesh(new THREE.ConeGeometry(7, 13, 7), MAT.leaf);
  c1.position.y = 15; c1.castShadow = true; g.add(c1);
  const c2 = new THREE.Mesh(new THREE.ConeGeometry(5.4, 9, 7), MAT.leafDark);
  c2.position.y = 21; c2.castShadow = true; g.add(c2);
  g.position.set(x, 0, z); g.scale.setScalar(s);
  return g;
}

function buildRock(x, z, s) {
  const r = new THREE.Mesh(new THREE.DodecahedronGeometry(4), MAT.rock);
  r.position.set(x, 2, z); r.scale.setScalar(s); r.castShadow = true; r.receiveShadow = true;
  return r;
}

function farFromBases(x, z) {
  const d = (ax, az) => Math.hypot(x - ax, z - az);
  return d(120, 430) > 90 && d(840, 110) > 90 && d(480, 270) > 70 &&
    d(220, 470) > 50 && d(90, 320) > 50 && d(740, 70) > 50 && d(870, 230) > 50 &&
    d(165, 120) > 105 && d(820, 445) > 118;
}

function buildPond(x, z, r) {
  const sand = new THREE.Mesh(new THREE.CircleGeometry(r * 1.32, 30), MAT.sand);
  sand.rotation.x = -Math.PI / 2;
  sand.position.set(x, 0.06, z);
  sand.receiveShadow = true;
  scene.add(sand);
  const water = new THREE.Mesh(new THREE.CircleGeometry(r, 30), MAT.water);
  water.rotation.x = -Math.PI / 2;
  water.position.set(x, 0.14, z);
  scene.add(water);
}

export function init(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xcfe8f7);
  scene.fog = new THREE.Fog(0xcfe8f7, 760, 1700);

  camera = new THREE.PerspectiveCamera(40, W / H, 1, 4000);
  camera.position.set(480, 500, 880);
  camera.lookAt(480, 0, 244);

  raycaster = new THREE.Raycaster();

  scene.add(new THREE.HemisphereLight(0xcfe8f7, 0x5f7d36, 0.95));
  sun = new THREE.DirectionalLight(0xfff3d8, 1.05);
  sun.position.set(260, 700, 560);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 2200;
  sun.shadow.camera.left = -640;
  sun.shadow.camera.right = 640;
  sun.shadow.camera.top = 520;
  sun.shadow.camera.bottom = -520;
  sun.shadow.bias = -0.0004;
  sun.target.position.set(480, 0, 270);
  scene.add(sun);
  scene.add(sun.target);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(1500, 1100),
    new THREE.MeshStandardMaterial({ color: 0x7bbd5a, roughness: 0.98 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(480, 0, 270);
  ground.receiveShadow = true;
  scene.add(ground);

  for (let i = 0; i < 14; i++) {
    const x = rng(i + 200) * 980, z = rng(i + 240) * 560;
    const patch = new THREE.Mesh(new THREE.CircleGeometry(30 + rng(i + 260) * 40, 14), MAT.grassPatch);
    patch.rotation.x = -Math.PI / 2;
    patch.position.set(x, 0.02, z);
    patch.receiveShadow = true;
    scene.add(patch);
  }

  buildPond(165, 120, 70);
  buildPond(820, 445, 82);

  for (let i = 0; i < 130; i++) {
    const x = rng(i + 500) * 980, z = rng(i + 560) * 560;
    if (!farFromBases(x, z)) continue;
    const f = new THREE.Mesh(GEO.flower, MAT.flower);
    f.position.set(x, 1.4, z);
    scene.add(f);
  }

  for (let i = 0; i < 36; i++) {
    const x = rng(i + 3) * 1000 - 20, z = rng(i + 41) * 600 - 30;
    if (farFromBases(x, z)) scene.add(buildTree(x, z, 0.8 + rng(i + 7) * 0.7));
  }
  for (let i = 0; i < 16; i++) {
    const x = rng(i + 70) * 1000 - 20, z = rng(i + 90) * 600 - 30;
    if (farFromBases(x, z)) scene.add(buildRock(x, z, 0.7 + rng(i + 11) * 0.9));
  }

  rallyRing = new THREE.Mesh(
    new THREE.TorusGeometry(11, 1.6, 8, 24),
    new THREE.MeshBasicMaterial({ color: 0x2f5fb0 })
  );
  rallyRing.rotation.x = -Math.PI / 2;
  rallyRing.visible = false;
  scene.add(rallyRing);
}

function buildTc(tc) {
  const g = new THREE.Group();
  const base = new THREE.Mesh(new THREE.BoxGeometry(58, 28, 44), MAT.stone);
  base.position.y = 14; base.castShadow = true; base.receiveShadow = true; g.add(base);
  const roof = new THREE.Mesh(new THREE.ConeGeometry(42, 26, 4), ownerRoof(tc.owner));
  roof.position.y = 41; roof.rotation.y = Math.PI / 4; roof.castShadow = true; g.add(roof);
  const door = new THREE.Mesh(new THREE.BoxGeometry(14, 18, 2), MAT.wood);
  door.position.set(0, 9, 22.5); g.add(door);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 24, 6), MAT.wood);
  pole.position.set(22, 40, 0); g.add(pole);
  const flag = new THREE.Mesh(new THREE.PlaneGeometry(13, 8), ownerMat(tc.owner));
  flag.material.side = THREE.DoubleSide; flag.position.set(29, 47, 0); g.add(flag);
  g.position.set(tc.x, 0, tc.y);
  return g;
}

function buildMine() {
  const g = new THREE.Group();
  const base = new THREE.Mesh(new THREE.CylinderGeometry(12, 14, 4, 10), MAT.rock);
  base.position.y = 2; base.receiveShadow = true; g.add(base);
  const seeds = [[0, 6, 0, 4], [6, 8, 2, 3], [-5, 7, 4, 3], [3, 9, -5, 2.4], [-4, 8, -4, 2.6]];
  const crystals = [];
  for (const [x, y, z, s] of seeds) {
    const c = new THREE.Mesh(GEO.crystal, MAT.gold);
    c.position.set(x, y, z); c.scale.setScalar(s); c.castShadow = true;
    g.add(c); crystals.push(c);
  }
  g.userData.crystals = crystals;
  return g;
}

function buildVillager(owner) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(GEO.villBody, ownerMat(owner));
  body.position.y = 4.5; body.castShadow = true; g.add(body);
  const head = new THREE.Mesh(GEO.villHead, MAT.skin);
  head.position.y = 11.5; head.castShadow = true; g.add(head);
  return g;
}

function buildSoldier(owner) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(GEO.solBody, ownerMat(owner));
  body.position.y = 5; body.castShadow = true; g.add(body);
  const head = new THREE.Mesh(GEO.solHead, MAT.skin);
  head.position.y = 12.5; head.castShadow = true; g.add(head);
  const helmet = new THREE.Mesh(GEO.solHelmet, MAT.stone);
  helmet.position.y = 15; g.add(helmet);
  const spear = new THREE.Mesh(GEO.spear, MAT.wood);
  spear.position.set(5.5, 11, 0); spear.rotation.z = 0.16; spear.castShadow = true; g.add(spear);
  const tip = new THREE.Mesh(GEO.spearTip, MAT.stone);
  tip.position.set(7.2, 21, 0); g.add(tip);
  return g;
}

function buildBar(width) {
  const grp = new THREE.Group();
  const bg = new THREE.Mesh(
    new THREE.PlaneGeometry(width + 1.6, 3.4),
    new THREE.MeshBasicMaterial({ color: 0x241f16, depthTest: false, transparent: true })
  );
  bg.renderOrder = 10;
  const fg = new THREE.Mesh(
    new THREE.PlaneGeometry(width, 2.3),
    new THREE.MeshBasicMaterial({ color: 0x3fae46, depthTest: false, transparent: true })
  );
  fg.position.z = 0.2; fg.renderOrder = 11;
  grp.add(bg); grp.add(fg);
  grp.userData = { fg, width };
  grp.visible = false;
  return grp;
}

function updateBar(grp, ratio) {
  if (ratio >= 1) { grp.visible = false; return; }
  grp.visible = true;
  ratio = Math.max(0, ratio);
  const { fg, width } = grp.userData;
  fg.scale.x = ratio;
  fg.position.x = -(width * (1 - ratio)) / 2;
  fg.material.color.setHex(ratio > 0.5 ? 0x3fae46 : ratio > 0.25 ? 0xd9a521 : 0xc43c2c);
}

function createUnitEntry(u) {
  const root = u.type === "villager" ? buildVillager(u.owner) : buildSoldier(u.owner);
  root.scale.setScalar(u.type === "villager" ? 1.5 : 1.9);
  root.position.set(u.x, 0, u.y);
  scene.add(root);
  const bar = buildBar(20);
  scene.add(bar);
  return { root, bar, px: u.x, pz: u.y, barY: u.type === "villager" ? 28 : 42 };
}

export function newMatch(game) {
  for (const [, e] of units) { scene.remove(e.root); scene.remove(e.bar); }
  units.clear();
  for (const m of mineMeshes) scene.remove(m);
  mineMeshes.length = 0;
  for (const owner of ["me", "ai"]) {
    if (tcEntries[owner]) { scene.remove(tcEntries[owner].root); scene.remove(tcEntries[owner].bar); }
  }
  game.mines.forEach((m) => {
    const mesh = buildMine();
    mesh.position.set(m.x, 0, m.y);
    scene.add(mesh);
    mineMeshes.push(mesh);
  });
  for (const owner of ["me", "ai"]) {
    const tc = game.tc[owner];
    const root = buildTc(tc);
    scene.add(root);
    const bar = buildBar(70);
    bar.position.set(tc.x, 80, tc.y);
    scene.add(bar);
    tcEntries[owner] = { root, bar };
  }
}

export function update(game) {
  const seen = new Set();
  for (const u of game.units) {
    seen.add(u.id);
    let e = units.get(u.id);
    if (!e) { e = createUnitEntry(u); units.set(u.id, e); }
    const dx = u.x - e.px, dz = u.y - e.pz;
    if (dx * dx + dz * dz > 0.6) { e.root.rotation.y = Math.atan2(dx, dz); e.px = u.x; e.pz = u.y; }
    e.root.position.set(u.x, 0, u.y);
    e.bar.position.set(u.x, e.barY, u.y);
    e.bar.quaternion.copy(camera.quaternion);
    updateBar(e.bar, u.hp / u.maxHp);
  }
  for (const [id, e] of units) {
    if (!seen.has(id)) { scene.remove(e.root); scene.remove(e.bar); units.delete(id); }
  }

  for (const owner of ["me", "ai"]) {
    const tc = game.tc[owner];
    const e = tcEntries[owner];
    if (!e) continue;
    e.bar.quaternion.copy(camera.quaternion);
    updateBar(e.bar, tc.hp / tc.maxHp);
    e.root.visible = tc.hp > 0;
  }

  for (let i = 0; i < mineMeshes.length; i++) {
    const m = game.mines[i];
    const mesh = mineMeshes[i];
    if (m && m.amount <= 0 && !mesh.userData.dead) {
      mesh.userData.crystals.forEach((c) => { c.material = MAT.goldDead; c.scale.multiplyScalar(0.6); });
      mesh.userData.dead = true;
    }
  }

  if (game.controller.me === "human" && game.order.me.mode === "rally") {
    rallyRing.visible = true;
    rallyRing.position.set(game.order.me.rx, 0.6, game.order.me.ry);
  } else {
    rallyRing.visible = false;
  }

  renderer.render(scene, camera);
}

export function screenToWorld(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(groundPlane, hitPoint)) return null;
  return { x: hitPoint.x, y: hitPoint.z };
}
