import * as THREE from "three";

const params = new URLSearchParams(location.search);
let seed = 20260704;
function rnd() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x8dc6ea, 220, 1900);

const camera = new THREE.PerspectiveCamera(62, innerWidth / innerHeight, 0.1, 7000);
camera.position.set(880, 9, 220);

const hemi = new THREE.HemisphereLight(0xbfd8ff, 0x51653f, 0.9);
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
scene.add(hemi, sun);

function lam(color) {
  return new THREE.MeshLambertMaterial({ color });
}

const ground = new THREE.Mesh(new THREE.CircleGeometry(3200, 72), lam(0x6f9450));
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const waypoints = [
  [900, 0], [850, -350], [650, -650], [350, -900], [-250, -880], [-800, -600],
  [-1010, -200], [-1000, 220], [-680, 620], [-200, 900], [420, 720], [800, 360]
].map(p => new THREE.Vector3(p[0], 0, p[1]));
const curve = new THREE.CatmullRomCurve3(waypoints, true, "centripetal");
const trackLen = curve.getLength();

const SAMPLES = 2400;
const samplePts = [];
for (let i = 0; i < SAMPLES; i++) samplePts.push(curve.getPointAt(i / SAMPLES));

function nearestU(v) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < SAMPLES; i++) {
    const d = samplePts[i].distanceToSquared(v);
    if (d < bd) { bd = d; best = i; }
  }
  return best / SAMPLES;
}

function trackDist(x, z) {
  let bd = Infinity;
  for (let i = 0; i < SAMPLES; i += 3) {
    const dx = samplePts[i].x - x, dz = samplePts[i].z - z;
    const d = dx * dx + dz * dz;
    if (d < bd) bd = d;
  }
  return Math.sqrt(bd);
}

function ribbon(width, y, offset, color) {
  const n = 1600;
  const pos = new Float32Array(n * 6);
  const nor = new Float32Array(n * 6);
  const idx = [];
  for (let i = 0; i < n; i++) {
    const u = i / n;
    const p = curve.getPointAt(u);
    const t = curve.getTangentAt(u);
    const nx = t.z, nz = -t.x;
    const cx = p.x + nx * offset, cz = p.z + nz * offset;
    pos.set([cx - nx * width / 2, y, cz - nz * width / 2, cx + nx * width / 2, y, cz + nz * width / 2], i * 6);
    nor.set([0, 1, 0, 0, 1, 0], i * 6);
    const a = i * 2, b = a + 1, c = (a + 2) % (n * 2), d = (a + 3) % (n * 2);
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  g.setIndex(idx);
  const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide }));
  scene.add(m);
}

ribbon(5.4, 0.06, 0, 0x6f6252);
ribbon(0.3, 0.3, -1.0, 0x50555b);
ribbon(0.3, 0.3, 1.0, 0x50555b);

const sleeperCount = Math.floor(trackLen / 5);
const sleepers = new THREE.InstancedMesh(new THREE.BoxGeometry(3.2, 0.18, 1.05), lam(0x4c3b28), sleeperCount);
const dummy = new THREE.Object3D();
for (let i = 0; i < sleeperCount; i++) {
  const u = i / sleeperCount;
  const p = curve.getPointAt(u), t = curve.getTangentAt(u);
  dummy.position.set(p.x, 0.18, p.z);
  dummy.rotation.y = Math.atan2(t.x, t.z);
  dummy.updateMatrix();
  sleepers.setMatrixAt(i, dummy.matrix);
}
sleepers.instanceMatrix.needsUpdate = true;
scene.add(sleepers);

function box(g, w, h, d, color, px, py, pz, ry = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), lam(color));
  m.position.set(px, py, pz);
  m.rotation.y = ry;
  g.add(m);
  return m;
}

function cyl(g, r, h, color, px, py, pz) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 14), lam(color));
  m.position.set(px, py, pz);
  g.add(m);
  return m;
}

function cone(g, r, h, color, px, py, pz, seg = 14, ry = 0) {
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, seg), lam(color));
  m.position.set(px, py, pz);
  m.rotation.y = ry;
  g.add(m);
  return m;
}

function sph(g, r, color, px, py, pz) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 9), lam(color));
  m.position.set(px, py, pz);
  g.add(m);
  return m;
}

function flat(g, w, d, color, px, py, pz) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), lam(color));
  m.rotation.x = -Math.PI / 2;
  m.position.set(px, py, pz);
  g.add(m);
  return m;
}

function tree(g, x, z, s) {
  cyl(g, 0.32 * s, 8.6 * s, 0x6b4a33, x, 4.3 * s, z);
  cone(g, 4.3 * s, 2.6 * s, 0x2f5b34, x, 9.4 * s, z, 9);
}

function textPanel(text) {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 128;
  const x = c.getContext("2d");
  x.fillStyle = "#101c26";
  x.fillRect(0, 0, 512, 128);
  x.strokeStyle = "#e8c987";
  x.lineWidth = 10;
  x.strokeRect(8, 8, 496, 112);
  x.fillStyle = "#f4ead6";
  let size = 58;
  x.font = "bold " + size + "px Georgia";
  while (x.measureText(text).width > 455 && size > 22) {
    size -= 2;
    x.font = "bold " + size + "px Georgia";
  }
  x.textAlign = "center";
  x.textBaseline = "middle";
  x.fillText(text, 256, 66);
  return new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide });
}

function drawBrazil(x, w, h) {
  x.fillStyle = "#009b3a";
  x.fillRect(0, 0, w, h);
  x.fillStyle = "#fedf00";
  x.beginPath();
  x.moveTo(w / 2, 8);
  x.lineTo(w - 12, h / 2);
  x.lineTo(w / 2, h - 8);
  x.lineTo(12, h / 2);
  x.closePath();
  x.fill();
  x.fillStyle = "#002776";
  x.beginPath();
  x.arc(w / 2, h / 2, 17, 0, 7);
  x.fill();
}

function drawUruguay(x, w, h) {
  x.fillStyle = "#ffffff";
  x.fillRect(0, 0, w, h);
  x.fillStyle = "#0038a8";
  for (const i of [1, 3, 5, 7]) x.fillRect(0, i * h / 9, w, h / 9);
  x.fillStyle = "#ffffff";
  x.fillRect(0, 0, w / 2, h * 5 / 9);
  x.fillStyle = "#fcd116";
  x.beginPath();
  x.arc(w / 4, h * 5 / 18, 14, 0, 7);
  x.fill();
}

function drawArgentina(x, w, h) {
  x.fillStyle = "#74acdf";
  x.fillRect(0, 0, w, h);
  x.fillStyle = "#ffffff";
  x.fillRect(0, h / 3, w, h / 3);
  x.fillStyle = "#f6b40e";
  x.beginPath();
  x.arc(w / 2, h / 2, 12, 0, 7);
  x.fill();
}

function flag(g, px, pz, draw) {
  cyl(g, 0.12, 11, 0x9aa0a6, px, 5.5, pz);
  const c = document.createElement("canvas");
  c.width = 160;
  c.height = 100;
  draw(c.getContext("2d"), 160, 100);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(6.6, 4.1),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide }));
  m.position.set(px + 3.4, 8.7, pz);
  g.add(m);
}

function buildPortoAlegre(g) {
  box(g, 34, 11, 11, 0xe6e0d1, 0, 5.5, -8);
  box(g, 35, 1.2, 12, 0xb9b2a2, 0, 11.6, -8);
  box(g, 34.2, 2.4, 11.1, 0x2f3a42, 0, 6.4, -8);
  cyl(g, 2.1, 46, 0xa14b32, 14, 23, -8);
  cyl(g, 2.4, 2, 0x7c3423, 14, 45, -8);
  box(g, 8, 30, 8, 0x8e99a5, -28, 15, -34);
  box(g, 7, 24, 7, 0x9fa8b2, -16, 12, -38);
  box(g, 9, 38, 9, 0x7d8794, -40, 19, -24);
  box(g, 6, 20, 6, 0xa8b0b8, 28, 10, -34);
  box(g, 16, 6, 10, 0xdfc06a, 32, 3, -8);
  box(g, 17, 1, 11, 0xf0e7d4, 32, 6.7, -8);
  cyl(g, 1.3, 2.4, 0x8b8b85, -10, 1.2, 6);
  cyl(g, 0.45, 3.2, 0x74572e, -10, 4, 6);
  sph(g, 0.55, 0x74572e, -10, 6.1, 6);
  flat(g, 540, 320, 0x3e6e8c, 0, 0.04, -210);
}

function buildGramado(g) {
  function chalet(x, z, s, wall, roof) {
    box(g, 6 * s, 3 * s, 5 * s, wall, x, 1.5 * s, z);
    cone(g, 4.4 * s, 2.8 * s, roof, x, 4.4 * s, z, 4, Math.PI / 4);
  }
  chalet(-16, -8, 1, 0x7a5236, 0x6e2f22);
  chalet(-7, -14, 1.1, 0xe9dcc3, 0x74362a);
  chalet(14, -10, 0.9, 0x8a5f3a, 0x5f2a20);
  chalet(22, -4, 0.8, 0xdbc9a6, 0x6e2f22);
  box(g, 12, 5.5, 9, 0xe8ddc4, 2, 2.75, -4);
  cone(g, 8.6, 4.2, 0x77402c, 2, 7.8, -4, 4, Math.PI / 4);
  flat(g, 3, 12, 0x9c2430, 2, 0.06, 6);
  box(g, 1, 1, 1, 0x6d6a63, -1.5, 0.5, 10);
  cone(g, 0.45, 1.6, 0xc9a437, -1.5, 1.8, 10, 10);
  box(g, 1, 1, 1, 0x6d6a63, 5.5, 0.5, 10);
  cone(g, 0.45, 1.6, 0xc9a437, 5.5, 1.8, 10, 10);
  const hues = [0x7c86c9, 0x8f7cc9, 0x6c9ac9];
  for (let i = 0; i < 9; i++) {
    const r = 0.8 + rnd() * 0.9;
    sph(g, r, hues[i % 3], -22 + rnd() * 48, r * 0.7, 2 + rnd() * 8);
  }
  tree(g, -26, -18, 0.9);
  tree(g, 28, -16, 1);
}

function buildCanela(g) {
  box(g, 9, 9, 16, 0x92928a, 0, 4.5, -10);
  box(g, 5.5, 16, 5.5, 0x9a9a92, 0, 8, 0);
  cone(g, 3.9, 15, 0x83837b, 0, 23.5, 0, 8);
  box(g, 0.35, 3, 0.35, 0xd9d9d2, 0, 32.5, 0);
  box(g, 1.7, 0.35, 0.35, 0xd9d9d2, 0, 32.9, 0);
  box(g, 2.6, 5, 0.6, 0x3a3630, 0, 2.5, 2.85);
  cone(g, 0.7, 3, 0x83837b, -3.2, 17.5, 3.2, 6);
  cone(g, 0.7, 3, 0x83837b, 3.2, 17.5, 3.2, 6);
  box(g, 22, 16, 8, 0x6e6a60, -32, 8, -28);
  const fall = new THREE.Mesh(new THREE.PlaneGeometry(4.4, 15), new THREE.MeshBasicMaterial({ color: 0xeef7fb }));
  fall.position.set(-32, 8, -23.9);
  g.add(fall);
  flat(g, 12, 9, 0x4f86a0, -32, 0.05, -16);
  tree(g, -14, -20, 1.1);
  tree(g, 12, -18, 1.2);
  tree(g, 20, -8, 0.9);
}

function buildMissoes(g) {
  const red = 0x94502f, red2 = 0x8a4629;
  for (const x of [-12, -4, 4, 12]) box(g, 3, 10, 2.4, red, x, 5, 0);
  box(g, 28, 2.4, 2.4, red2, 0, 11.2, 0);
  for (const x of [-8, 0, 8]) box(g, 2.6, 5, 2.2, red, x, 14.9, 0);
  box(g, 20, 2, 2.2, red2, 0, 18.4, 0);
  box(g, 10, 2.2, 2.2, red, 0, 20.5, 0);
  box(g, 0.4, 3.2, 0.4, red2, 0, 23.2, 0);
  box(g, 1.8, 0.4, 0.4, red2, 0, 23.6, 0);
  box(g, 2, 6, 20, red2, -15.5, 3, -11);
  box(g, 2, 5, 16, red, 15.5, 2.5, -9);
  for (let i = 0; i < 10; i++) {
    const h = 2 + rnd() * 5;
    cyl(g, 0.9, h, red, -9 + (i % 5) * 4.5, h / 2, -8 - Math.floor(i / 5) * 6);
  }
  flat(g, 44, 30, 0x9b8a66, 0, 0.03, -12);
}

function buildLivramento(g) {
  flat(g, 52, 44, 0xcfc4ae, 0, 0.05, -10);
  box(g, 3.4, 2, 3.4, 0xd8d8d2, 0, 1, -10);
  box(g, 1.7, 13, 1.7, 0xe8e8e2, 0, 8.5, -10);
  cone(g, 1.35, 2.2, 0xe8e8e2, 0, 16.1, -10, 4, Math.PI / 4);
  flag(g, -10, -10, drawBrazil);
  flag(g, 10, -10, drawUruguay);
  for (const x of [-16, 16]) box(g, 3, 0.5, 1, 0x6d5b43, x, 0.6, -2);
  box(g, 10, 4, 6, 0xd9c9a8, -26, 2, -30);
  box(g, 8, 5, 6, 0xc9b498, 26, 2.5, -30);
  box(g, 7, 3.5, 6, 0xbfae90, 14, 1.75, -32);
}

function buildUruguaiana(g) {
  flat(g, 540, 110, 0x3e6e8c, 0, 0.04, -90);
  box(g, 7, 1.6, 132, 0xb5b0a6, 22, 10, -90);
  box(g, 0.5, 1.4, 132, 0x8b867c, 19, 11.4, -90);
  box(g, 0.5, 1.4, 132, 0x8b867c, 25, 11.4, -90);
  for (const z of [-140, -115, -90, -65, -40]) box(g, 5, 10, 3, 0x9d988e, 22, 5, z);
  box(g, 8, 24, 4, 0xa8a398, 22, 12, -42);
  box(g, 8, 24, 4, 0xa8a398, 22, 12, -138);
  flag(g, 8, -22, drawBrazil);
  flag(g, 34, -160, drawArgentina);
  box(g, 12, 5, 7, 0xd8cdb4, -30, 2.5, -20);
  for (let i = 0; i < 5; i++) {
    const cx = -34 + rnd() * 20, cz = -12 - rnd() * 26;
    box(g, 2.4, 1.4, 1.2, 0x5b3d28, cx, 1, cz);
    box(g, 0.8, 0.8, 0.8, 0x4e3322, cx + 1.5, 1.3, cz);
  }
}

const cities = [
  {
    name: "Porto Alegre", stop: "Porto Alegre", sign: "Porto Alegre", wp: 0, build: buildPortoAlegre,
    info: "Capital of Rio Grande do Sul on the shore of the Guaíba. The 1928 Gasômetro powerhouse, now a cultural center, anchors the waterfront where the whole city gathers to watch the sunset."
  },
  {
    name: "Gramado", stop: "Gramado", sign: "Gramado", wp: 2, build: buildGramado,
    info: "Serra Gaúcha mountain town settled by German and Italian immigrants, famous for its chalets, hydrangeas, the Gramado Film Festival at the Palácio dos Festivais, and the Natal Luz."
  },
  {
    name: "Canela", stop: "Canela", sign: "Canela", wp: 3, build: buildCanela,
    info: "Home of the stone Cathedral of Our Lady of Lourdes and the Caracol Falls, a 131 meter free-falling cascade hidden among araucaria pine forests."
  },
  {
    name: "São Miguel das Missões", stop: "São Miguel das Missões", sign: "São Miguel", wp: 5, build: buildMissoes,
    info: "UNESCO World Heritage ruins of São Miguel Arcanjo, a Jesuit-Guarani mission raised in red sandstone in the 17th century, heart of the old Sete Povos das Missões."
  },
  {
    name: "Uruguaiana · Fronteira Argentina", stop: "Uruguaiana (Argentina)", sign: "Uruguaiana", wp: 7, build: buildUruguaiana,
    info: "Gaucho cattle country on the Uruguay River, tied to Paso de los Libres, Argentina by the 1945 international bridge, one of the busiest land border crossings in South America."
  },
  {
    name: "Sant'Ana do Livramento · Fronteira Uruguai", stop: "Livramento (Uruguay)", sign: "Livramento", wp: 9, build: buildLivramento,
    info: "Twin city with Rivera, Uruguay. A single open square, the Parque Internacional, joins the two countries with no fence at all, the friendliest border in the world."
  }
];

const lampLights = [], lampHeads = [];
for (const c of cities) {
  c.u = nearestU(waypoints[c.wp]);
  const p = curve.getPointAt(c.u), t = curve.getTangentAt(c.u);
  const n = new THREE.Vector3(t.z, 0, -t.x);
  c.station = p.clone();
  const g = new THREE.Group();
  g.position.copy(p).addScaledVector(n, 46);
  g.lookAt(p.x, 0, p.z);
  scene.add(g);
  c.build(g);
  c.origin = g.position;
  const plat = new THREE.Mesh(new THREE.BoxGeometry(5, 0.8, 46), lam(0xb7ac97));
  plat.position.copy(p).addScaledVector(n, 6.8);
  plat.position.y = 0.4;
  plat.rotation.y = Math.atan2(t.x, t.z);
  scene.add(plat);
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(13, 3.2), textPanel(c.sign));
  sign.position.copy(p).addScaledVector(n, 11).addScaledVector(t, 12);
  sign.position.y = 4.6;
  sign.rotation.y = Math.atan2(-n.x, -n.z);
  scene.add(sign);
  for (const s of [-5.2, 5.2]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 4.4, 8), lam(0x4a4038));
    post.position.copy(sign.position).addScaledVector(t, s);
    post.position.y = 2.2;
    scene.add(post);
  }
  for (const s of [-16, 16]) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 6.4, 8), lam(0x3a3f45));
    pole.position.copy(p).addScaledVector(n, 9).addScaledVector(t, s);
    pole.position.y = 3.2;
    scene.add(pole);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.42, 10, 8), new THREE.MeshBasicMaterial({ color: 0x4a4238 }));
    head.position.copy(pole.position);
    head.position.y = 6.6;
    scene.add(head);
    lampHeads.push(head);
  }
  const light = new THREE.PointLight(0xffb066, 0, 110, 1.6);
  light.position.copy(p).addScaledVector(n, 9);
  light.position.y = 6.2;
  scene.add(light);
  lampLights.push(light);
}

let planted = 0, tries = 0;
while (planted < 170 && tries < 4000) {
  tries++;
  const serra = planted < 100;
  const x = serra ? 60 + rnd() * 950 : -1500 + rnd() * 3000;
  const z = serra ? -1350 + rnd() * 950 : -1500 + rnd() * 3000;
  if (Math.hypot(x, z) > 2900) continue;
  if (trackDist(x, z) < 30) continue;
  if (cities.some(c => Math.hypot(c.origin.x - x, c.origin.z - z) < 110)) continue;
  tree(scene, x, z, 0.8 + rnd() * 0.7);
  planted++;
}

let made = 0;
tries = 0;
while (made < 12 && tries < 500) {
  tries++;
  const serra = made < 8;
  const x = serra ? 150 + rnd() * 950 : -900 + rnd() * 1800;
  const z = serra ? -1550 + rnd() * 900 : 1000 + rnd() * 700;
  const r = 110 + rnd() * 150, h = 55 + rnd() * 90;
  if (Math.hypot(x, z) > 2800) continue;
  if (trackDist(x, z) < r + 60) continue;
  if (cities.some(c => Math.hypot(c.origin.x - x, c.origin.z - z) < r + 130)) continue;
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, 7), lam(0x5e7f52));
  m.position.set(x, h / 2 - 2, z);
  scene.add(m);
  made++;
}

function wheel(g, x, y, z, r) {
  const w = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.35, 12), lam(0x1a1d20));
  w.rotation.z = Math.PI / 2;
  w.position.set(x, y, z);
  g.add(w);
}

function makeLoco() {
  const g = new THREE.Group();
  box(g, 2.6, 0.5, 8, 0x14171a, 0, 0.95, 0);
  const boiler = new THREE.Mesh(new THREE.CylinderGeometry(1.05, 1.05, 4.8, 16), lam(0x1d2126));
  boiler.rotation.x = Math.PI / 2;
  boiler.position.set(0, 2.05, 1.2);
  g.add(boiler);
  cyl(g, 0.34, 1.4, 0x101214, 0, 3.4, 3.1);
  cone(g, 0.55, 0.5, 0x101214, 0, 4.3, 3.1, 12);
  sph(g, 0.5, 0xb08d3f, 0, 3.2, 0.4);
  box(g, 2.5, 2.3, 2.3, 0x7c1f1f, 0, 2.45, -2.3);
  box(g, 2.9, 0.25, 2.7, 0x2a1114, 0, 3.72, -2.3);
  const cc = new THREE.Mesh(new THREE.ConeGeometry(1.5, 1.4, 4), lam(0x8a2a1e));
  cc.rotation.x = Math.PI / 2;
  cc.position.set(0, 0.8, 4.5);
  g.add(cc);
  const lampMesh = new THREE.Mesh(new THREE.SphereGeometry(0.3, 10, 8), new THREE.MeshBasicMaterial({ color: 0xfff3cf }));
  lampMesh.position.set(0, 3.1, 4.1);
  g.add(lampMesh);
  for (const z of [-2.6, -0.9, 0.8, 2.5]) {
    wheel(g, -1.15, 0.55, z, 0.55);
    wheel(g, 1.15, 0.55, z, 0.55);
  }
  return g;
}

function makeCoach(color) {
  const g = new THREE.Group();
  box(g, 2.5, 2.2, 8, color, 0, 2, 0);
  const wb = new THREE.Mesh(new THREE.BoxGeometry(2.56, 0.85, 6.8), new THREE.MeshBasicMaterial({ color: 0xffe0a0 }));
  wb.position.set(0, 2.35, 0);
  g.add(wb);
  box(g, 2.8, 0.25, 8.4, 0x23262a, 0, 3.22, 0);
  box(g, 2.3, 0.5, 7.4, 0x17191c, 0, 0.85, 0);
  for (const z of [-2.7, 2.7]) {
    wheel(g, -1.1, 0.45, z, 0.45);
    wheel(g, 1.1, 0.45, z, 0.45);
  }
  return g;
}

const loco = makeLoco();
const cars = [loco, makeCoach(0x25502e), makeCoach(0x6e2430)];
for (const c of cars) scene.add(c);

const head = new THREE.SpotLight(0xfff2cc, 0, 340, 0.5, 0.55, 1.1);
head.position.set(0, 3.1, 4.2);
const headTarget = new THREE.Object3D();
headTarget.position.set(0, 1.2, 90);
loco.add(head, headTarget);
head.target = headTarget;

const CRUISE = 26, SPACING = 10.5;
let dist = cities[0].u * trackLen - 200;
let speed = 0, moving = true;

function placeCar(obj, d) {
  const u = ((d % trackLen) + trackLen) % trackLen / trackLen;
  const p = curve.getPointAt(u), t = curve.getTangentAt(u);
  obj.position.set(p.x, 0.34, p.z);
  obj.rotation.y = Math.atan2(t.x, t.z);
}

function updateTrain(dt) {
  const target = moving && mode === "ride" ? CRUISE : 0;
  speed += (target - speed) * Math.min(1, dt * 0.6);
  dist += speed * dt;
  placeCar(loco, dist);
  placeCar(cars[1], dist - SPACING);
  placeCar(cars[2], dist - SPACING * 2);
}

const puffs = [];
const puffGeo = new THREE.SphereGeometry(0.55, 8, 6);
for (let i = 0; i < 26; i++) {
  const m = new THREE.Mesh(puffGeo, new THREE.MeshBasicMaterial({ color: 0xd6dade, transparent: true, opacity: 0 }));
  m.visible = false;
  scene.add(m);
  puffs.push({ m, age: 9, vel: new THREE.Vector3() });
}
let puffTimer = 0, puffIdx = 0;

function updateSmoke(dt) {
  puffTimer -= dt;
  if (speed > 3 && puffTimer <= 0) {
    puffTimer = 0.16;
    loco.updateMatrixWorld(true);
    const p = puffs[puffIdx++ % puffs.length];
    p.age = 0;
    p.m.visible = true;
    p.m.position.copy(loco.localToWorld(new THREE.Vector3(0, 4.4, 2.8)));
    p.vel.set((Math.random() - 0.5) * 3 - 2, 11 + Math.random() * 4, (Math.random() - 0.5) * 3);
    p.m.scale.setScalar(1);
  }
  for (const p of puffs) {
    if (p.age > 1.8) {
      p.m.visible = false;
      continue;
    }
    p.age += dt;
    p.m.position.addScaledVector(p.vel, dt);
    p.m.scale.setScalar(1 + p.age * 2);
    p.m.material.opacity = Math.max(0, 0.26 * (1 - p.age / 1.8));
  }
}

const starGeo = new THREE.BufferGeometry();
const sp = new Float32Array(900 * 3);
for (let i = 0; i < 900; i++) {
  const a = rnd() * Math.PI * 2, e = Math.asin(rnd());
  sp[i * 3] = Math.cos(a) * Math.cos(e) * 3000;
  sp[i * 3 + 1] = 150 + Math.sin(e) * 2700;
  sp[i * 3 + 2] = Math.sin(a) * Math.cos(e) * 3000;
}
starGeo.setAttribute("position", new THREE.BufferAttribute(sp, 3));
const starMat = new THREE.PointsMaterial({ color: 0xdfe8ff, size: 2.2, sizeAttenuation: false, transparent: true, opacity: 0, fog: false });
scene.add(new THREE.Points(starGeo, starMat));

const DROPS = 900;
const rainGeo = new THREE.BufferGeometry();
const rp = new Float32Array(DROPS * 6);
for (let i = 0; i < DROPS; i++) {
  const x = (Math.random() - 0.5) * 260, y = Math.random() * 150, z = (Math.random() - 0.5) * 260;
  rp.set([x, y, z, x + 0.4, y + 1.8, z], i * 6);
}
rainGeo.setAttribute("position", new THREE.BufferAttribute(rp, 3));
const rain = new THREE.LineSegments(rainGeo, new THREE.LineBasicMaterial({ color: 0xa9c3d4, transparent: true, opacity: 0 }));
rain.frustumCulled = false;
scene.add(rain);

let raining = false, rainLevel = 0, weatherTimer = 26;

function updateWeather(dt) {
  weatherTimer -= dt;
  if (weatherTimer <= 0) {
    if (raining) {
      raining = false;
      weatherTimer = 35 + Math.random() * 40;
    } else if (Math.random() < 0.6) {
      raining = true;
      weatherTimer = 22 + Math.random() * 25;
    } else {
      weatherTimer = 12;
    }
  }
  rainLevel += ((raining ? 1 : 0) - rainLevel) * Math.min(1, dt * 0.5);
  rain.material.opacity = 0.55 * rainLevel;
  rain.visible = rainLevel > 0.02;
  if (!rain.visible) return;
  const a = rainGeo.attributes.position.array;
  const fall = 190 * dt, drift = 14 * dt;
  for (let i = 0; i < DROPS; i++) {
    a[i * 6 + 1] -= fall;
    a[i * 6 + 4] -= fall;
    a[i * 6] -= drift;
    a[i * 6 + 3] -= drift;
    if (a[i * 6 + 1] < 0) {
      const x = (Math.random() - 0.5) * 260, y = 130 + Math.random() * 30, z = (Math.random() - 0.5) * 260;
      a.set([x, y, z, x + 0.4, y + 1.8, z], i * 6);
    }
  }
  rainGeo.attributes.position.needsUpdate = true;
  rain.position.set(camera.position.x, 0, camera.position.z);
}

const DAY = 240;
const tod0 = params.has("tod") ? parseFloat(params.get("tod")) : 0.33;
let elapsed = 0, dayFrac = tod0, daylight = 1;
const colDay = new THREE.Color(0x8dc6ea), colNight = new THREE.Color(0x0b1026);
const colDusk = new THREE.Color(0xe8925f), colRain = new THREE.Color(0x778089);
const duskSun = new THREE.Color(0xff9a5a);
const sky = new THREE.Color();
scene.background = sky;

function updateSky() {
  dayFrac = (tod0 + elapsed / DAY) % 1;
  const s = Math.sin((dayFrac - 0.25) * Math.PI * 2);
  daylight = THREE.MathUtils.clamp(s * 1.25, 0, 1);
  const duskAmt = Math.max(0, 1 - Math.abs(s) / 0.32);
  sky.copy(colNight).lerp(colDay, daylight);
  sky.lerp(colDusk, duskAmt * 0.55);
  sky.lerp(colRain, rainLevel * (0.15 + 0.6 * daylight));
  scene.fog.color.copy(sky);
  scene.fog.far = 1900 - rainLevel * 900;
  const ang = (dayFrac - 0.25) * Math.PI * 2;
  sun.position.set(Math.cos(ang) * 900, Math.sin(ang) * 900, 350);
  sun.intensity = daylight * 2.4 * (1 - 0.55 * rainLevel);
  sun.color.setHex(0xffffff).lerp(duskSun, duskAmt * 0.8);
  hemi.intensity = 0.18 + daylight * (1 - 0.4 * rainLevel);
  starMat.opacity = (1 - daylight) ** 2 * (1 - rainLevel) * 0.9;
  const night = 1 - daylight;
  for (const l of lampLights) l.intensity = night * 60;
  for (const h of lampHeads) h.material.color.setHex(night > 0.4 ? 0xffd9a0 : 0x4a4238);
  head.intensity = Math.min(1, night + rainLevel * 0.5) * 950;
}

const keys = new Set();
let mode = "ride";
const walker = { pos: new THREE.Vector3(), yaw: 0, pitch: 0 };

function lockPointer() {
  try {
    const p = canvas.requestPointerLock();
    if (p && p.catch) p.catch(() => {});
  } catch (err) {
    void err;
  }
}

function toggleWalk() {
  if (mode === "ride") {
    if (Math.abs(speed) > 0.5) return;
    moving = false;
    const u = ((dist % trackLen) + trackLen) % trackLen / trackLen;
    const t = curve.getTangentAt(u);
    const n = new THREE.Vector3(t.z, 0, -t.x);
    walker.pos.copy(loco.position).addScaledVector(n, 7);
    walker.pos.y = 2;
    walker.yaw = Math.atan2(-n.x, -n.z);
    walker.pitch = 0;
    mode = "walk";
    lockPointer();
  } else if (walker.pos.distanceTo(loco.position) < 15) {
    mode = "ride";
    document.exitPointerLock();
  }
}

addEventListener("keydown", e => {
  if (e.code === "Space") {
    e.preventDefault();
    if (mode === "ride") moving = !moving;
  }
  if (e.code === "KeyE") toggleWalk();
  keys.add(e.code);
});
addEventListener("keyup", e => keys.delete(e.code));
addEventListener("mousemove", e => {
  if (mode === "walk" && document.pointerLockElement === canvas) {
    walker.yaw -= e.movementX * 0.0022;
    walker.pitch = THREE.MathUtils.clamp(walker.pitch - e.movementY * 0.0022, -1.35, 1.35);
  }
});
canvas.addEventListener("click", () => {
  if (mode === "walk" && !document.pointerLockElement) lockPointer();
});

const euler = new THREE.Euler(0, 0, 0, "YXZ");

function updateWalk(dt) {
  const run = keys.has("ShiftLeft") || keys.has("ShiftRight");
  const f = new THREE.Vector3(-Math.sin(walker.yaw), 0, -Math.cos(walker.yaw));
  const r = new THREE.Vector3(-f.z, 0, f.x);
  const mv = new THREE.Vector3();
  if (keys.has("KeyW")) mv.add(f);
  if (keys.has("KeyS")) mv.sub(f);
  if (keys.has("KeyD")) mv.add(r);
  if (keys.has("KeyA")) mv.sub(r);
  if (mv.lengthSq() > 0) {
    mv.normalize();
    walker.pos.addScaledVector(mv, (run ? 26 : 12) * dt);
    if (walker.pos.length() > 3100) walker.pos.setLength(3100);
    walker.pos.y = 2;
  }
  camera.position.copy(walker.pos);
  euler.set(walker.pitch, walker.yaw, 0);
  camera.quaternion.setFromEuler(euler);
}

const camLook = new THREE.Vector3(900, 4, 0);

function updateCamera(dt) {
  if (mode === "walk") {
    updateWalk(dt);
    return;
  }
  const u = ((dist % trackLen) + trackLen) % trackLen / trackLen;
  const p = curve.getPointAt(u), t = curve.getTangentAt(u);
  const desired = p.clone().addScaledVector(t, -17);
  desired.x += t.z * 5.5;
  desired.z -= t.x * 5.5;
  desired.y = 8.2;
  const k = 1 - Math.exp(-3.2 * dt);
  camera.position.lerp(desired, k);
  const look = p.clone().addScaledVector(t, 26);
  look.y = 3.4;
  camLook.lerp(look, k);
  camera.lookAt(camLook);
}

const cityName = document.getElementById("cityName");
const cityInfo = document.getElementById("cityInfo");
const locPanel = document.getElementById("loc");
const clockEl = document.getElementById("clock");
const weatherEl = document.getElementById("weather");
const nextEl = document.getElementById("next");
const hintEl = document.getElementById("hint");
let currentCity = null;

function setHtml(el, s) {
  if (el.dataset.s !== s) {
    el.dataset.s = s;
    el.innerHTML = s;
  }
}

function updateHud() {
  const ref = mode === "walk" ? walker.pos : loco.position;
  let near = null;
  for (const c of cities) if (ref.distanceTo(c.station) < 150) near = c;
  if (near !== currentCity) {
    currentCity = near;
    if (near) {
      cityName.textContent = near.name;
      cityInfo.textContent = near.info;
      locPanel.classList.add("show");
    } else {
      locPanel.classList.remove("show");
    }
  }
  const hh = Math.floor(dayFrac * 24), mm = Math.floor((dayFrac * 24 % 1) * 60);
  setHtml(clockEl, String(hh).padStart(2, "0") + ":" + String(mm).padStart(2, "0"));
  setHtml(weatherEl, raining ? "Rain over the pampa" : "Clear skies");
  if (mode === "ride") {
    const dNow = ((dist % trackLen) + trackLen) % trackLen;
    let bestD = Infinity, bestC = cities[0];
    for (const c of cities) {
      let d = c.u * trackLen - dNow;
      if (d < 60) d += trackLen;
      if (d < bestD) {
        bestD = d;
        bestC = c;
      }
    }
    setHtml(nextEl, "Next stop <b>" + bestC.stop + "</b> · " + (bestD * 0.15).toFixed(1) + " km");
    setHtml(hintEl, moving
      ? "<b>SPACE</b> stop the train"
      : "<b>SPACE</b> depart · <b>E</b> step off and walk");
  } else {
    setHtml(nextEl, currentCity ? "Walking in <b>" + currentCity.stop + "</b>" : "Walking the pampa");
    setHtml(hintEl, walker.pos.distanceTo(loco.position) < 15
      ? "<b>E</b> board the train · <b>WASD</b> walk"
      : "<b>WASD</b> walk · <b>SHIFT</b> run · click to look · press <b>E</b> back at the train");
  }
}

const overlay = document.getElementById("overlay");
overlay.addEventListener("click", () => {
  overlay.style.display = "none";
});
if (params.has("auto")) overlay.style.display = "none";

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  elapsed += dt;
  updateTrain(dt);
  updateCamera(dt);
  updateWeather(dt);
  updateSky();
  updateSmoke(dt);
  updateHud();
  renderer.render(scene, camera);
}
animate();
