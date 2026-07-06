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
  while (x.measureText(text).width > 455 && size > 20) {
    size -= 2;
    x.font = "bold " + size + "px Georgia";
  }
  x.textAlign = "center";
  x.textBaseline = "middle";
  x.fillText(text, 256, 66);
  return new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide });
}

function label(g, text, x, z) {
  cyl(g, 0.12, 6.6, 0x4a4038, x, 3.3, z);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(11.5, 2.9), textPanel(text));
  m.position.set(x, 7.6, z);
  g.add(m);
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

function skyGrad(x, w, h, top, bot) {
  const gr = x.createLinearGradient(0, 0, 0, h);
  gr.addColorStop(0, top);
  gr.addColorStop(1, bot);
  x.fillStyle = gr;
  x.fillRect(0, 0, w, h);
}

function paintGuaiba(x, w, h) {
  skyGrad(x, w, h, "#ffd688", "#ff7a44");
  x.fillStyle = "#fff2c8";
  x.beginPath();
  x.arc(w * 0.72, h * 0.4, h * 0.13, 0, 7);
  x.fill();
  x.fillStyle = "#6a4f92";
  x.fillRect(0, h * 0.64, w, h * 0.36);
  x.fillStyle = "rgba(255,214,140,.45)";
  x.fillRect(w * 0.5, h * 0.64, w * 0.35, h * 0.36);
  x.fillStyle = "#33283a";
  x.fillRect(w * 0.16, h * 0.4, w * 0.18, h * 0.26);
  x.fillRect(w * 0.31, h * 0.16, w * 0.035, h * 0.5);
}

function paintGramado(x, w, h) {
  skyGrad(x, w, h, "#c4e6f4", "#eaf6ea");
  x.fillStyle = "#5c8a4a";
  x.beginPath();
  x.moveTo(0, h * 0.7);
  x.quadraticCurveTo(w * 0.5, h * 0.5, w, h * 0.72);
  x.lineTo(w, h);
  x.lineTo(0, h);
  x.fill();
  x.fillStyle = "#2f5b34";
  for (const cx of [w * 0.12, w * 0.85, w * 0.7]) {
    x.beginPath();
    x.moveTo(cx, h * 0.4);
    x.lineTo(cx - h * 0.09, h * 0.72);
    x.lineTo(cx + h * 0.09, h * 0.72);
    x.fill();
  }
  x.fillStyle = "#efe6d0";
  x.fillRect(w * 0.36, h * 0.5, w * 0.28, h * 0.24);
  x.fillStyle = "#8a3626";
  x.beginPath();
  x.moveTo(w * 0.34, h * 0.5);
  x.lineTo(w * 0.5, h * 0.34);
  x.lineTo(w * 0.66, h * 0.5);
  x.fill();
  for (const c of ["#7c86c9", "#c96b8e", "#8f7cc9"]) {
    x.fillStyle = c;
    for (let i = 0; i < 4; i++) {
      x.beginPath();
      x.arc(w * 0.08 + Math.random() * w * 0.84, h * 0.8 + Math.random() * h * 0.16, h * 0.028, 0, 7);
      x.fill();
    }
  }
}

function paintCaracol(x, w, h) {
  skyGrad(x, w, h, "#bcd8ec", "#dfeee4");
  x.fillStyle = "#3f5f3a";
  x.fillRect(0, h * 0.18, w * 0.34, h * 0.82);
  x.fillStyle = "#4a6b40";
  x.fillRect(w * 0.66, h * 0.18, w * 0.34, h * 0.82);
  x.fillStyle = "#eef7fb";
  x.fillRect(w * 0.42, h * 0.14, w * 0.16, h * 0.64);
  x.fillStyle = "rgba(255,255,255,.6)";
  x.fillRect(w * 0.37, h * 0.7, w * 0.26, h * 0.14);
  x.fillStyle = "#4f86a0";
  x.fillRect(0, h * 0.82, w, h * 0.18);
}

function paintMissoes(x, w, h) {
  skyGrad(x, w, h, "#e88a4a", "#7a3a52");
  x.fillStyle = "#94502f";
  x.fillRect(w * 0.1, h * 0.26, w * 0.8, h * 0.52);
  x.fillStyle = "#e88a4a";
  for (const cx of [w * 0.28, w * 0.5, w * 0.72]) {
    x.beginPath();
    x.moveTo(cx - w * 0.07, h * 0.78);
    x.lineTo(cx - w * 0.07, h * 0.46);
    x.arc(cx, h * 0.46, w * 0.07, Math.PI, 0);
    x.lineTo(cx + w * 0.07, h * 0.78);
    x.fill();
  }
  x.fillStyle = "#6b3a22";
  x.fillRect(0, h * 0.78, w, h * 0.22);
}

function paintPampa(x, w, h) {
  skyGrad(x, w, h, "#8fc6ea", "#cfe6f0");
  x.fillStyle = "rgba(255,255,255,.85)";
  for (const cx of [w * 0.2, w * 0.55, w * 0.8]) {
    x.beginPath();
    x.arc(cx, h * 0.22, h * 0.08, 0, 7);
    x.fill();
  }
  x.fillStyle = "#6f9450";
  x.fillRect(0, h * 0.6, w, h * 0.4);
  x.strokeStyle = "#6b5335";
  x.lineWidth = h * 0.012;
  x.beginPath();
  x.moveTo(0, h * 0.68);
  x.lineTo(w, h * 0.66);
  x.stroke();
  for (let i = 0; i < 7; i++) {
    const px = w * 0.05 + i * w * 0.15;
    x.beginPath();
    x.moveTo(px, h * 0.6);
    x.lineTo(px, h * 0.72);
    x.stroke();
  }
  x.fillStyle = "#3a2b22";
  for (const cx of [w * 0.3, w * 0.55, w * 0.75]) {
    x.fillRect(cx, h * 0.74, w * 0.09, h * 0.08);
    x.fillRect(cx + w * 0.06, h * 0.72, w * 0.035, h * 0.05);
  }
}

function paintFarol(x, w, h) {
  skyGrad(x, w, h, "#bcd8ec", "#eef2ea");
  x.fillStyle = "#4f86a0";
  x.fillRect(0, h * 0.72, w, h * 0.28);
  x.fillStyle = "#f2f0ea";
  x.fillRect(w * 0.44, h * 0.2, w * 0.12, h * 0.55);
  x.fillStyle = "#c23b2e";
  x.fillRect(w * 0.44, h * 0.32, w * 0.12, h * 0.09);
  x.fillRect(w * 0.44, h * 0.5, w * 0.12, h * 0.09);
  x.fillStyle = "#fff3cf";
  x.fillRect(w * 0.46, h * 0.12, w * 0.08, h * 0.08);
  x.fillStyle = "#3a3630";
  x.fillRect(w * 0.43, h * 0.1, w * 0.14, h * 0.03);
}

function paintTorres(x, w, h) {
  skyGrad(x, w, h, "#9ecdec", "#dfeef0");
  x.fillStyle = "#2e6d94";
  x.fillRect(0, h * 0.6, w, h * 0.4);
  x.fillStyle = "#3e3a36";
  x.fillRect(w * 0.05, h * 0.34, w * 0.28, h * 0.4);
  x.fillRect(w * 0.68, h * 0.4, w * 0.27, h * 0.34);
  x.fillStyle = "#4a6b40";
  x.fillRect(w * 0.05, h * 0.31, w * 0.28, h * 0.05);
  x.fillRect(w * 0.68, h * 0.37, w * 0.27, h * 0.05);
  x.fillStyle = "#f2f0ea";
  x.fillRect(w * 0.14, h * 0.16, w * 0.06, h * 0.18);
  x.fillStyle = "#c23b2e";
  x.fillRect(w * 0.14, h * 0.22, w * 0.06, h * 0.05);
  x.fillStyle = "#fff3cf";
  x.beginPath();
  x.arc(w * 0.17, h * 0.14, h * 0.03, 0, 7);
  x.fill();
}

function picture(g, w, h, x, y, z, draw, ry = 0) {
  const nx = Math.sin(ry), nz = Math.cos(ry);
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = Math.round(256 * h / w);
  draw(c.getContext("2d"), c.width, c.height);
  box(g, w + 0.7, h + 0.7, 0.3, 0x4a3728, x - nx * 0.18, y, z - nz * 0.18, ry);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) }));
  m.position.set(x, y, z);
  m.rotation.y = ry;
  g.add(m);
}

const COLLISION_CELL = 32;
const collisionGrid = new Map();

function collisionCells(minX, maxX, minZ, maxZ, fn) {
  const x0 = Math.floor(minX / COLLISION_CELL), x1 = Math.floor(maxX / COLLISION_CELL);
  const z0 = Math.floor(minZ / COLLISION_CELL), z1 = Math.floor(maxZ / COLLISION_CELL);
  for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) fn(x + ":" + z);
}

function registerCollider(c) {
  collisionCells(c.minX, c.maxX, c.minZ, c.maxZ, key => {
    if (!collisionGrid.has(key)) collisionGrid.set(key, []);
    collisionGrid.get(key).push(c);
  });
}

function registerCircleCollider(x, z, radius) {
  registerCollider({ type: "circle", x, z, radius, minX: x - radius, maxX: x + radius, minZ: z - radius, maxZ: z + radius });
}

function registerGroupColliders(g) {
  g.updateMatrixWorld(true);
  g.traverse(o => {
    if (!o.isMesh || o.userData.walkThrough || !o.geometry) return;
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    const b = o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld);
    if (b.max.y < 1.05 || b.max.x - b.min.x > 190 || b.max.z - b.min.z > 190) return;
    registerCollider({ type: "box", minX: b.min.x, maxX: b.max.x, minZ: b.min.z, maxZ: b.max.z });
  });
}

function blockedAt(v, radius = 0.72) {
  const found = new Set();
  let blocked = false;
  collisionCells(v.x - radius, v.x + radius, v.z - radius, v.z + radius, key => {
    if (blocked) return;
    for (const c of collisionGrid.get(key) || []) {
      if (found.has(c)) continue;
      found.add(c);
      if (c.type === "circle") {
        const rr = radius + c.radius;
        if ((v.x - c.x) ** 2 + (v.z - c.z) ** 2 < rr * rr) blocked = true;
      } else if (v.x + radius > c.minX && v.x - radius < c.maxX && v.z + radius > c.minZ && v.z - radius < c.maxZ) {
        blocked = true;
      }
      if (blocked) break;
    }
  });
  return blocked;
}

function mediaCaption(text) {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 72;
  const x = c.getContext("2d");
  x.fillStyle = "#101c26";
  x.fillRect(0, 0, c.width, c.height);
  x.fillStyle = "#f4ead6";
  x.font = "bold 25px Georgia";
  x.textAlign = "center";
  x.textBaseline = "middle";
  x.fillText(text, c.width / 2, c.height / 2, 960);
  return new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide });
}

const textureLoader = new THREE.TextureLoader();
const videoBoards = [];
const videoLayer = document.getElementById("videoLayer");

function addPhotoBillboard(c, p, t, n) {
  const g = new THREE.Group();
  g.position.copy(p).addScaledVector(n, 28).addScaledVector(t, c.wp % 2 ? 24 : -24);
  g.lookAt(p.x, 6, p.z);
  scene.add(g);
  for (const x of [-7.1, 7.1]) cyl(g, 0.22, 7, 0x4a4038, x, 3.5, 0);
  box(g, 16, 10, 0.45, 0x4a3728, 0, 8.7, 0);
  const photo = new THREE.Mesh(new THREE.PlaneGeometry(15, 8.5), new THREE.MeshBasicMaterial({ color: 0x182127, side: THREE.DoubleSide }));
  photo.position.set(0, 9, 0.24);
  g.add(photo);
  textureLoader.load(c.photo, texture => {
    if ("colorSpace" in texture && THREE.SRGBColorSpace) texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    photo.material.map = texture;
    photo.material.color.setHex(0xffffff);
    photo.material.needsUpdate = true;
  });
  const credit = new THREE.Mesh(new THREE.PlaneGeometry(15, 1.05), mediaCaption(c.photoCredit));
  credit.position.set(0, 4.2, 0.25);
  g.add(credit);
  registerGroupColliders(g);
}

function addVideoBillboard(c, p, t, n) {
  const g = new THREE.Group();
  g.position.copy(p).addScaledVector(n, -38).addScaledVector(t, c.wp % 2 ? -24 : 24);
  g.lookAt(p.x, 6, p.z);
  scene.add(g);
  for (const x of [-8.6, 8.6]) cyl(g, 0.24, 8, 0x3a3f45, x, 4, 0);
  box(g, 19, 11, 0.5, 0x20272b, 0, 9.5, 0);
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(18, 10.1), new THREE.MeshBasicMaterial({ color: 0x080b0d, side: THREE.DoubleSide }));
  screen.position.set(0, 9.7, 0.28);
  g.add(screen);
  const el = document.createElement("div");
  el.className = "videoBillboard";
  const iframe = document.createElement("iframe");
  iframe.allow = "autoplay; encrypted-media; picture-in-picture";
  iframe.title = c.stop + " video";
  const title = document.createElement("div");
  title.textContent = c.stop + " · YouTube";
  el.append(iframe, title);
  videoLayer.append(el);
  videoBoards.push({ c, screen, el, iframe, loaded: false });
  registerGroupColliders(g);
}

function buildPortoAlegre(g) {
  flat(g, 560, 300, 0x3e6e8c, 0, 0.04, -260);
  box(g, 48, 16, 15, 0xe6e0d1, 0, 8, -28);
  box(g, 50, 1.6, 16.5, 0xb9b2a2, 0, 16.8, -28);
  box(g, 48.3, 3, 15.2, 0x2f3a42, 0, 9.5, -28);
  for (const x of [-18, -6, 6, 18]) box(g, 3.4, 7, 0.5, 0x21313c, x, 8, -20.4);
  cyl(g, 2.8, 62, 0xa14b32, 19, 31, -28);
  cyl(g, 3.2, 2.6, 0x7c3423, 19, 62.3, -28);
  label(g, "Usina do Gasômetro", 0, -14);
  box(g, 34, 11, 20, 0xf0e3c8, -62, 5.5, -30);
  box(g, 35, 1.4, 21, 0xd9c9a4, -62, 11.9, -30);
  for (const x of [-77, -47]) for (const z of [-39, -21]) box(g, 4.5, 14, 4.5, 0xe6d5ae, x, 7, z);
  box(g, 7, 15, 7, 0xe6d5ae, -62, 7.5, -30);
  cone(g, 4.8, 3.6, 0x8a4a32, -62, 16.8, -30, 4, Math.PI / 4);
  for (const x of [-72, -67, -57, -52]) box(g, 3, 5.6, 0.5, 0x5c4630, x, 3.4, -19.7);
  label(g, "Mercado Público", -62, -14);
  box(g, 20, 24, 14, 0xf2f0ea, 58, 12, -34);
  for (const [y, rz] of [[7, 0.12], [13.5, -0.1], [20, 0.13]]) {
    const r = box(g, 25, 2.4, 3, 0xe9e7e0, 58, y, -26.6);
    r.rotation.z = rz;
  }
  label(g, "Fundação Iberê Camargo", 58, -18);
  flat(g, 74, 44, 0x5c8a4a, -58, 0.05, -70);
  flat(g, 4, 44, 0xc9b189, -58, 0.07, -70);
  flat(g, 60, 4, 0xc9b189, -58, 0.07, -66);
  cyl(g, 3.6, 1.4, 0x8b8b85, -58, 0.7, -60);
  cyl(g, 2.8, 0.5, 0x69a3c0, -58, 1.6, -60);
  cyl(g, 0.5, 4.6, 0x74572e, -58, 3.4, -60);
  sph(g, 0.8, 0x74572e, -58, 6.2, -60);
  box(g, 1.7, 11, 1.7, 0xd8d3c4, -76, 5.5, -82);
  box(g, 1.7, 11, 1.7, 0xd8d3c4, -69, 5.5, -82);
  box(g, 9, 1.7, 2.2, 0xd8d3c4, -72.5, 11.6, -82);
  tree(g, -84, -60, 1.1);
  tree(g, -34, -58, 1);
  tree(g, -82, -84, 0.9);
  tree(g, -36, -84, 1.2);
  label(g, "Parque Farroupilha", -58, -46);
  box(g, 10, 46, 10, 0x8e99a5, 18, 23, -72);
  box(g, 9, 58, 9, 0x7d8794, 34, 29, -80);
  box(g, 8, 36, 8, 0x9fa8b2, 48, 18, -68);
  box(g, 7, 28, 7, 0xa8b0b8, 6, 14, -80);
  box(g, 8, 40, 8, 0x97a1ac, 62, 20, -78);
  flat(g, 170, 10, 0xc9b189, 0, 0.06, -104);
  for (let i = 0; i < 7; i++) {
    const x = -66 + i * 22;
    cyl(g, 0.14, 3.2, 0x6d5b43, x, 1.6, -101);
    cone(g, 2.7, 1.5, [0xd96b43, 0xe0b13e, 0x5b8fb0][i % 3], x, 3.6, -101, 8);
  }
  for (const x of [-55, -33, -11, 11, 33, 55]) box(g, 3, 0.5, 1, 0x6d5b43, x, 0.6, -99);
  flat(g, 16, 9, 0x3e7e58, 78, 0.07, -102);
  box(g, 0.2, 1, 9, 0xe8e8e2, 78, 1, -102);
  label(g, "Orla do Guaíba", 0, -94);
  cyl(g, 1.3, 2.4, 0x8b8b85, -20, 1.2, 4);
  cyl(g, 0.45, 3.2, 0x74572e, -20, 4, 4);
  sph(g, 0.55, 0x74572e, -20, 6.1, 4);
  picture(g, 15, 8, 0, 9.5, -20.1, paintGuaiba);
}

function buildGramado(g) {
  function chalet(x, z, s, wall, roof) {
    box(g, 6 * s, 3 * s, 5 * s, wall, x, 1.5 * s, z);
    cone(g, 4.4 * s, 2.8 * s, roof, x, 4.4 * s, z, 4, Math.PI / 4);
  }
  box(g, 13, 12, 26, 0x8f8b82, 0, 6, -32);
  box(g, 7.5, 26, 7.5, 0x95918a, 0, 13, -16);
  cone(g, 5.4, 17, 0x6e6a62, 0, 34.5, -16, 8);
  box(g, 0.4, 3, 0.4, 0xd9d9d2, 0, 44, -16);
  box(g, 1.8, 0.4, 0.4, 0xd9d9d2, 0, 44.5, -16);
  box(g, 3.2, 6.5, 0.7, 0x3a3630, 0, 3.2, -12.1);
  for (const z of [-26, -32, -38]) for (const x of [-6.8, 6.8]) box(g, 0.5, 4.5, 2, 0x3f6f9f, x, 7, z);
  label(g, "Igreja Matriz São Pedro", 0, -6);
  box(g, 26, 13, 15, 0x9c3a2c, -48, 6.5, -24);
  box(g, 27, 2.6, 4, 0xefe6d0, -48, 10.5, -16.2);
  flat(g, 5, 14, 0x9c2430, -48, 0.06, -9);
  cyl(g, 1, 1, 0x6d6a63, -54, 0.5, -12);
  cone(g, 0.5, 1.9, 0xc9a437, -54, 1.9, -12, 10);
  cyl(g, 1, 1, 0x6d6a63, -42, 0.5, -12);
  cone(g, 0.5, 1.9, 0xc9a437, -42, 1.9, -12, 10);
  label(g, "Palácio dos Festivais", -48, -4);
  flat(g, 42, 32, 0x69945a, -56, 0.05, -66);
  for (let i = 0; i < 7; i++) {
    const x = -70 + (i % 4) * 9, z = -60 - Math.floor(i / 4) * 12;
    box(g, 2.6, 2, 2.2, [0xe9dcc3, 0xc9a98a, 0xb0b6c9][i % 3], x, 1, z);
    cone(g, 2, 1.6, 0x6e2f22, x, 2.8, z, 4, Math.PI / 4);
  }
  box(g, 1.4, 3.4, 1.4, 0xd8d3c4, -46, 1.7, -72);
  cone(g, 1.2, 1.6, 0x5b6b9c, -46, 4.2, -72, 8);
  label(g, "Mini Mundo", -56, -48);
  const lake = new THREE.Mesh(new THREE.CircleGeometry(24, 26), lam(0x14333c));
  lake.rotation.x = -Math.PI / 2;
  lake.position.set(56, 0.05, -74);
  g.add(lake);
  box(g, 2.6, 1, 1.4, 0xf4f4f0, 50, 0.8, -68);
  cyl(g, 0.28, 1.6, 0xf4f4f0, 49.2, 1.8, -68);
  sph(g, 0.42, 0xf4f4f0, 49.2, 2.7, -68);
  tree(g, 38, -60, 1);
  tree(g, 74, -62, 1.1);
  tree(g, 78, -84, 0.9);
  tree(g, 36, -88, 1.2);
  tree(g, 58, -98, 1);
  label(g, "Lago Negro", 56, -46);
  for (let i = 0; i < 5; i++) {
    const s = flat(g, 6, 17, 0x8b8880, 16 + (i % 2 ? 3.6 : -3.6), 0.06, -48 - i * 13);
    s.rotation.z = i % 2 ? 0.42 : -0.42;
  }
  for (let i = 0; i < 12; i++) {
    const r = 0.7 + rnd() * 0.6;
    sph(g, r, [0xc96b8e, 0x7c86c9, 0xd88ab0][i % 3], 9 + rnd() * 14, r * 0.7, -46 - rnd() * 62);
  }
  label(g, "Rua Torta", 16, -40);
  chalet(28, -16, 1.1, 0x7a5236, 0x6e2f22);
  chalet(38, -8, 0.9, 0xe9dcc3, 0x74362a);
  chalet(-20, -10, 1, 0x8a5f3a, 0x5f2a20);
  chalet(-30, -16, 0.9, 0xdbc9a6, 0x6e2f22);
  const hues = [0x7c86c9, 0x8f7cc9, 0x6c9ac9];
  for (let i = 0; i < 14; i++) {
    const r = 0.8 + rnd() * 0.9;
    sph(g, r, hues[i % 3], -34 + rnd() * 70, r * 0.7, rnd() * 8);
  }
  tree(g, -70, -20, 1);
  tree(g, 70, -20, 1.1);
  picture(g, 12, 7, -48, 8, -16.2, paintGramado);
}

function buildCanela(g) {
  box(g, 12, 13, 24, 0x8a8a82, 0, 6.5, -30);
  box(g, 8, 24, 8, 0x92928a, 0, 12, -13);
  cone(g, 5.6, 22, 0x7b7b73, 0, 35, -13, 8);
  box(g, 0.4, 3.4, 0.4, 0xd9d9d2, 0, 47.5, -13);
  box(g, 1.9, 0.4, 0.4, 0xd9d9d2, 0, 48, -13);
  box(g, 3, 6.5, 0.7, 0x3a3630, 0, 3.2, -8.9);
  for (const x of [-4.5, 4.5]) cone(g, 1, 4.5, 0x7b7b73, x, 26, -13, 6);
  for (const z of [-22, -30, -38]) for (const x of [-6.5, 6.5]) cone(g, 0.8, 3.4, 0x7b7b73, x, 14.5, z, 6);
  label(g, "Catedral de Pedra", 0, -4);
  box(g, 28, 28, 12, 0x6e6a60, -74, 14, -64);
  box(g, 20, 28, 12, 0x6e6a60, -46, 14, -64);
  const fall = new THREE.Mesh(new THREE.PlaneGeometry(5, 27), new THREE.MeshBasicMaterial({ color: 0xeef7fb }));
  fall.position.set(-58, 13.5, -57.9);
  g.add(fall);
  flat(g, 14, 12, 0x4f86a0, -58, 0.06, -50);
  flat(g, 6, 22, 0x4f86a0, -58, 0.05, -34);
  tree(g, -78, -46, 1.1);
  tree(g, -38, -44, 1.2);
  label(g, "Cascata do Caracol", -58, -26);
  box(g, 26, 24, 14, 0x6e6a60, 52, 12, -78);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(7, 0.7, 20), new THREE.MeshLambertMaterial({ color: 0xbfe3ee, transparent: true, opacity: 0.5 }));
  deck.position.set(52, 24.3, -61);
  g.add(deck);
  for (const x of [48.8, 55.2]) box(g, 0.25, 1.3, 20, 0x9db8c2, x, 25.2, -61);
  box(g, 7, 1.3, 0.25, 0x9db8c2, 52, 25.2, -51.1);
  label(g, "Skyglass Canela", 52, -44);
  const hill = new THREE.Mesh(new THREE.ConeGeometry(24, 20, 8), lam(0x5e7f52));
  hill.position.set(92, 8, -80);
  g.add(hill);
  for (let i = 0; i < 6; i++) {
    const a = -0.6 + i * 0.5;
    const rr = 26 - i * 1.5;
    const px = 92 + Math.sin(a) * rr, pz = -80 + Math.cos(a) * rr;
    const py = 13 - i * 2.1;
    box(g, 2.4, 0.4, 3.4, 0x8a5a3a, px, py, pz, a);
    cyl(g, 0.16, py, 0x6b5a48, px, py / 2, pz);
  }
  box(g, 1.6, 0.9, 2.4, 0xc23b2e, 92 + Math.sin(0.4) * 24.5, 9.4, -80 + Math.cos(0.4) * 24.5);
  cyl(g, 0.2, 16, 0x6b5a48, 70, 8, -58);
  cyl(g, 0.2, 7, 0x6b5a48, 96, 3.5, -50);
  const zip = box(g, 27, 0.12, 0.12, 0x2c2c2c, 83, 12, -54, 0.3);
  zip.rotation.z = -0.33;
  label(g, "Alpen Park", 88, -50);
  flat(g, 60, 44, 0x557a48, -12, 0.04, -110);
  const horse = new THREE.Mesh(new THREE.RingGeometry(10, 17, 26, 1, 0.5, 4.2), new THREE.MeshLambertMaterial({ color: 0x4f86a0, side: THREE.DoubleSide }));
  horse.rotation.x = -Math.PI / 2;
  horse.position.set(-12, 0.07, -112);
  g.add(horse);
  box(g, 5, 0.5, 4, 0x8a6b4a, -12, 1.6, -92);
  for (const x of [-14.2, -9.8]) box(g, 0.18, 1.4, 4, 0x6b5a48, x, 2.5, -92);
  tree(g, -34, -100, 1);
  tree(g, 8, -104, 1.1);
  label(g, "Parque da Ferradura", -12, -84);
  tree(g, -20, -18, 1.1);
  tree(g, 24, -20, 1.2);
  picture(g, 12, 7, 52, 11, -70.8, paintCaracol);
}

function buildMissoes(g) {
  const red = 0x94502f, red2 = 0x8a4629;
  flat(g, 90, 64, 0x9b8a66, 0, 0.03, -40);
  for (const x of [-21, -7, 7, 21]) box(g, 5, 18, 4, red, x, 9, -26);
  box(g, 50, 4, 4, red2, 0, 20, -26);
  for (const x of [-14, 0, 14]) box(g, 4.6, 9, 3.6, red, x, 26.5, -26);
  box(g, 36, 3.4, 3.6, red2, 0, 33.2, -26);
  box(g, 18, 3.8, 3.6, red, 0, 36.8, -26);
  box(g, 0.6, 5, 0.6, red2, 0, 41.2, -26);
  box(g, 2.8, 0.6, 0.6, red2, 0, 42, -26);
  box(g, 3, 10, 34, red2, -27, 5, -45);
  box(g, 3, 8, 28, red, 27, 4, -42);
  for (let i = 0; i < 12; i++) {
    const h = 2.5 + rnd() * 6;
    cyl(g, 1.2, h, red, -15 + (i % 6) * 6, h / 2, -36 - Math.floor(i / 6) * 9);
  }
  label(g, "Ruínas de São Miguel Arcanjo", 0, -10);
  for (const x of [-16, 16]) {
    cyl(g, 0.2, 9, 0x3a3f45, x, 4.5, -14);
    sph(g, 0.5, 0xffd9a0, x, 9.2, -14);
  }
  for (const z of [-8, -4]) for (const x of [-8, 0, 8]) box(g, 6, 0.5, 1, 0x6d5b43, x, 0.6, z);
  label(g, "Espetáculo Som e Luz", 26, -6);
  box(g, 26, 1.2, 13, 0xf2efe6, -58, 6.6, -30);
  for (const x of [-69, -61, -55, -47]) box(g, 1.2, 6, 1.2, 0xe8e4da, x, 3, -30);
  const glass = new THREE.Mesh(new THREE.BoxGeometry(22, 4.6, 9), new THREE.MeshLambertMaterial({ color: 0xa8c6cf, transparent: true, opacity: 0.55 }));
  glass.position.set(-58, 3.4, -30);
  g.add(glass);
  label(g, "Museu das Missões", -58, -18);
  cyl(g, 4.2, 1, 0x8f8577, 52, 0.5, -30);
  cyl(g, 3.2, 0.5, 0x69a3c0, 52, 1.15, -30);
  box(g, 7, 4.5, 1.4, 0x8f8577, 52, 2.25, -34);
  cone(g, 1.1, 2, 0x8f8577, 52, 5.5, -34, 4, Math.PI / 4);
  label(g, "Fonte Missioneira", 52, -20);
  box(g, 11, 6.5, 15, 0xf0ead8, 74, 3.25, -68);
  cone(g, 8.4, 4.2, 0x8a3626, 74, 8.6, -68, 4, Math.PI / 4);
  box(g, 0.4, 3, 0.4, 0xf0ead8, 74, 12.4, -68);
  box(g, 1.7, 0.4, 0.4, 0xf0ead8, 74, 13, -68);
  box(g, 2.4, 4.5, 0.6, 0x4a3a2c, 74, 2.25, -60.4);
  label(g, "Santuário do Caaró", 74, -52);
  tree(g, -80, -60, 1);
  tree(g, 40, -70, 1.1);
  tree(g, -40, -76, 0.9);
  picture(g, 13, 6.5, -58, 7.4, -23.2, paintMissoes);
}

function buildUruguaiana(g) {
  flat(g, 560, 130, 0x3e6e8c, 0, 0.08, -110);
  box(g, 8, 1.8, 150, 0xb5b0a6, 28, 11, -110);
  box(g, 0.5, 1.5, 150, 0x8b867c, 24.5, 12.4, -110);
  box(g, 0.5, 1.5, 150, 0x8b867c, 31.5, 12.4, -110);
  for (const z of [-170, -140, -110, -80, -50]) box(g, 6, 11, 3.4, 0x9d988e, 28, 5.5, z);
  box(g, 9, 26, 4.5, 0xa8a398, 28, 13, -48);
  box(g, 9, 26, 4.5, 0xa8a398, 28, 13, -172);
  flag(g, 12, -28, drawBrazil);
  flag(g, 42, -30, drawArgentina);
  label(g, "Ponte Getúlio Vargas", 28, -14);
  cyl(g, 1.4, 36, 0xd9d9d2, -64, 18, -110);
  box(g, 7, 1.6, 150, 0xc9c4ba, -64, 10, -110);
  for (const dz of [14, 30, 48, 66]) for (const sgn of [1, -1]) {
    const len = Math.hypot(25, dz);
    const cb = box(g, 0.16, len, 0.16, 0xd9d9d2, -64, 23, -110 + sgn * dz / 2);
    cb.rotation.x = sgn * Math.atan2(dz, 25);
  }
  label(g, "Ponte da Integração", -64, -16);
  box(g, 16, 9, 12, 0xefe6d0, 66, 4.5, -24);
  for (const x of [60, 72]) {
    box(g, 4.5, 16, 4.5, 0xe8ddc4, x, 8, -19);
    cone(g, 3, 4.5, 0x8a3626, x, 18.2, -19, 4, Math.PI / 4);
  }
  sph(g, 3.4, 0xc9b498, 66, 10.5, -27);
  box(g, 2.6, 5.5, 0.7, 0x4a3a2c, 66, 2.75, -17.6);
  label(g, "Catedral de Sant'Ana", 66, -6);
  flat(g, 30, 22, 0x5c8a4a, -24, 0.05, -22);
  flat(g, 3, 22, 0xc9b189, -24, 0.07, -22);
  box(g, 2.2, 1.6, 2.2, 0xd8d3c4, -24, 0.8, -22);
  cyl(g, 0.5, 5.5, 0xd8d3c4, -24, 4.35, -22);
  sph(g, 0.7, 0x74572e, -24, 7.5, -22);
  for (const x of [-34, -14]) box(g, 3, 0.5, 1, 0x6d5b43, x, 0.6, -16);
  tree(g, -36, -28, 0.9);
  tree(g, -12, -28, 1);
  label(g, "Praça Barão do Rio Branco", -24, -6);
  cyl(g, 13, 1.4, 0x69945a, -10, 0.7, -110);
  tree(g, -14, -108, 0.9);
  tree(g, -6, -114, 0.8);
  label(g, "Ilha Brasileira", -10, -38);
  box(g, 12, 5, 7, 0xd8cdb4, -88, 2.5, -24);
  for (let i = 0; i < 6; i++) {
    const cx = -96 + rnd() * 22, cz = -10 - rnd() * 18;
    box(g, 2.4, 1.4, 1.2, 0x5b3d28, cx, 1, cz);
    box(g, 0.8, 0.8, 0.8, 0x4e3322, cx + 1.5, 1.3, cz);
  }
  picture(g, 11, 6, 66, 6.8, -17.6, paintPampa);
}

function buildChui(g) {
  flat(g, 9, 120, 0x7d7a72, -2.5, 0.05, -58);
  flat(g, 9, 120, 0x8a867c, 6.5, 0.05, -58);
  for (let z = -6; z > -112; z -= 9) flat(g, 0.5, 4, 0xf4f4f0, 2, 0.07, z);
  flag(g, -12, -34, drawBrazil);
  flag(g, 16, -34, drawUruguay);
  label(g, "Avenida Internacional", 2, -8);
  for (let i = 0; i < 4; i++) {
    box(g, 10, 5.5 + (i % 2), 8, [0xd9c9a8, 0xc9b498, 0xd4c4b0, 0xbfae90][i], -22, 3, -20 - i * 14);
  }
  for (let i = 0; i < 3; i++) {
    box(g, 14, 8, 10, [0xdfe3e8, 0xc9d6e0, 0xe8e2d4][i], 26, 4, -22 - i * 16);
    box(g, 10, 2.2, 0.6, [0xc23b2e, 0x2e6d94, 0xc9a437][i], 26, 9.4, -17.6 - i * 16);
  }
  label(g, "Duty Free Shops", 26, -8);
  cyl(g, 2, 7, 0xf2f0ea, -52, 3.5, -66);
  cyl(g, 2, 3.4, 0xc23b2e, -52, 8.7, -66);
  cyl(g, 2, 3.4, 0xf2f0ea, -52, 12.1, -66);
  sph(g, 1.2, 0xfff3cf, -52, 14.6, -66);
  cyl(g, 2.5, 0.5, 0x3a3630, -52, 15.6, -66);
  label(g, "Farol do Chuí", -52, -50);
  box(g, 26, 5, 2.2, 0x8f8577, 58, 2.5, -74);
  box(g, 26, 5, 2.2, 0x8f8577, 58, 2.5, -98);
  box(g, 2.2, 5, 26, 0x8f8577, 46, 2.5, -86);
  box(g, 2.2, 5, 26, 0x8f8577, 70, 2.5, -86);
  for (const [bx, bz] of [[46, -74], [70, -74], [46, -98], [70, -98]]) box(g, 5, 7, 5, 0x857b6d, bx, 3.5, bz);
  box(g, 9, 6, 7, 0x9a8f80, 58, 3, -86);
  label(g, "Forte de São Miguel", 58, -58);
  for (let i = 0; i < 5; i++) {
    const a = i * Math.PI * 2 / 5;
    const m = a + Math.PI / 5;
    box(g, 21, 5.5, 3, 0x8f8577, -68 + Math.sin(m) * 14, 2.75, -100 + Math.cos(m) * 14, m);
    cone(g, 2.6, 4, 0x857b6d, -68 + Math.sin(a) * 17, 4.5, -100 + Math.cos(a) * 17, 4, a);
  }
  box(g, 8, 7, 6, 0x9a8f80, -68, 3.5, -100);
  label(g, "Fortaleza de Santa Teresa", -68, -80);
  tree(g, -34, -66, 1);
  tree(g, 40, -52, 1.1);
  picture(g, 10, 6, 26, 6.8, -16.8, paintFarol);
}

function buildTorres(g) {
  flat(g, 560, 300, 0x2e6d94, 0, 0.12, -250);
  flat(g, 230, 34, 0xe8d7a8, -10, 0.16, -88);
  label(g, "Praia da Cal", 0, -66);
  for (let i = 0; i < 5; i++) {
    const bx = -62 - i * 9, bh = 30 + (i % 3) * 9, bz = -98 - (i % 2) * 8;
    box(g, 10, bh, 12 + (i % 2) * 4, 0x3e3a36, bx, bh / 2, bz);
    box(g, 9, 0.8, 11, 0x4a6b40, bx, bh + 0.4, bz);
  }
  label(g, "Parque da Guarita", -62, -66);
  const hill = new THREE.Mesh(new THREE.ConeGeometry(26, 24, 9), lam(0x5e7f52));
  hill.position.set(64, 10, -84);
  g.add(hill);
  cyl(g, 1.7, 5, 0xf2f0ea, 64, 24.5, -84);
  cyl(g, 1.7, 2.4, 0xc23b2e, 64, 28.2, -84);
  cyl(g, 1.7, 2.4, 0xf2f0ea, 64, 30.6, -84);
  sph(g, 1.1, 0xfff3cf, 64, 32.6, -84);
  cyl(g, 2.2, 0.5, 0x3a3630, 64, 33.6, -84);
  label(g, "Morro do Farol", 64, -56);
  cyl(g, 11, 2.4, 0x5a544c, 18, 1.2, -190);
  for (let i = 0; i < 6; i++) sph(g, 0.9, 0x5c4331, 13 + rnd() * 10, 2.7, -194 + rnd() * 8);
  label(g, "Ilha dos Lobos", 24, -70);
  flat(g, 14, 130, 0x4f86a0, -108, 0.14, -55);
  cyl(g, 0.5, 11, 0x8a6b4a, -116.5, 5.5, -30);
  cyl(g, 0.5, 11, 0x8a6b4a, -99.5, 5.5, -30);
  box(g, 22, 0.5, 2.8, 0x8a6b4a, -108, 3.4, -30);
  const cab1 = box(g, 20, 0.14, 0.14, 0x2c2c2c, -112.8, 7.4, -30);
  cab1.rotation.z = 0.32;
  const cab2 = box(g, 20, 0.14, 0.14, 0x2c2c2c, -103.2, 7.4, -30);
  cab2.rotation.z = -0.32;
  label(g, "Ponte Pênsil do Mampituba", -108, -14);
  for (let i = 0; i < 4; i++) {
    const hx = -34 + i * 18, hz = -22 - (i % 2) * 8;
    box(g, 7, 4, 6, [0xf0e3c8, 0xd9c9a4, 0xe6d5ae, 0xc9d6e0][i], hx, 2, hz);
    cone(g, 5.2, 2.8, 0x8a3626, hx, 5.4, hz, 4, Math.PI / 4);
  }
  tree(g, -52, -30, 1);
  tree(g, 44, -26, 1.1);
  picture(g, 12, 7, -62, 13, -91.6, paintTorres);
}

const cities = [
  {
    name: "Porto Alegre", stop: "Porto Alegre", sign: "Porto Alegre", wp: 0, clear: 500, build: buildPortoAlegre,
    photo: "assets/places/porto-alegre.jpg", photoCredit: "Lechatjaune · CC BY-SA 3.0", video: "MTrPJ4YgBSo",
    info: "Capital of Rio Grande do Sul on the Guaíba. The 1928 Usina do Gasômetro, the Mercado Público open since 1869, the revitalized Orla do Guaíba, the green Parque Farroupilha and the white Fundação Iberê Camargo by Álvaro Siza."
  },
  {
    name: "Gramado", stop: "Gramado", sign: "Gramado", wp: 2, clear: 300, build: buildGramado,
    photo: "assets/places/gramado.jpg", photoCredit: "Jrbresolin · CC BY-SA 3.0", video: "78H0z4N8cVc",
    info: "Serra Gaúcha town of chalets and hydrangeas. Lago Negro framed by Black Forest pines, the Mini Mundo miniature park, the winding Rua Torta, the basalt Igreja Matriz São Pedro and the Palácio dos Festivais of the film festival."
  },
  {
    name: "Canela", stop: "Canela", sign: "Canela", wp: 3, clear: 320, build: buildCanela,
    photo: "assets/places/canela.jpg", photoCredit: "Adelano Lázaro · Public domain", video: "fg8vpO3Ki64",
    info: "The Gothic Catedral de Pedra rises over town. Around it, the 131 meter Cascata do Caracol, the glass Skyglass platform over the Vale da Ferradura, the alpine slides of Alpen Park and the horseshoe canyon of Parque da Ferradura."
  },
  {
    name: "São Miguel das Missões", stop: "São Miguel das Missões", sign: "São Miguel", wp: 5, clear: 300, build: buildMissoes,
    photo: "assets/places/missoes.jpg", photoCredit: "Goldemberg Fonseca · CC BY 2.0", video: "-kGXT0Trv0E",
    info: "UNESCO ruins of São Miguel Arcanjo, red sandstone heart of the Jesuit-Guarani missions. The nightly Som e Luz show, the Museu das Missões by Lúcio Costa, the original Fonte Missioneira and the pilgrimage Santuário do Caaró."
  },
  {
    name: "Uruguaiana · Fronteira Argentina", stop: "Uruguaiana (Argentina)", sign: "Uruguaiana", wp: 7, clear: 500, build: buildUruguaiana,
    photo: "assets/places/uruguaiana.jpg", photoCredit: "Mauricio V. Genta · CC BY-SA 4.0", video: "lCOszhoc0rg",
    info: "Border hub on the Uruguay River: the 1945 Ponte Getúlio Vargas to Paso de los Libres, the Ponte da Integração, the Catedral de Sant'Ana, the Praça Barão do Rio Branco and the tri-border Ilha Brasileira."
  },
  {
    name: "Chuí · Fronteira Uruguai", stop: "Chuí (Uruguay)", sign: "Chuí", wp: 9, clear: 320, build: buildChui,
    photo: "assets/places/chui.jpg", photoCredit: "Aranha Márcio Eliese · CC BY-SA 3.0", video: "yPeQsBZgqCg",
    info: "Southernmost city of Brazil, split down the Avenida Internacional with Chuy, Uruguay. The Farol do Chuí, the Spanish Forte de São Miguel, the star-shaped Fortaleza de Santa Teresa and the duty free corridor."
  },
  {
    name: "Torres", stop: "Torres", sign: "Torres", wp: 11, clear: 500, build: buildTorres,
    photo: "assets/places/torres.jpg", photoCredit: "Paulo Hopper · CC BY-SA 4.0", video: "ge5WusKgA40",
    info: "Where basalt cliffs meet the Atlantic: the towers of Parque da Guarita, the lighthouse on Morro do Farol, the sea lion refuge of Ilha dos Lobos, Praia da Cal between the cliffs and the Ponte Pênsil over the Mampituba."
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
  registerGroupColliders(g);
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
    registerCircleCollider(post.position.x, post.position.z, 0.3);
  }
  for (const s of [-16, 16]) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 6.4, 8), lam(0x3a3f45));
    pole.position.copy(p).addScaledVector(n, 9).addScaledVector(t, s);
    pole.position.y = 3.2;
    scene.add(pole);
    registerCircleCollider(pole.position.x, pole.position.z, 0.3);
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
  addPhotoBillboard(c, p, t, n);
  addVideoBillboard(c, p, t, n);
}

function vegOk(x, z, margin) {
  if (Math.hypot(x, z) > 2880) return false;
  if (trackDist(x, z) < margin) return false;
  return !cities.some(c => Math.hypot(c.origin.x - x, c.origin.z - z) < c.clear);
}

const VEG = [
  [new THREE.CylinderGeometry(0.32, 0.46, 8.6, 7), 4.3, new THREE.ConeGeometry(4.3, 2.9, 9), 9.5, 0x2f5b34, 1000],
  [new THREE.CylinderGeometry(0.28, 0.4, 4.4, 7), 2.2, new THREE.ConeGeometry(3.1, 9.5, 8), 8.4, 0x39633a, 800],
  [new THREE.CylinderGeometry(0.34, 0.5, 5, 7), 2.5, new THREE.SphereGeometry(3.3, 10, 8), 7.6, 0x4a7440, 700],
  [new THREE.CylinderGeometry(0.42, 0.62, 17, 7), 8.5, new THREE.ConeGeometry(6.2, 3.2, 9), 17.4, 0x244d2b, 340],
  [new THREE.CylinderGeometry(0.24, 0.38, 7.4, 7), 3.7, new THREE.SphereGeometry(2.3, 9, 7), 8.1, 0x3d7345, 420]
];
for (const [tg, ty, cg, cy, cc, n] of VEG) {
  const trunks = new THREE.InstancedMesh(tg, lam(0x6b4a33), n);
  const tops = new THREE.InstancedMesh(cg, lam(cc), n);
  let count = 0;
  const spots = [];
  for (let tr = 0; tr < n * 8 && spots.length < n; tr++) {
    const x = -2900 + rnd() * 5800, z = -2900 + rnd() * 5800;
    if (!vegOk(x, z, 26)) continue;
    spots.push([x, z]);
    for (let e = 0; e < 3 && spots.length < n; e++) {
      const ex = x + (rnd() - 0.5) * 44, ez = z + (rnd() - 0.5) * 44;
      if (vegOk(ex, ez, 26)) spots.push([ex, ez]);
    }
  }
  for (const [x, z] of spots) {
    const sc = 0.7 + rnd() * 0.9;
    dummy.rotation.set(0, rnd() * Math.PI * 2, 0);
    dummy.scale.setScalar(sc);
    dummy.position.set(x, ty * sc, z);
    dummy.updateMatrix();
    trunks.setMatrixAt(count, dummy.matrix);
    dummy.position.y = cy * sc;
    dummy.updateMatrix();
    tops.setMatrixAt(count, dummy.matrix);
    registerCircleCollider(x, z, Math.max(0.65, sc * 0.55));
    count++;
  }
  trunks.count = count;
  tops.count = count;
  scene.add(trunks, tops);
}
const BUSHES = 1200;
const bushes = new THREE.InstancedMesh(new THREE.SphereGeometry(1.5, 8, 6), lam(0x557f3f), BUSHES);
let bushCount = 0;
for (let tr = 0; tr < BUSHES * 8 && bushCount < BUSHES; tr++) {
  const x = -2900 + rnd() * 5800, z = -2900 + rnd() * 5800;
  if (!vegOk(x, z, 14)) continue;
  const sc = 0.6 + rnd() * 1.1;
  dummy.rotation.set(0, rnd() * Math.PI * 2, 0);
  dummy.scale.set(sc, sc * 0.62, sc);
  dummy.position.set(x, 0.55 * sc, z);
  dummy.updateMatrix();
  bushes.setMatrixAt(bushCount, dummy.matrix);
  bushCount++;
}
bushes.count = bushCount;
scene.add(bushes);
dummy.rotation.set(0, 0, 0);
dummy.scale.setScalar(1);

let made = 0, tries = 0;
while (made < 12 && tries < 500) {
  tries++;
  const serra = made < 8;
  const x = serra ? 150 + rnd() * 950 : -900 + rnd() * 1800;
  const z = serra ? -1550 + rnd() * 900 : 1000 + rnd() * 700;
  const r = 110 + rnd() * 150, h = 55 + rnd() * 90;
  if (Math.hypot(x, z) > 2800) continue;
  if (trackDist(x, z) < r + 60) continue;
  if (cities.some(c => Math.hypot(c.origin.x - x, c.origin.z - z) < r + c.clear)) continue;
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, 7), lam(0x5e7f52));
  m.position.set(x, h / 2 - 2, z);
  scene.add(m);
  registerCircleCollider(x, z, r * 0.72);
  made++;
}

const GRASS = 1500;
const grass = new THREE.InstancedMesh(new THREE.ConeGeometry(0.45, 3.4, 5), lam(0xcabb86), GRASS);
let grassCount = 0;
for (let tr = 0; tr < GRASS * 8 && grassCount < GRASS; tr++) {
  const x = -2900 + rnd() * 5800, z = -2900 + rnd() * 5800;
  if (!vegOk(x, z, 10)) continue;
  const sc = 0.7 + rnd() * 1.2;
  dummy.rotation.set(0, rnd() * Math.PI * 2, 0);
  dummy.scale.set(sc * 0.5, sc, sc * 0.5);
  dummy.position.set(x, 1.7 * sc, z);
  dummy.updateMatrix();
  grass.setMatrixAt(grassCount, dummy.matrix);
  grassCount++;
}
grass.count = grassCount;
scene.add(grass);
dummy.rotation.set(0, 0, 0);
dummy.scale.setScalar(1);

function makeCow(bodyCol, headCol) {
  const g = new THREE.Group();
  box(g, 1.5, 1.15, 2.7, bodyCol, 0, 1.5, 0);
  box(g, 0.9, 0.85, 0.95, headCol, 0, 1.75, 1.7);
  for (const sx of [-0.28, 0.28]) box(g, 0.2, 0.5, 0.2, headCol, sx, 2.15, 1.95);
  for (const sx of [-0.5, 0.5]) for (const sz of [-0.95, 0.95]) box(g, 0.25, 1.0, 0.25, 0x2a1f18, sx, 0.5, sz);
  const tail = box(g, 0.1, 0.9, 0.1, bodyCol, 0, 1.2, -1.4);
  tail.rotation.x = 0.4;
  return g;
}

function makeSheep() {
  const g = new THREE.Group();
  sph(g, 0.95, 0xe9e4d8, 0, 1.15, 0);
  box(g, 0.55, 0.5, 0.5, 0x2e2822, 0, 1.3, 1.05);
  for (const sx of [-0.4, 0.4]) for (const sz of [-0.55, 0.55]) box(g, 0.16, 0.8, 0.16, 0x3a332b, sx, 0.4, sz);
  return g;
}

function makeHorse(col) {
  const g = new THREE.Group();
  box(g, 0.85, 1.05, 2.5, col, 0, 2.0, 0);
  const neck = box(g, 0.5, 1.3, 0.55, col, 0, 2.75, 1.15);
  neck.rotation.x = -0.5;
  box(g, 0.42, 0.5, 1.0, col, 0, 3.35, 1.8);
  for (const sx of [-0.5, 0.5]) for (const sz of [-1.0, 1.0]) box(g, 0.22, 1.7, 0.22, 0x2a1c12, sx, 0.85, sz);
  const tail = box(g, 0.12, 1.2, 0.12, 0x2a1c12, 0, 1.6, -1.35);
  tail.rotation.x = 0.5;
  return g;
}

function makeRhea() {
  const g = new THREE.Group();
  sph(g, 1.0, 0x9a927f, 0, 2.1, 0);
  const neck = cyl(g, 0.17, 1.7, 0x8a8270, 0, 3.1, 0.5);
  neck.rotation.x = -0.35;
  sph(g, 0.32, 0x8a8270, 0, 3.95, 0.85);
  for (const sx of [-0.35, 0.35]) cyl(g, 0.13, 2.0, 0x6f5b48, sx, 1.0, -0.1);
  return g;
}

function makeCapybara() {
  const g = new THREE.Group();
  box(g, 1.05, 0.85, 1.9, 0x6f4d30, 0, 0.85, 0);
  box(g, 0.65, 0.62, 0.75, 0x5f4028, 0, 0.95, 1.15);
  for (const sx of [-0.38, 0.38]) for (const sz of [-0.6, 0.6]) box(g, 0.22, 0.5, 0.22, 0x4a3220, sx, 0.25, sz);
  return g;
}

function makeLapwing() {
  const g = new THREE.Group();
  box(g, 0.5, 0.4, 0.85, 0x9a8f7a, 0, 0.55, 0);
  box(g, 0.32, 0.34, 0.34, 0x2b2b2b, 0, 0.72, 0.5);
  const crest = box(g, 0.05, 0.4, 0.05, 0x2b2b2b, 0, 0.98, 0.35);
  crest.rotation.x = 0.7;
  box(g, 0.12, 0.24, 0.55, 0xf2efe6, 0, 0.5, 0.05);
  for (const sx of [-0.16, 0.16]) cyl(g, 0.04, 0.5, 0xc98a3a, sx, 0.25, 0);
  return g;
}

function makeDeer() {
  const g = new THREE.Group();
  box(g, 0.8, 1.0, 2.0, 0x7a5030, 0, 1.65, 0);
  const neck = box(g, 0.45, 1.35, 0.45, 0x805534, 0, 2.35, 0.8);
  neck.rotation.x = -0.34;
  box(g, 0.5, 0.48, 0.75, 0x805534, 0, 3.0, 1.25);
  for (const sx of [-0.34, 0.34]) for (const sz of [-0.7, 0.7]) box(g, 0.16, 1.3, 0.16, 0x4a3020, sx, 0.65, sz);
  for (const sx of [-0.2, 0.2]) {
    const antler = box(g, 0.06, 0.75, 0.06, 0x5a4029, sx, 3.6, 1.35);
    antler.rotation.z = sx * 1.8;
  }
  return g;
}

function makeArmadillo() {
  const g = new THREE.Group();
  const shell = sph(g, 0.72, 0x766b59, 0, 0.7, 0);
  shell.scale.set(0.8, 0.65, 1.35);
  const head = cone(g, 0.42, 0.8, 0x827561, 0, 0.62, 0.95, 7);
  head.rotation.x = Math.PI / 2;
  const tail = cone(g, 0.16, 1.0, 0x6d6252, 0, 0.55, -1.0, 7);
  tail.rotation.x = -Math.PI / 2;
  return g;
}

function scatterAnimals(templates, total, margin, clump, jitter) {
  let placed = 0, guard = 0;
  while (placed < total && guard < total * 60) {
    guard++;
    const ax = -2700 + rnd() * 5400, az = -2700 + rnd() * 5400;
    if (!vegOk(ax, az, margin)) continue;
    const n = Math.min(clump, total - placed);
    for (let k = 0; k < n; k++) {
      const x = k === 0 ? ax : ax + (rnd() - 0.5) * jitter;
      const z = k === 0 ? az : az + (rnd() - 0.5) * jitter;
      if (!vegOk(x, z, margin)) continue;
      const a = templates[(rnd() * templates.length) | 0].clone();
      const sc = 0.85 + rnd() * 0.4;
      a.scale.setScalar(sc);
      a.position.set(x, 0, z);
      a.rotation.y = rnd() * Math.PI * 2;
      scene.add(a);
      registerCircleCollider(x, z, 1.1 * sc);
      placed++;
    }
  }
}

scatterAnimals([makeCow(0x4a3324, 0x3a271b), makeCow(0x2c241f, 0x201a15), makeCow(0xcfc4b0, 0x8a5a3a)], 32, 22, 5, 26);
scatterAnimals([makeSheep()], 42, 16, 7, 22);
scatterAnimals([makeHorse(0x5a3a22), makeHorse(0x2e2018)], 16, 22, 3, 22);
scatterAnimals([makeRhea()], 12, 20, 2, 20);
scatterAnimals([makeCapybara()], 12, 20, 4, 16);
scatterAnimals([makeLapwing()], 24, 8, 2, 26);
scatterAnimals([makeDeer()], 14, 18, 3, 22);
scatterAnimals([makeArmadillo()], 18, 12, 2, 18);

const windmillRotors = [];
function makeWindmill() {
  const g = new THREE.Group();
  cyl(g, 0.5, 13, 0x9a9f9c, 0, 6.5, 0);
  cyl(g, 0.34, 13.4, 0x8a8f8c, 0, 6.7, 0);
  cyl(g, 0.75, 0.7, 0x7a7f7c, 0, 13.2, 0);
  const rotor = new THREE.Group();
  rotor.position.set(0, 13.4, 0.7);
  for (let i = 0; i < 12; i++) {
    const a = i * Math.PI / 6;
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.45, 2.6, 0.05), lam(0xd0d4d1));
    b.position.set(Math.sin(a) * 1.4, Math.cos(a) * 1.4, 0);
    b.rotation.z = a;
    rotor.add(b);
  }
  g.add(rotor);
  box(g, 0.08, 1.6, 2.4, 0xb5b0a6, 0, 13.2, -1.7);
  windmillRotors.push(rotor);
  return g;
}
let wmPlaced = 0, wmGuard = 0;
while (wmPlaced < 9 && wmGuard < 400) {
  wmGuard++;
  const x = -2600 + rnd() * 5200, z = -2600 + rnd() * 5200;
  if (!vegOk(x, z, 60)) continue;
  const g = makeWindmill();
  g.position.set(x, 0, z);
  g.rotation.y = rnd() * Math.PI * 2;
  scene.add(g);
  registerCircleCollider(x, z, 1.2);
  wmPlaced++;
}

function fenceLine(x, z, ang, segs) {
  const g = new THREE.Group();
  const dx = Math.sin(ang), dz = Math.cos(ang), gap = 4.2, len = segs * gap;
  for (let i = 0; i <= segs; i++) cyl(g, 0.11, 2.2, 0x6b5335, x + dx * i * gap, 1.1, z + dz * i * gap);
  for (const y of [0.85, 1.65]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, len), lam(0x8a6b4a));
    rail.position.set(x + dx * len / 2, y, z + dz * len / 2);
    rail.rotation.y = ang;
    g.add(rail);
  }
  scene.add(g);
  for (let d = 0; d <= len; d += 1.25) registerCircleCollider(x + dx * d, z + dz * d, 0.45);
}
let fnPlaced = 0, fnGuard = 0;
while (fnPlaced < 12 && fnGuard < 500) {
  fnGuard++;
  const x = -2500 + rnd() * 5000, z = -2500 + rnd() * 5000;
  const ang = rnd() * Math.PI, segs = 6 + (rnd() * 6 | 0);
  const dx = Math.sin(ang), dz = Math.cos(ang);
  if (!vegOk(x, z, 30) || !vegOk(x + dx * segs * 4.2, z + dz * segs * 4.2, 30)) continue;
  fenceLine(x, z, ang, segs);
  fnPlaced++;
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

const SPACING = 10.5;
let cruise = 26;
let dist = cities[0].u * trackLen - 200;
let speed = 0, moving = true;

function placeCar(obj, d) {
  const u = ((d % trackLen) + trackLen) % trackLen / trackLen;
  const p = curve.getPointAt(u), t = curve.getTangentAt(u);
  obj.position.set(p.x, 0.34, p.z);
  obj.rotation.y = Math.atan2(t.x, t.z);
}

function updateTrain(dt) {
  const target = moving && mode === "ride" ? cruise : 0;
  speed += (target - speed) * Math.min(1, dt * (target > speed ? 0.6 : 1.8));
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

let weatherMode = "rain", weatherActive = false, weatherManual = false;
let rainLevel = 0, fogLevel = 0, weatherTimer = 26;

function updateWeather(dt) {
  if (!weatherManual) weatherTimer -= dt;
  if (!weatherManual && weatherTimer <= 0) {
    if (weatherActive) {
      weatherActive = false;
      weatherTimer = 35 + Math.random() * 40;
    } else if (Math.random() < 0.6) {
      weatherActive = true;
      weatherMode = Math.random() < 0.72 ? "rain" : "fog";
      weatherTimer = 22 + Math.random() * 25;
    } else {
      weatherTimer = 12;
    }
  }
  const rainTarget = weatherActive && weatherMode === "rain" ? 1 : 0;
  const fogTarget = weatherActive && weatherMode === "fog" ? 1 : 0;
  rainLevel += (rainTarget - rainLevel) * Math.min(1, dt * 0.7);
  fogLevel += (fogTarget - fogLevel) * Math.min(1, dt * 0.55);
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

const switchButtons = [...document.querySelectorAll("[data-switch]")];
for (const button of switchButtons) button.addEventListener("click", () => {
  const kind = button.dataset.switch;
  if (kind === "time") {
    const target = daylight > 0.4 ? 0 : 0.5;
    elapsed = ((target - tod0 + 1) % 1) * DAY;
  } else if (kind === "weather") {
    weatherMode = weatherMode === "rain" ? "fog" : "rain";
    weatherManual = true;
  } else {
    weatherActive = !weatherActive;
    weatherManual = true;
  }
});

function updateEnvironmentControls() {
  for (const button of switchButtons) {
    const value = button.querySelector("strong");
    if (button.dataset.switch === "time") value.textContent = daylight > 0.4 ? "Day" : "Night";
    if (button.dataset.switch === "weather") value.textContent = weatherMode === "rain" ? "Rain" : "Fog";
    if (button.dataset.switch === "sky") value.textContent = weatherActive ? "Rainy" : "Sunny";
  }
}

function updateSky() {
  dayFrac = (tod0 + elapsed / DAY) % 1;
  const s = Math.sin((dayFrac - 0.25) * Math.PI * 2);
  daylight = THREE.MathUtils.clamp(s * 1.25, 0, 1);
  const duskAmt = Math.max(0, 1 - Math.abs(s) / 0.32);
  sky.copy(colNight).lerp(colDay, daylight);
  sky.lerp(colDusk, duskAmt * 0.55);
  const weatherLevel = Math.max(rainLevel, fogLevel);
  sky.lerp(colRain, weatherLevel * (0.15 + 0.6 * daylight));
  scene.fog.color.copy(sky);
  scene.fog.near = 220 - fogLevel * 180;
  scene.fog.far = 1900 - rainLevel * 900 - fogLevel * 1380;
  const ang = (dayFrac - 0.25) * Math.PI * 2;
  sun.position.set(Math.cos(ang) * 900, Math.sin(ang) * 900, 350);
  sun.intensity = daylight * 2.4 * (1 - 0.55 * weatherLevel);
  sun.color.setHex(0xffffff).lerp(duskSun, duskAmt * 0.8);
  hemi.intensity = 0.18 + daylight * (1 - 0.4 * weatherLevel);
  starMat.opacity = (1 - daylight) ** 2 * (1 - weatherLevel) * 0.9;
  const night = 1 - daylight;
  for (const l of lampLights) l.intensity = night * 60;
  for (const h of lampHeads) h.material.color.setHex(night > 0.4 ? 0xffd9a0 : 0x4a4238);
  head.intensity = Math.min(1, night + weatherLevel * 0.5) * 950;
}

let audioCtx = null, masterGain = null, chugGain = null, chugFilter = null, rainGain = null;
let muted = false, chugPhase = 0;

function noiseBuffer(ctx) {
  const b = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  const d = b.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return b;
}

function initAudio() {
  if (audioCtx) {
    if (audioCtx.state === "suspended") audioCtx.resume();
    return;
  }
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  audioCtx = new AC();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = muted ? 0 : 1;
  masterGain.connect(audioCtx.destination);
  const nb = noiseBuffer(audioCtx);
  const chugSrc = audioCtx.createBufferSource();
  chugSrc.buffer = nb;
  chugSrc.loop = true;
  chugFilter = audioCtx.createBiquadFilter();
  chugFilter.type = "bandpass";
  chugFilter.frequency.value = 320;
  chugFilter.Q.value = 0.9;
  chugGain = audioCtx.createGain();
  chugGain.gain.value = 0;
  chugSrc.connect(chugFilter);
  chugFilter.connect(chugGain);
  chugGain.connect(masterGain);
  chugSrc.start();
  const rainSrc = audioCtx.createBufferSource();
  rainSrc.buffer = nb;
  rainSrc.loop = true;
  rainSrc.playbackRate.value = 1.7;
  const rainFilter = audioCtx.createBiquadFilter();
  rainFilter.type = "highpass";
  rainFilter.frequency.value = 1400;
  rainGain = audioCtx.createGain();
  rainGain.gain.value = 0;
  rainSrc.connect(rainFilter);
  rainFilter.connect(rainGain);
  rainGain.connect(masterGain);
  rainSrc.start();
}

function whistle() {
  if (!audioCtx || muted) return;
  const t = audioCtx.currentTime;
  for (const f of [524, 660]) {
    const o = audioCtx.createOscillator();
    o.type = "triangle";
    o.frequency.value = f;
    const gn = audioCtx.createGain();
    gn.gain.setValueAtTime(0, t);
    gn.gain.linearRampToValueAtTime(0.14, t + 0.06);
    gn.gain.setValueAtTime(0.14, t + 0.7);
    gn.gain.linearRampToValueAtTime(0, t + 1.0);
    o.connect(gn);
    gn.connect(masterGain);
    o.start(t);
    o.stop(t + 1.05);
  }
}

function updateAudio(dt) {
  if (!audioCtx) return;
  chugPhase += speed * dt * 0.55;
  const pulse = Math.max(0, Math.sin(chugPhase * Math.PI * 2)) ** 3;
  const level = Math.min(1, speed / 40);
  chugGain.gain.value = level * (0.05 + 0.3 * pulse);
  chugFilter.frequency.value = 250 + speed * 6;
  rainGain.gain.value = rainLevel * 0.12;
}

const keys = new Set();
let mode = "ride";
const walker = { pos: new THREE.Vector3(), yaw: 0, pitch: 0 };
const orbit = { yaw: 0, pitch: 0.3 };
let dragging = false;

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
    if (Math.abs(speed) > 3) return;
    speed = 0;
    moving = false;
    dragging = false;
    const u = ((dist % trackLen) + trackLen) % trackLen / trackLen;
    const t = curve.getTangentAt(u);
    const n = new THREE.Vector3(t.z, 0, -t.x);
    walker.pos.copy(loco.position).addScaledVector(n, 7);
    walker.pos.y = 2;
    walker.yaw = Math.atan2(-n.x, -n.z);
    walker.pitch = 0;
    mode = "walk";
    lockPointer();
  } else if (walker.pos.distanceTo(loco.position) < 18) {
    mode = "ride";
    document.exitPointerLock();
  }
}

addEventListener("keydown", e => {
  initAudio();
  if (e.code === "Space") {
    e.preventDefault();
    if (mode === "ride") {
      moving = !moving;
      if (moving) whistle();
    }
  }
  if (e.code === "KeyE") toggleWalk();
  if (e.code === "KeyM") {
    muted = !muted;
    if (masterGain) masterGain.gain.value = muted ? 0 : 1;
  }
  if (mode === "ride" && e.code === "ArrowUp") {
    e.preventDefault();
    cruise = Math.min(55, cruise + 7);
  }
  if (mode === "ride" && e.code === "ArrowDown") {
    e.preventDefault();
    cruise = Math.max(8, cruise - 7);
  }
  keys.add(e.code);
});
addEventListener("keyup", e => keys.delete(e.code));
addEventListener("mousemove", e => {
  if (mode === "walk" && document.pointerLockElement === canvas) {
    walker.yaw -= e.movementX * 0.0022;
    walker.pitch = THREE.MathUtils.clamp(walker.pitch - e.movementY * 0.0022, -1.35, 1.35);
  } else if (mode === "ride" && dragging) {
    orbit.yaw -= e.movementX * 0.005;
    orbit.pitch = THREE.MathUtils.clamp(orbit.pitch + e.movementY * 0.004, 0.06, 1.3);
  }
});
canvas.addEventListener("mousedown", () => {
  if (mode === "ride") dragging = true;
});
addEventListener("mouseup", () => {
  dragging = false;
});
canvas.addEventListener("click", () => {
  initAudio();
  if (mode === "walk" && !document.pointerLockElement) lockPointer();
});

const euler = new THREE.Euler(0, 0, 0, "YXZ");
const localWalker = new THREE.Vector3();

function trainBlocks(v) {
  for (const car of cars) {
    car.updateMatrixWorld(true);
    localWalker.copy(v);
    car.worldToLocal(localWalker);
    if (Math.abs(localWalker.x) < 2.1 && Math.abs(localWalker.z) < 5.2) return true;
  }
  return false;
}

function moveWalker(delta) {
  const next = walker.pos.clone();
  next.x += delta.x;
  if (!blockedAt(next) && !trainBlocks(next)) walker.pos.x = next.x;
  next.copy(walker.pos);
  next.z += delta.z;
  if (!blockedAt(next) && !trainBlocks(next)) walker.pos.z = next.z;
  if (walker.pos.length() > 3100) walker.pos.setLength(3100);
  walker.pos.y = 2;
}

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
    moveWalker(mv.multiplyScalar((run ? 26 : 12) * dt));
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
  const heading = Math.atan2(t.x, t.z);
  const a = heading + Math.PI + orbit.yaw;
  const r = 27 * Math.cos(orbit.pitch);
  const desired = new THREE.Vector3(p.x + Math.sin(a) * r, 3 + 27 * Math.sin(orbit.pitch), p.z + Math.cos(a) * r);
  const k = 1 - Math.exp(-3.2 * dt);
  camera.position.lerp(desired, k);
  const look = p.clone().addScaledVector(t, 12);
  look.y = 3;
  camLook.lerp(look, k);
  camera.lookAt(camLook);
}

const boardCenter = new THREE.Vector3();
const boardTopLeft = new THREE.Vector3();
const boardTopRight = new THREE.Vector3();
const boardBottomLeft = new THREE.Vector3();
const cameraDirection = new THREE.Vector3();

function projectBoardPoint(v) {
  v.project(camera);
  v.x = (v.x * 0.5 + 0.5) * innerWidth;
  v.y = (-v.y * 0.5 + 0.5) * innerHeight;
}

function updateVideoBoards() {
  camera.getWorldDirection(cameraDirection);
  let active = null, activeDistance = Infinity;
  for (const board of videoBoards) {
    board.screen.updateWorldMatrix(true, false);
    board.screen.getWorldPosition(boardCenter);
    const distance = boardCenter.distanceTo(camera.position);
    const facing = boardCenter.clone().sub(camera.position).dot(cameraDirection) > 0;
    if (distance < 330 && distance < activeDistance && facing) {
      active = board;
      activeDistance = distance;
    }
  }
  for (const board of videoBoards) {
    if (board !== active) {
      board.el.style.display = "none";
      if (board.loaded) {
        board.iframe.removeAttribute("src");
        board.loaded = false;
      }
      continue;
    }
    boardTopLeft.set(-9, 5.05, 0);
    boardTopRight.set(9, 5.05, 0);
    boardBottomLeft.set(-9, -5.05, 0);
    board.screen.localToWorld(boardTopLeft);
    board.screen.localToWorld(boardTopRight);
    board.screen.localToWorld(boardBottomLeft);
    projectBoardPoint(boardTopLeft);
    projectBoardPoint(boardTopRight);
    projectBoardPoint(boardBottomLeft);
    const width = boardTopLeft.distanceTo(boardTopRight);
    const height = boardTopLeft.distanceTo(boardBottomLeft);
    const angle = Math.atan2(boardTopRight.y - boardTopLeft.y, boardTopRight.x - boardTopLeft.x);
    if (width < 34 || boardTopLeft.x > innerWidth || boardTopRight.x < 0 || boardTopLeft.y > innerHeight || boardBottomLeft.y < 0) {
      board.el.style.display = "none";
      continue;
    }
    board.el.style.display = "block";
    board.el.style.left = boardTopLeft.x + "px";
    board.el.style.top = boardTopLeft.y + "px";
    board.el.style.width = width + "px";
    board.el.style.height = height + "px";
    board.el.style.transform = "rotate(" + angle + "rad)";
    if (!board.loaded) {
      board.iframe.src = "https://www.youtube-nocookie.com/embed/" + board.c.video + "?autoplay=1&mute=1&loop=1&playlist=" + board.c.video + "&controls=0&rel=0&playsinline=1";
      board.loaded = true;
    }
  }
}

const cityName = document.getElementById("cityName");
const cityInfo = document.getElementById("cityInfo");
const locPanel = document.getElementById("loc");
const clockEl = document.getElementById("clock");
const weatherEl = document.getElementById("weather");
const speedEl = document.getElementById("speed");
const nextEl = document.getElementById("next");
const controlsEl = document.getElementById("controls");
let currentCity = null;

function setHtml(el, s) {
  if (el.dataset.s !== s) {
    el.dataset.s = s;
    el.innerHTML = s;
  }
}

function cmdRows(list) {
  let s = "";
  for (const r of list) s += '<div class="' + (r[2] ? "dim" : "") + '"><b>' + r[0] + "</b>" + r[1] + "</div>";
  return s;
}

function updateControls() {
  let list;
  if (mode === "ride") {
    list = [
      ["SPACE", moving ? "stop the train" : "depart", false],
      ["&#8593; &#8595;", "faster · slower", false],
      ["E", "step off the train", Math.abs(speed) > 3],
      ["DRAG", "rotate the camera", false],
      ["M", muted ? "sound on" : "sound off", false]
    ];
  } else {
    const near = walker.pos.distanceTo(loco.position) < 18;
    list = [
      ["W A S D", "walk", false],
      ["SHIFT", "run", false],
      ["MOUSE", "look around", false],
      ["E", "board the train", !near],
      ["M", muted ? "sound on" : "sound off", false]
    ];
  }
  setHtml(controlsEl, '<div class="ttl">Commands</div>' + cmdRows(list));
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
  setHtml(weatherEl, weatherActive ? (weatherMode === "rain" ? "Rain over the pampa" : "Fog over the pampa") : "Sunny skies");
  setHtml(speedEl, mode === "ride" ? Math.round(speed * 2.2) + " km/h" : "on foot");
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
  } else {
    setHtml(nextEl, currentCity ? "Walking in <b>" + currentCity.stop + "</b>" : "Walking the pampa");
  }
  updateControls();
  updateEnvironmentControls();
}

const overlay = document.getElementById("overlay");
overlay.addEventListener("click", () => {
  overlay.style.display = "none";
  initAudio();
  whistle();
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
  updateVideoBoards();
  updateWeather(dt);
  updateSky();
  updateSmoke(dt);
  for (const r of windmillRotors) r.rotation.z += dt * 1.1;
  updateAudio(dt);
  updateHud();
  renderer.render(scene, camera);
}
animate();
