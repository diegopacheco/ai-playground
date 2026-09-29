import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { camoTexture, carbonTexture, tireTexture } from './textures.js';

export const COLORS = [
  { name: 'Firecracker Red', hex: '#b3161b' },
  { name: 'Sarge Green', hex: '#4b5a34' },
  { name: 'Hydro Blue', hex: '#1f6fb5' },
  { name: 'Granite', hex: '#5b5f63' },
  { name: 'Bright White', hex: '#eceeee' },
  { name: 'Black Onyx', hex: '#16181a' },
  { name: 'Sunset Orange', hex: '#e2581d' },
  { name: 'Mojave Sand', hex: '#c2a878' },
  { name: 'Solar Yellow', hex: '#e8b820' },
  { name: 'Tahoe Teal', hex: '#1d8a86' },
];

export const FINISHES = ['Gloss', 'Metallic', 'Matte', 'Pearl', 'Camo', 'Carbon'];

const MUD = new THREE.Color('#5a4431');
const SNOW = new THREE.Color('#dfe5ea');

export function paintMaterial(hex, finish) {
  const opts = { color: hex, roughness: 0.3, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.06 };
  if (finish === 'Metallic') Object.assign(opts, { metalness: 0.75, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.08 });
  if (finish === 'Matte') Object.assign(opts, { roughness: 0.82, clearcoat: 0, metalness: 0.0 });
  if (finish === 'Pearl') Object.assign(opts, { metalness: 0.45, roughness: 0.22, iridescence: 0.45, iridescenceIOR: 1.4 });
  if (finish === 'Camo') Object.assign(opts, { color: '#ffffff', map: camoTexture(hex), roughness: 0.75, clearcoat: 0.1 });
  if (finish === 'Carbon') Object.assign(opts, { color: '#ffffff', map: carbonTexture(hex), roughness: 0.28, metalness: 0.4, clearcoat: 1 });
  const mat = new THREE.MeshPhysicalMaterial(opts);
  mat.userData.base = mat.color.clone();
  mat.userData.baseRoughness = mat.roughness;
  mat.userData.baseClearcoat = mat.clearcoat;
  return mat;
}

export function applyDirt(mat, dirt, weather) {
  const tint = weather === 'snow' ? SNOW : MUD;
  mat.color.copy(mat.userData.base).lerp(tint, dirt * 0.62);
  mat.roughness = mat.userData.baseRoughness + (0.95 - mat.userData.baseRoughness) * dirt;
  mat.clearcoat = mat.userData.baseClearcoat * (1 - dirt * 0.9);
}

function sharedMaterials() {
  return {
    plastic: new THREE.MeshStandardMaterial({ color: '#1c1c1d', roughness: 0.72 }),
    flare: new THREE.MeshStandardMaterial({ color: '#1c1c1d', roughness: 0.72, side: THREE.DoubleSide }),
    trim: new THREE.MeshStandardMaterial({ color: '#2b2c2e', roughness: 0.5, metalness: 0.3 }),
    chrome: new THREE.MeshStandardMaterial({ color: '#d7dadd', roughness: 0.12, metalness: 1 }),
    glass: new THREE.MeshPhysicalMaterial({ color: '#0e1419', roughness: 0.04, metalness: 0.1, clearcoat: 1, transparent: true, opacity: 0.88 }),
    head: new THREE.MeshStandardMaterial({ color: '#fffbe8', emissive: '#fff4d0', emissiveIntensity: 1.6, roughness: 0.1 }),
    tail: new THREE.MeshStandardMaterial({ color: '#5a0707', emissive: '#ff1a0a', emissiveIntensity: 0.5, roughness: 0.3 }),
    amber: new THREE.MeshStandardMaterial({ color: '#b35a00', emissive: '#ff8a00', emissiveIntensity: 0.4 }),
    rubber: new THREE.MeshStandardMaterial({ color: '#ffffff', map: tireTexture(), roughness: 0.92 }),
    sidewall: new THREE.MeshStandardMaterial({ color: '#1a1918', roughness: 0.85 }),
    rim: new THREE.MeshStandardMaterial({ color: '#2f3235', roughness: 0.35, metalness: 0.85 }),
    softTop: new THREE.MeshStandardMaterial({ color: '#121212', roughness: 0.95 }),
    lightbar: new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: '#e6f0ff', emissiveIntensity: 1.2 }),
  };
}

function part(parent, geo, mat, x, y, z, rx = 0, ry = 0, rz = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.rotation.set(rx, ry, rz);
  m.castShadow = true;
  m.receiveShadow = true;
  parent.add(m);
  return m;
}

function rbox(parent, w, h, d, mat, x, y, z, r = 0.06) {
  return part(parent, new RoundedBoxGeometry(w, h, d, 3, Math.min(r, w / 2 - 0.001, h / 2 - 0.001, d / 2 - 0.001)), mat, x, y, z);
}

function cyl(parent, rt, rb, h, mat, x, y, z, rx = 0, ry = 0, rz = 0, seg = 20) {
  return part(parent, new THREE.CylinderGeometry(rt, rb, h, seg), mat, x, y, z, rx, ry, rz);
}

function roundLights(b, m, W, y, z, r = 0.1) {
  for (const s of [-1, 1]) {
    cyl(b, r, r, 0.06, m.chrome, s * (W / 2 - 0.3), y, z, Math.PI / 2);
    cyl(b, r * 0.82, r * 0.82, 0.07, m.head, s * (W / 2 - 0.3), y, z + 0.01, Math.PI / 2);
  }
}

function rectLights(b, m, W, y, z, w = 0.34, h = 0.13) {
  for (const s of [-1, 1]) rbox(b, w, h, 0.05, m.head, s * (W / 2 - w / 2 - 0.12), y, z, 0.02);
}

function tailLights(b, m, W, y, z, w = 0.12, h = 0.28) {
  for (const s of [-1, 1]) rbox(b, w, h, 0.05, m.tail, s * (W / 2 - 0.1), y, z, 0.02);
}

function slotGrille(b, m, y, z, slots, w, h) {
  const gap = w / slots;
  for (let k = 0; k < slots; k++) rbox(b, gap * 0.55, h, 0.04, m.plastic, -w / 2 + gap * (k + 0.5), y, z, 0.01);
}

function flares(b, m, spec, y) {
  const r = spec.wheelR;
  for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    const g = new THREE.CylinderGeometry(r + 0.13, r + 0.13, 0.26, 20, 1, true, 0, Math.PI);
    part(b, g, m.flare, sx * (spec.width / 2 - 0.02), y, (sz * spec.wheelbase) / 2, 0, 0, Math.PI / 2);
  }
}

function spareTire(b, m, r, x, y, z, flat = false) {
  const t = cyl(b, r, r, 0.26, m.sidewall, x, y, z, flat ? 0 : Math.PI / 2, 0, 0, 24);
  cyl(b, r * 0.55, r * 0.55, 0.28, m.rim, x, y, z, flat ? 0 : Math.PI / 2);
  return t;
}

function lightBar(b, m, W, y, z) {
  rbox(b, W * 0.8, 0.09, 0.12, m.trim, 0, y, z, 0.03);
  rbox(b, W * 0.76, 0.06, 0.02, m.lightbar, 0, y, z + 0.065, 0.01);
}

function mirrors(b, m, W, y, z) {
  for (const s of [-1, 1]) rbox(b, 0.08, 0.16, 0.2, m.plastic, s * (W / 2 + 0.08), y, z, 0.03);
}

const STYLES = {
  wrangler(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.2;
    const y0 = s.wheelR + 0.2;
    rbox(b, W, 0.72, L * 0.9, paint, 0, y0 + 0.36, 0, 0.07);
    rbox(b, W * 0.94, 0.2, L * 0.32, paint, 0, y0 + 0.8, L * 0.28, 0.08);
    rbox(b, W - 0.02, 0.8, L * 0.5, m.softTop, 0, y0 + 1.1, -L * 0.14, 0.06);
    const ws = rbox(b, W - 0.1, 0.62, 0.05, m.glass, 0, y0 + 1.1, L * 0.12, 0.02);
    ws.rotation.x = -0.18;
    for (const sz of [0.02, -0.26]) for (const sx of [-1, 1]) rbox(b, 0.03, 0.46, L * 0.2, m.glass, sx * (W / 2 + 0.005), y0 + 1.14, sz * L, 0.01);
    rbox(b, W * 0.94, 0.5, 0.06, m.plastic, 0, y0 + 0.56, L * 0.45, 0.02);
    slotGrille(b, m, y0 + 0.56, L * 0.455 + 0.02, 7, W * 0.42, 0.34);
    roundLights(b, m, W, y0 + 0.6, L * 0.458, 0.12);
    rbox(b, W + 0.16, 0.2, 0.28, m.plastic, 0, y0 + 0.08, L * 0.47, 0.04);
    rbox(b, W + 0.1, 0.2, 0.24, m.plastic, 0, y0 + 0.08, -L * 0.46, 0.04);
    flares(b, m, s, s.wheelR + 0.02);
    spareTire(b, m, s.wheelR * 0.92, 0, y0 + 0.6, -L * 0.48 - 0.14);
    tailLights(b, m, W, y0 + 0.45, -L * 0.452);
    mirrors(b, m, W, y0 + 0.95, L * 0.1);
    lightBar(b, m, W, y0 + 1.54, L * 0.1);
  },
  bronco(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.22;
    const y0 = s.wheelR + 0.22;
    rbox(b, W, 0.78, L * 0.92, paint, 0, y0 + 0.39, 0, 0.09);
    rbox(b, W * 0.96, 0.16, L * 0.34, paint, 0, y0 + 0.84, L * 0.28, 0.08);
    rbox(b, W - 0.04, 0.76, L * 0.52, paint, 0, y0 + 1.16, -L * 0.14, 0.1);
    rbox(b, W - 0.02, 0.1, L * 0.52, m.plastic, 0, y0 + 1.56, -L * 0.14, 0.05);
    const ws = rbox(b, W - 0.12, 0.6, 0.05, m.glass, 0, y0 + 1.14, L * 0.125, 0.02);
    ws.rotation.x = -0.24;
    for (const sz of [0.03, -0.14, -0.31]) for (const sx of [-1, 1]) rbox(b, 0.03, 0.44, L * 0.14, m.glass, sx * (W / 2 - 0.015), y0 + 1.18, sz * L, 0.01);
    rbox(b, W * 0.96, 0.44, 0.06, m.plastic, 0, y0 + 0.58, L * 0.462, 0.02);
    roundLights(b, m, W, y0 + 0.6, L * 0.468, 0.13);
    rbox(b, W * 0.45, 0.07, 0.03, m.chrome, 0, y0 + 0.64, L * 0.468, 0.01);
    rbox(b, W + 0.24, 0.24, 0.3, m.plastic, 0, y0 + 0.1, L * 0.47, 0.04);
    rbox(b, W + 0.14, 0.22, 0.24, m.plastic, 0, y0 + 0.08, -L * 0.47, 0.04);
    flares(b, m, s, s.wheelR + 0.03);
    spareTire(b, m, s.wheelR * 0.94, 0, y0 + 0.66, -L * 0.48 - 0.15);
    tailLights(b, m, W, y0 + 0.5, -L * 0.462, 0.1, 0.34);
    mirrors(b, m, W, y0 + 0.98, L * 0.1);
  },
  cruiser(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.06;
    const y0 = s.wheelR + 0.18;
    rbox(b, W, 0.74, L * 0.94, paint, 0, y0 + 0.37, 0, 0.04);
    rbox(b, W * 0.92, 0.22, L * 0.3, paint, 0, y0 + 0.83, L * 0.31, 0.05);
    rbox(b, W - 0.02, 0.82, L * 0.6, paint, 0, y0 + 1.15, -L * 0.15, 0.05);
    const ws = rbox(b, W - 0.12, 0.6, 0.05, m.glass, 0, y0 + 1.14, L * 0.155, 0.02);
    ws.rotation.x = -0.12;
    for (const sz of [0.06, -0.14, -0.34]) for (const sx of [-1, 1]) rbox(b, 0.03, 0.44, L * 0.16, m.glass, sx * (W / 2 + 0.005), y0 + 1.2, sz * L, 0.01);
    rbox(b, W * 0.9, 0.4, 0.05, m.chrome, 0, y0 + 0.58, L * 0.472, 0.01);
    slotGrille(b, m, y0 + 0.58, L * 0.475, 5, W * 0.6, 0.3);
    roundLights(b, m, W, y0 + 0.58, L * 0.476, 0.11);
    rbox(b, W + 0.1, 0.18, 0.2, m.chrome, 0, y0 + 0.1, L * 0.48, 0.03);
    rbox(b, W + 0.06, 0.18, 0.2, m.plastic, 0, y0 + 0.08, -L * 0.48, 0.03);
    cyl(b, 0.07, 0.07, 1.2, m.plastic, W / 2 + 0.06, y0 + 1.1, L * 0.2);
    rbox(b, 0.16, 0.12, 0.16, m.plastic, W / 2 + 0.06, y0 + 1.72, L * 0.2, 0.03);
    rbox(b, W * 0.9, 0.06, L * 0.5, m.trim, 0, y0 + 1.62, -L * 0.15, 0.02);
    for (const sx of [-1, 1]) rbox(b, 0.05, 0.12, L * 0.5, m.trim, sx * W * 0.44, y0 + 1.66, -L * 0.15, 0.02);
    tailLights(b, m, W, y0 + 0.35, -L * 0.472, 0.1, 0.2);
    mirrors(b, m, W, y0 + 0.95, L * 0.13);
  },
  defender(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.06;
    const y0 = s.wheelR + 0.16;
    rbox(b, W, 0.8, L * 0.92, paint, 0, y0 + 0.4, 0, 0.16);
    rbox(b, W * 0.96, 0.18, L * 0.3, paint, 0, y0 + 0.86, L * 0.3, 0.1);
    rbox(b, W - 0.06, 0.8, L * 0.62, paint, 0, y0 + 1.18, -L * 0.14, 0.18);
    rbox(b, W - 0.04, 0.08, L * 0.6, m.softTop, 0, y0 + 1.6, -L * 0.14, 0.04);
    const ws = rbox(b, W - 0.16, 0.6, 0.05, m.glass, 0, y0 + 1.16, L * 0.17, 0.02);
    ws.rotation.x = -0.28;
    for (const sx of [-1, 1]) {
      rbox(b, 0.03, 0.46, L * 0.42, m.glass, sx * (W / 2 - 0.025), y0 + 1.2, -L * 0.08, 0.01);
      rbox(b, W * 0.12, 0.05, L * 0.12, m.glass, sx * W * 0.3, y0 + 1.6, -L * 0.3, 0.01);
    }
    rbox(b, W * 0.94, 0.3, 0.05, m.plastic, 0, y0 + 0.6, L * 0.46, 0.04);
    rectLights(b, m, W, y0 + 0.66, L * 0.462, 0.3, 0.1);
    rbox(b, W + 0.06, 0.26, 0.26, m.plastic, 0, y0 + 0.1, L * 0.465, 0.06);
    rbox(b, W + 0.02, 0.24, 0.2, m.plastic, 0, y0 + 0.08, -L * 0.465, 0.05);
    spareTire(b, m, s.wheelR * 0.9, W * 0.12, y0 + 0.8, -L * 0.47 - 0.14);
    tailLights(b, m, W, y0 + 0.72, -L * 0.46, 0.08, 0.36);
    mirrors(b, m, W, y0 + 0.98, L * 0.14);
  },
  raptor(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.2;
    const y0 = s.wheelR + 0.24;
    rbox(b, W, 0.8, L * 0.94, paint, 0, y0 + 0.4, 0, 0.1);
    rbox(b, W * 0.96, 0.16, L * 0.26, paint, 0, y0 + 0.86, L * 0.34, 0.08);
    rbox(b, W - 0.04, 0.74, L * 0.3, paint, 0, y0 + 1.16, L * 0.04, 0.12);
    const ws = rbox(b, W - 0.16, 0.62, 0.05, m.glass, 0, y0 + 1.12, L * 0.2, 0.02);
    ws.rotation.x = -0.45;
    for (const sz of [0.1, -0.03]) for (const sx of [-1, 1]) rbox(b, 0.03, 0.44, L * 0.11, m.glass, sx * (W / 2 - 0.015), y0 + 1.18, sz * L, 0.01);
    rbox(b, W - 0.3, 0.44, 0.03, m.glass, 0, y0 + 1.2, -L * 0.108, 0.01);
    rbox(b, W - 0.16, 0.08, L * 0.36, m.softTop, 0, y0 + 0.78, -L * 0.28, 0.02);
    for (const sx of [-1, 1]) rbox(b, 0.08, 0.3, L * 0.37, paint, sx * (W / 2 - 0.04), y0 + 0.9, -L * 0.28, 0.03);
    rbox(b, W * 0.9, 0.46, 0.06, m.plastic, 0, y0 + 0.6, L * 0.472, 0.03);
    rbox(b, W * 0.62, 0.1, 0.03, m.amber, 0, y0 + 0.74, L * 0.476, 0.01);
    rectLights(b, m, W, y0 + 0.66, L * 0.472, 0.36, 0.14);
    rbox(b, W + 0.2, 0.26, 0.3, m.plastic, 0, y0 + 0.1, L * 0.47, 0.05);
    rbox(b, W + 0.1, 0.22, 0.2, m.chrome, 0, y0 + 0.1, -L * 0.475, 0.03);
    flares(b, m, s, s.wheelR + 0.05);
    tailLights(b, m, W, y0 + 0.6, -L * 0.472, 0.1, 0.4);
    mirrors(b, m, W, y0 + 1.0, L * 0.17);
    lightBar(b, m, W, y0 + 1.56, L * 0.1);
  },
  hummer(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.04;
    const y0 = s.wheelR + 0.14;
    rbox(b, W, 0.7, L * 0.96, paint, 0, y0 + 0.35, 0, 0.04);
    const hood = rbox(b, W * 0.98, 0.18, L * 0.34, paint, 0, y0 + 0.76, L * 0.3, 0.03);
    hood.rotation.x = 0.07;
    rbox(b, W * 0.84, 0.66, L * 0.56, paint, 0, y0 + 1.03, -L * 0.14, 0.04);
    const ws = rbox(b, W * 0.8, 0.44, 0.05, m.glass, 0, y0 + 1.04, L * 0.14, 0.02);
    ws.rotation.x = -0.05;
    for (const sz of [0.02, -0.2]) for (const sx of [-1, 1]) rbox(b, 0.03, 0.34, L * 0.18, m.glass, sx * (W * 0.42 + 0.005), y0 + 1.08, sz * L, 0.01);
    rbox(b, W * 0.5, 0.3, 0.05, m.plastic, 0, y0 + 0.5, L * 0.482, 0.02);
    slotGrille(b, m, y0 + 0.5, L * 0.485, 8, W * 0.46, 0.26);
    roundLights(b, m, W, y0 + 0.52, L * 0.482, 0.09);
    rbox(b, W + 0.1, 0.22, 0.22, m.trim, 0, y0 + 0.08, L * 0.48, 0.03);
    rbox(b, W, 0.2, 0.2, m.trim, 0, y0 + 0.08, -L * 0.48, 0.03);
    tailLights(b, m, W, y0 + 0.4, -L * 0.482, 0.12, 0.18);
    mirrors(b, m, W * 0.86, y0 + 0.95, L * 0.1);
    lightBar(b, m, W * 0.84, y0 + 1.42, L * 0.08);
  },
  trophy(b, s, paint, m) {
    const L = s.length;
    const W = s.width - 0.3;
    const y0 = s.wheelR + 0.36;
    rbox(b, W * 0.8, 0.46, L * 0.6, paint, 0, y0 + 0.23, L * 0.12, 0.12);
    const nose = rbox(b, W * 0.86, 0.34, L * 0.3, paint, 0, y0 + 0.36, L * 0.33, 0.14);
    nose.rotation.x = 0.12;
    rbox(b, W * 0.74, 0.66, L * 0.26, paint, 0, y0 + 0.78, L * 0.02, 0.16);
    const ws = rbox(b, W * 0.68, 0.46, 0.05, m.glass, 0, y0 + 0.82, L * 0.155, 0.02);
    ws.rotation.x = -0.6;
    for (const sx of [-1, 1]) rbox(b, 0.03, 0.34, L * 0.14, m.glass, sx * W * 0.37, y0 + 0.86, L * 0.02, 0.01);
    rbox(b, W * 0.98, 0.1, L * 0.3, paint, 0, y0 + 0.44, -L * 0.28, 0.04);
    for (const sx of [-1, 1]) {
      cyl(b, 0.05, 0.05, L * 0.42, m.trim, sx * W * 0.42, y0 + 0.55, -L * 0.22, Math.PI / 2);
      cyl(b, 0.05, 0.05, 0.9, m.trim, sx * W * 0.4, y0 + 0.9, -L * 0.1, -0.5);
    }
    spareTire(b, m, s.wheelR * 0.9, -W * 0.22, y0 + 0.62, -L * 0.3, true);
    spareTire(b, m, s.wheelR * 0.9, W * 0.22, y0 + 0.62, -L * 0.3, true);
    rbox(b, W * 0.7, 0.14, 0.2, m.trim, 0, y0 + 1.18, L * 0.08, 0.04);
    for (let k = 0; k < 4; k++) cyl(b, 0.09, 0.09, 0.08, m.head, -W * 0.27 + k * W * 0.18, y0 + 1.18, L * 0.08 + 0.12, Math.PI / 2);
    rectLights(b, m, W, y0 + 0.4, L * 0.48, 0.28, 0.1);
    tailLights(b, m, W, y0 + 0.5, -L * 0.44, 0.1, 0.16);
    for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
      cyl(b, 0.06, 0.06, 0.9, m.chrome, sx * W * 0.34, y0 - 0.05, sz * s.wheelbase * 0.5, 0, 0, sx * 0.5);
    }
  },
};

function wheel(m, r, width) {
  const g = new THREE.Group();
  const spin = new THREE.Group();
  g.add(spin);
  const tire = new THREE.Mesh(new THREE.CylinderGeometry(r, r, width, 32, 1, true), m.rubber);
  tire.rotation.z = Math.PI / 2;
  tire.castShadow = true;
  spin.add(tire);
  for (const s of [-1, 1]) {
    const side = new THREE.Mesh(new THREE.RingGeometry(r * 0.6, r, 32), m.sidewall);
    side.rotation.y = (s * Math.PI) / 2;
    side.position.x = (s * width) / 2;
    spin.add(side);
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.6, r * 0.6, 0.04, 24), m.rim);
    rim.rotation.z = Math.PI / 2;
    rim.position.x = s * (width / 2 - 0.03);
    spin.add(rim);
    for (let k = 0; k < 6; k++) {
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.03, r * 0.95, 0.08), m.rim);
      spoke.rotation.x = (k * Math.PI) / 3;
      spoke.position.x = s * (width / 2 - 0.02);
      spin.add(spoke);
    }
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.16, r * 0.16, 0.06, 12), m.chrome);
    hub.rotation.z = Math.PI / 2;
    hub.position.x = s * (width / 2 - 0.01);
    spin.add(hub);
  }
  return { group: g, spin };
}

export function buildCar(spec, hex, finish) {
  const m = sharedMaterials();
  const paint = paintMaterial(hex, finish);
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  STYLES[spec.style](body, spec, paint, m);
  body.traverse((o) => {
    if (o.isMesh) o.castShadow = true;
  });
  const wheels = [];
  const width = spec.wheelR * 0.72;
  for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    const w = wheel(m, spec.wheelR, width);
    w.group.position.set(sx * spec.track / 2, spec.wheelR, sz * spec.wheelbase / 2);
    root.add(w.group);
    wheels.push(w);
  }
  const shadowBlob = new THREE.Mesh(
    new THREE.PlaneGeometry(spec.width * 1.15, spec.length * 1.05),
    new THREE.MeshBasicMaterial({ color: '#000000', transparent: true, opacity: 0.35, depthWrite: false }),
  );
  shadowBlob.rotation.x = -Math.PI / 2;
  shadowBlob.position.y = 0.05;
  root.add(shadowBlob);
  return { root, body, wheels, paint, materials: m, spec, shadowBlob };
}

export function updateCarVisual(visual, car, weather) {
  const { root, body, wheels, spec, materials } = visual;
  root.position.set(car.x, car.y, car.z);
  root.rotation.set(car.pitch, car.heading, car.roll, 'YXZ');
  body.rotation.set(-car.bodyPitch, 0, car.bodyRoll);
  body.position.y = 0.06 + Math.sin(car.wheelAngle * 0.9) * 0.004 * Math.min(1, Math.abs(car.u) / 5);
  wheels.forEach((w, k) => {
    w.group.position.y = spec.wheelR + car.wheelComp[k];
    w.group.rotation.y = k < 2 ? car.steerAngle : 0;
    w.spin.rotation.x = car.wheelAngle;
  });
  materials.tail.emissiveIntensity = car.braking ? 3.2 : 0.6;
  applyDirt(visual.paint, car.dirt, weather);
}
