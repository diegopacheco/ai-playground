import * as THREE from 'three';
import { camoTexture, carbonTexture } from './textures.js';
import { createBuilder } from './cars/builder.js';
import { buildWheel } from './cars/wheel.js';
import { STYLES, RIMS, GRILLE_TEXT } from './cars/styles.js';

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

function grilleTexture(text) {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 160;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#0d0e0f';
  ctx.fillRect(0, 0, 512, 160);
  ctx.strokeStyle = '#2b2d30';
  ctx.lineWidth = 3;
  for (let y = 0; y < 170; y += 16) {
    for (let x = (y / 16) % 2 ? 8 : 0; x < 520; x += 16) {
      ctx.beginPath();
      ctx.moveTo(x, y - 6);
      ctx.lineTo(x + 7, y);
      ctx.lineTo(x, y + 6);
      ctx.lineTo(x - 7, y);
      ctx.closePath();
      ctx.stroke();
    }
  }
  if (text) {
    ctx.font = `900 ${text.length > 6 ? 70 : 104}px "Arial Black", Impact, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#26282b';
    ctx.fillText(text, 256, 84);
    ctx.fillStyle = '#9da1a6';
    ctx.fillText(text, 256, 80);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function materialsFor(spec, paint) {
  const two = THREE.DoubleSide;
  return {
    paint,
    top: new THREE.MeshPhysicalMaterial({ color: '#141516', roughness: 0.45, clearcoat: 0.4, side: two }),
    trim: new THREE.MeshStandardMaterial({ color: '#1d1e20', roughness: 0.78, side: two }),
    gap: new THREE.MeshStandardMaterial({ color: '#070707', roughness: 0.9, side: two }),
    chrome: new THREE.MeshStandardMaterial({ color: '#d9dcdf', roughness: 0.1, metalness: 1 }),
    glass: new THREE.MeshPhysicalMaterial({ color: '#0a1016', roughness: 0.04, metalness: 0.3, clearcoat: 1, envMapIntensity: 1.3, side: two }),
    grille: new THREE.MeshStandardMaterial({ map: grilleTexture(GRILLE_TEXT[spec.style]), roughness: 0.55, metalness: 0.3 }),
    head: new THREE.MeshStandardMaterial({ color: '#fffbe8', emissive: '#fff4d0', emissiveIntensity: 1.6, roughness: 0.1 }),
    drl: new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: '#e8f2ff', emissiveIntensity: 2.2 }),
    tail: new THREE.MeshStandardMaterial({ color: '#5a0707', emissive: '#ff1a0a', emissiveIntensity: 0.6, roughness: 0.3 }),
    amber: new THREE.MeshStandardMaterial({ color: '#b35a00', emissive: '#ff8a00', emissiveIntensity: 0.5 }),
    red: new THREE.MeshStandardMaterial({ color: '#c3141a', roughness: 0.4 }),
    accent: new THREE.MeshStandardMaterial({ color: '#f2f2f2', roughness: 0.4, side: two }),
    rubber: new THREE.MeshStandardMaterial({ color: '#1c1b1a', roughness: 0.93 }),
    rim: new THREE.MeshStandardMaterial({ color: '#2b2e31', roughness: 0.4, metalness: 0.8 }),
    rimLip: new THREE.MeshStandardMaterial({ color: '#8d9297', roughness: 0.3, metalness: 0.9 }),
  };
}

export function buildCar(spec, hex, finish) {
  const paint = paintMaterial(hex, finish);
  const materials = materialsFor(spec, paint);
  const root = new THREE.Group();
  const b = createBuilder();
  STYLES[spec.style](b, spec);
  const body = b.build(materials);
  root.add(body);
  const wheels = [];
  for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    const w = buildWheel(materials, spec, RIMS[spec.style]);
    w.group.position.set((sx * spec.track) / 2, spec.wheelR, (sz * spec.wheelbase) / 2);
    root.add(w.group);
    wheels.push({ group: w.group, spin: w.spin });
  }
  const shadowBlob = new THREE.Mesh(
    new THREE.PlaneGeometry(spec.width * 1.15, spec.length * 1.05),
    new THREE.MeshBasicMaterial({ color: '#000000', transparent: true, opacity: 0.35, depthWrite: false }),
  );
  shadowBlob.rotation.x = -Math.PI / 2;
  shadowBlob.position.y = 0.05;
  root.add(shadowBlob);
  return { root, body, wheels, paint, materials, spec, shadowBlob };
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
