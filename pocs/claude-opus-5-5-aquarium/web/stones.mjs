import * as THREE from 'three';
import { STONES } from './catalog.mjs';
import { sandHeight, noise, fbm } from './patterns.mjs';

function dragonStone(r, h, seed) {
  const geo = new THREE.CylinderGeometry(1, 1, 1, 30, 26);
  geo.translate(0, 0.5, 0);
  const p = geo.attributes.position;
  const light = new THREE.Color('#a4a9ab');
  const dark = new THREE.Color('#3b3e42');
  const base = new THREE.Color('#575b60');
  const c = new THREE.Color();
  const colors = [];
  const lean = (noise(seed, 3, 64, 4) - 0.5) * 0.5;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i);
    const z = p.getZ(i);
    const t = p.getY(i);
    const ridge = noise(x * 2.2 + seed * 7, z * 2.2 + t * 5, 64, seed);
    const crease = Math.pow(Math.abs(Math.sin((t * 6 + x * 1.5 + ridge * 2.2) * Math.PI)), 8);
    const chip = noise(x * 9 + seed, z * 9 + t * 14, 64, seed + 5);
    const shelf = Math.floor(t * 5 + ridge * 2) / 5;
    const profile = Math.pow(1 - t, 0.75) * (0.65 + 0.7 * ridge) * (1 - 0.28 * crease) * (0.88 + 0.24 * chip) * (1 - 0.12 * (t - shelf) * 5);
    const peak = t * t * lean;
    p.setXYZ(i, (x * profile + peak) * r, t * h * (0.92 + 0.16 * ridge), z * profile * r * 0.8);
    const vein = fbm(x * 3 + seed, t * 9 + z * 3, 64, seed + 2);
    c.copy(base).lerp(light, Math.max(0, vein - 0.45) * 2.2).lerp(dark, crease * 0.8 + (1 - t) * 0.15);
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

export function buildStones(scene) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.82 });
  STONES.forEach(([x, z, r, h], i) => {
    const pieces = [[0, 0, r, h], [r * 0.9, r * 0.5, r * 0.45, h * 0.4], [-r * 0.7, r * 0.6, r * 0.35, h * 0.28]];
    for (const [dx, dz, pr, ph] of pieces) {
      const m = new THREE.Mesh(dragonStone(pr, ph, i * 3 + pr * 100), material);
      m.position.set(x + dx, sandHeight(x + dx, z + dz) - 0.012, z + dz);
      m.rotation.y = i * 1.7 + dx * 20;
      m.castShadow = true;
      m.receiveShadow = true;
      group.add(m);
    }
  });
  group.visible = false;
  scene.add(group);
  return {
    set(on) {
      group.visible = on;
    }
  };
}
