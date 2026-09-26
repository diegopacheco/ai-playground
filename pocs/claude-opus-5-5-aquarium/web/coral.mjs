import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { noise } from './patterns.mjs';
import { swayMaterial } from './plants.mjs';

function colorByHeight(geo, top, base, tip) {
  const p = geo.attributes.position;
  const a = new THREE.Color(base);
  const b = new THREE.Color(tip);
  const c = new THREE.Color();
  const colors = [];
  for (let i = 0; i < p.count; i++) {
    c.copy(a).lerp(b, Math.min(1, Math.max(0, p.getY(i) / top)));
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geo;
}

function limb(parts, from, dir, len, r, depth) {
  const to = from.clone().addScaledVector(dir, len);
  const geo = new THREE.CylinderGeometry(r * 0.72, r, len, 9, 3);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const k = 1 + 0.18 * (noise(p.getX(i) * 4000, p.getY(i) * 4000 + p.getZ(i) * 3000, 64, 3) - 0.5);
    p.setXYZ(i, p.getX(i) * k, p.getY(i), p.getZ(i) * k);
  }
  geo.translate(0, len / 2, 0);
  geo.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir));
  geo.translate(from.x, from.y, from.z);
  parts.push(geo.toNonIndexed());
  if (depth === 0) {
    const tip = new THREE.SphereGeometry(r * 0.75, 8, 6);
    tip.translate(to.x, to.y, to.z);
    parts.push(tip.toNonIndexed());
    return;
  }
  const kids = depth > 1 ? 3 : 2;
  for (let k = 0; k < kids; k++) {
    const a = (k / kids) * Math.PI * 2 + depth + Math.random();
    const next = dir.clone().add(new THREE.Vector3(Math.cos(a) * 0.55, 0.45, Math.sin(a) * 0.55)).normalize();
    limb(parts, to, next, len * (0.72 + Math.random() * 0.15), r * 0.72, depth - 1);
  }
}

function staghorn(base, tip, scale) {
  const parts = [];
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    const dir = new THREE.Vector3(Math.cos(a) * 0.35, 1, Math.sin(a) * 0.35).normalize();
    limb(parts, new THREE.Vector3(Math.cos(a) * 0.006, 0, Math.sin(a) * 0.006), dir, 0.022 * scale, 0.0045 * scale, 3);
  }
  return colorByHeight(mergeGeometries(parts), 0.08 * scale, base, tip);
}

function brainCoral(r) {
  const geo = new THREE.SphereGeometry(r, 72, 36, 0, Math.PI * 2, 0, Math.PI / 2);
  const p = geo.attributes.position;
  const ridge = new THREE.Color('#b89a4a');
  const groove = new THREE.Color('#3f5a22');
  const c = new THREE.Color();
  const colors = [];
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3(p.getX(i), p.getY(i), p.getZ(i)).divideScalar(r);
    const n = noise(v.x * 16 + 11, v.z * 16 + v.y * 9 + 11, 64, 4);
    const line = Math.min(1, Math.abs(n - 0.5) * 7);
    const k = 1 - 0.07 * (1 - line);
    p.setXYZ(i, v.x * r * k, v.y * r * 0.72 * k, v.z * r * k);
    c.copy(groove).lerp(ridge, line);
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

function fanTexture() {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.strokeStyle = '#ffffff';
  ctx.lineCap = 'round';
  const grow = (x, y, a, len, w, depth) => {
    const nx = x + Math.cos(a) * len;
    const ny = y - Math.sin(a) * len;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    if (depth === 0) return;
    grow(nx, ny, a + 0.35 + Math.random() * 0.2, len * 0.82, w * 0.8, depth - 1);
    grow(nx, ny, a - 0.35 - Math.random() * 0.2, len * 0.82, w * 0.8, depth - 1);
  };
  grow(size / 2, size, Math.PI / 2, 46, 6, 7);
  ctx.lineWidth = 1;
  for (let i = 0; i < 260; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size * 0.8;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (Math.random() - 0.5) * 14, y + (Math.random() - 0.5) * 14);
    ctx.stroke();
  }
  return new THREE.CanvasTexture(canvas);
}

function seaFan() {
  const geo = new THREE.PlaneGeometry(0.1, 0.1, 10, 10);
  geo.translate(0, 0.05, 0);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) p.setZ(i, 0.12 * p.getX(i) * p.getX(i) * 10 - 0.002);
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: '#8e2fa0', alphaMap: fanTexture(), alphaTest: 0.4, side: THREE.DoubleSide, roughness: 0.8 }));
  m.castShadow = true;
  return m;
}

function tubeSponge(r, h) {
  const pts = [[r * 0.8, 0], [r, h * 0.5], [r * 1.12, h], [r * 0.85, h], [r * 0.7, h * 0.2]].map(([x, y]) => new THREE.Vector2(x, y));
  const geo = new THREE.LatheGeometry(pts, 18);
  return colorByHeight(geo, h, '#b5521c', '#ffb23a');
}

function mushroomCoral(r) {
  const pts = [[0.004, 0], [0.005, 0.02], [r * 0.6, 0.026], [r, 0.03], [r * 0.9, 0.034], [0, 0.036]].map(([x, y]) => new THREE.Vector2(x, y));
  const geo = new THREE.LatheGeometry(pts, 32);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const a = Math.atan2(p.getZ(i), p.getX(i));
    const rim = Math.hypot(p.getX(i), p.getZ(i)) / r;
    p.setY(i, p.getY(i) + 0.0025 * Math.sin(a * 9) * rim * rim);
  }
  geo.computeVertexNormals();
  return colorByHeight(geo, 0.036, '#4b6a3a', '#8fc06a');
}

function anemone() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.014, 0.018, 20), new THREE.MeshStandardMaterial({ color: '#c8634a', roughness: 0.7 }));
  body.position.y = 0.009;
  g.add(body);
  const tentacle = new THREE.CapsuleGeometry(0.0022, 0.022, 4, 6);
  tentacle.translate(0, 0.013, 0);
  colorByHeight(tentacle, 0.026, '#a8e0a0', '#ff5fa8');
  const sway = swayMaterial({ vertexColors: true, roughness: 0.5 }, 18);
  const count = 70;
  const mesh = new THREE.InstancedMesh(tentacle, sway.material, count);
  const m = new THREE.Matrix4();
  for (let i = 0; i < count; i++) {
    const a = i * 2.4;
    const r = 0.017 * Math.sqrt((i + 1) / count);
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3 + r * 30, a, 0, 'YXZ'));
    mesh.setMatrixAt(i, m.compose(new THREE.Vector3(Math.cos(a) * r, 0.017, Math.sin(a) * r), q, new THREE.Vector3(1, 0.8 + Math.random() * 0.4, 1)));
  }
  g.add(mesh);
  return { group: g, time: sway.uTime };
}

export function coral() {
  const g = new THREE.Group();
  const add = (geo, x, z, ry = 0, extra = {}) => {
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, ...extra }));
    m.position.set(x, -0.004, z);
    m.rotation.y = ry;
    m.castShadow = true;
    m.receiveShadow = true;
    g.add(m);
    return m;
  };
  add(staghorn('#6e3a8c', '#f2a6e0', 1), -0.05, -0.03);
  add(staghorn('#b85a1e', '#ffd07a', 0.8), 0.045, -0.045, 1);
  add(staghorn('#2f6fa0', '#9fe4ff', 0.65), -0.075, 0.035, 2);
  add(brainCoral(0.032), 0.02, 0.025);
  add(mushroomCoral(0.024), 0.07, 0.035);
  add(mushroomCoral(0.018), -0.02, 0.07, 1);
  for (const [x, z, h] of [[-0.005, -0.07, 0.05], [0.01, -0.06, 0.035], [-0.018, -0.058, 0.028]]) add(tubeSponge(0.008, h), x, z);
  const fan = seaFan();
  fan.position.set(0.0, -0.004, -0.085);
  fan.rotation.y = 0.2;
  g.add(fan);
  const anem = anemone();
  anem.group.position.set(-0.055, -0.004, 0.065);
  g.add(anem.group);
  return {
    group: g,
    update(t) {
      anem.time.value = t;
      fan.rotation.x = Math.sin(t * 0.7) * 0.06;
    }
  };
}
