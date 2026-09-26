import * as THREE from 'three';
import { fishColor } from './patterns.mjs';

const SHARK = { id: 'shark', len: 0.21, height: 0.28, width: 0.26, pattern: 'shark', colors: ['#6d7a85'], fin: '#66737e', tail: 'shark', dorsal: 'shark' };

const bodies = new Map();

function bodyGeometry(f) {
  if (bodies.has(f.id)) return bodies.get(f.id);
  const geo = new THREE.SphereGeometry(1, 36, 24);
  const pos = geo.attributes.position;
  const colors = [];
  const c = new THREE.Color();
  const shark = f.pattern === 'shark';
  for (let i = 0; i < pos.count; i++) {
    const nx = pos.getX(i);
    const ny = pos.getY(i);
    const nz = pos.getZ(i);
    let taper = nx < 0 ? 1 - 0.62 * Math.pow(-nx, 1.6) : 1;
    if (shark && nx > 0.4) taper *= 1 - 0.5 * Math.pow((nx - 0.4) / 0.6, 2);
    const hump = 1 + (ny > 0 ? 0.12 * (1 - nx * nx) : 0);
    pos.setXYZ(i, nx * f.len / 2, ny * taper * hump * f.len * f.height / 2, nz * taper * f.len * f.width / 2);
    c.set(fishColor(f, nx, ny));
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  bodies.set(f.id, geo);
  return geo;
}

function shape(points, curves) {
  const s = new THREE.Shape();
  s.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    if (p.length === 4 && curves) s.quadraticCurveTo(p[0], p[1], p[2], p[3]);
    else s.lineTo(p[0], p[1]);
  }
  s.closePath();
  return s;
}

const TAILS = {
  fork: [[0, 0.04], [-0.36, 0.3], [-0.22, 0], [-0.36, -0.3], [0, -0.04]],
  round: [[0, 0.05], [-0.2, 0.2], [-0.36, 0, -0.2, -0.2], [0, -0.05]],
  fan: [[0, 0.05], [-0.42, 0.38], [-0.62, 0, -0.42, -0.38], [0, -0.05]],
  veil: [[0, 0.06], [-0.55, 0.6], [-0.95, 0.1, -0.7, -0.62], [0, -0.06]],
  shark: [[0, 0.04], [-0.12, 0.06], [-0.3, 0.34], [-0.2, 0.02], [-0.22, -0.16], [-0.02, -0.04]]
};

const DORSALS = {
  small: [[0.05, 0], [-0.1, 0.16], [-0.25, 0]],
  round: [[0.15, 0], [-0.05, 0.24, -0.3, -0.02]],
  long: [[0.3, 0], [-0.05, 0.17], [-0.42, 0.06], [-0.36, -0.04]],
  angel: [[0.1, 0], [-0.28, 0.6], [-0.34, 0.55], [-0.3, -0.05]],
  shark: [[0.12, 0], [-0.06, 0.24], [-0.12, 0.02]]
};

function finGeometry(points, len) {
  const g = new THREE.ShapeGeometry(shape(points.map(p => p.map(v => v * len)), true), 8);
  return g;
}

export function buildFish(f) {
  const def = f === 'shark' ? SHARK : f;
  const L = def.len;
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  const skin = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: def.pattern === 'neon' ? 0.25 : 0.45, metalness: def.pattern === 'neon' ? 0.35 : 0.08 });
  const finMat = new THREE.MeshStandardMaterial({ color: def.fin, transparent: def.pattern !== 'shark', opacity: 0.78, side: THREE.DoubleSide, roughness: 0.5 });
  const trunk = new THREE.Mesh(bodyGeometry(def), skin);
  trunk.castShadow = true;
  body.add(trunk);

  const top = (L * def.height) / 2 * 0.86;
  const dorsal = new THREE.Mesh(finGeometry(DORSALS[def.dorsal], L), finMat);
  dorsal.position.y = top;
  body.add(dorsal);
  if (def.dorsal === 'angel' || def.dorsal === 'long') {
    const anal = dorsal.clone();
    anal.scale.y = -0.8;
    anal.position.y = -top;
    body.add(anal);
  }

  const tail = new THREE.Group();
  tail.position.x = -L / 2 * 0.92;
  const tailMesh = new THREE.Mesh(finGeometry(TAILS[def.tail], L), finMat);
  tailMesh.castShadow = true;
  tail.add(tailMesh);
  body.add(tail);

  const pectorals = [-1, 1].map(side => {
    const p = new THREE.Mesh(finGeometry(def.pattern === 'shark' ? [[0, 0], [-0.12, -0.22], [-0.2, -0.2]] : [[0, 0], [-0.14, 0.06], [-0.16, -0.04]], L), finMat);
    p.position.set(L * 0.18, -L * def.height * 0.12, side * L * def.width * 0.34);
    p.rotation.x = side * (def.pattern === 'shark' ? 0.9 : 0.5);
    body.add(p);
    return p;
  });

  const eyeMat = new THREE.MeshStandardMaterial({ color: '#050505', roughness: 0.1, metalness: 0.4 });
  const ringMat = new THREE.MeshStandardMaterial({ color: def.pattern === 'shark' ? '#2c3136' : '#f4f1e6', roughness: 0.4 });
  for (const side of [-1, 1]) {
    const ex = L * (def.pattern === 'shark' ? 0.36 : 0.3);
    const ey = L * def.height * (def.pattern === 'shark' ? 0.08 : 0.12);
    const ez = side * L * def.width * (def.pattern === 'shark' ? 0.2 : 0.3);
    const ring = new THREE.Mesh(new THREE.SphereGeometry(L * 0.075, 12, 10), ringMat);
    ring.position.set(ex, ey, ez);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(L * 0.05, 12, 10), eyeMat);
    eye.position.set(ex + L * 0.012, ey, ez + side * L * 0.035);
    body.add(ring, eye);
  }

  root.rotation.order = 'YZX';
  let phase = Math.random() * 10;
  return {
    root,
    animate(dt, speed) {
      phase += dt * (5 + speed * 55);
      tail.rotation.y = Math.sin(phase) * 0.55;
      body.rotation.y = Math.sin(phase + 1.2) * 0.07;
      pectorals[0].rotation.y = 0.3 + Math.sin(phase * 0.6) * 0.3;
      pectorals[1].rotation.y = 0.3 + Math.sin(phase * 0.6 + 1) * 0.3;
    }
  };
}
