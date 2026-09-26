import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { TANK, BUSHES } from './catalog.mjs';
import { sandHeight, fbm, pathMask } from './patterns.mjs';
import { grassSpots, carpetSpots, covered, CLUMPS_PER_LEVEL } from './layout.mjs';

const BLADES = 18;

export function swayMaterial(params, amp) {
  const uTime = { value: 0 };
  const m = new THREE.MeshStandardMaterial(params);
  m.onBeforeCompile = shader => {
    shader.uniforms.uTime = uTime;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uTime;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vec3 swayAt = vec3(instanceMatrix[3]);
float swayH = max(transformed.y, 0.0) * length(instanceMatrix[1].xyz);
float swayK = swayH * swayH * ${amp.toFixed(3)};
transformed.x += sin(uTime * 1.1 + swayAt.x * 23.0 + swayAt.z * 17.0) * swayK / max(length(instanceMatrix[0].xyz), 0.001);
transformed.z += cos(uTime * 0.8 + swayAt.x * 11.0 - swayAt.z * 13.0) * swayK * 0.6 / max(length(instanceMatrix[2].xyz), 0.001);`);
  };
  m.customProgramCacheKey = () => `sway${amp}`;
  return { material: m, uTime };
}

function gradient(geo, height, base, tip) {
  const p = geo.attributes.position;
  const a = new THREE.Color(base);
  const b = new THREE.Color(tip);
  const c = new THREE.Color();
  const colors = [];
  for (let i = 0; i < p.count; i++) {
    c.copy(a).lerp(b, Math.min(1, Math.max(0, p.getY(i) / height)));
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geo;
}

function bladeGeometry() {
  const geo = new THREE.PlaneGeometry(0.0032, 1, 1, 10);
  geo.translate(0, 0.5, 0);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const t = p.getY(i);
    p.setX(i, p.getX(i) * (1 - t * 0.85) + 0.012 * t * t);
  }
  geo.computeVertexNormals();
  return gradient(geo, 1, '#1f4a1c', '#9bd862');
}

function tallyMatrix(mesh, i, x, y, z, rx, ry, rz, sx, sy, sz) {
  const m = new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'YXZ')),
    new THREE.Vector3(sx, sy, sz)
  );
  mesh.setMatrixAt(i, m);
  return m;
}

const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0);

export function buildGrass(scene) {
  const spots = grassSpots();
  const sway = swayMaterial({ vertexColors: true, roughness: 0.6, side: THREE.DoubleSide }, 0.5);
  const mesh = new THREE.InstancedMesh(bladeGeometry(), sway.material, spots.length * BLADES);
  const tint = new THREE.Color();
  const matrices = [];
  spots.forEach(({ x, z }, n) => {
    const depth = (z + TANK.depth / 2) / TANK.depth;
    for (let i = 0; i < BLADES; i++) {
      const k = n * BLADES + i;
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 0.02;
      const bx = x + Math.cos(a) * r;
      const bz = z + Math.sin(a) * r * 0.8;
      const h = (0.09 + (1 - depth) * 0.16) * (0.6 + Math.random() * 0.6);
      matrices.push(tallyMatrix(mesh, k, bx, sandHeight(bx, bz) - 0.003, bz, (Math.random() - 0.5) * 0.3, Math.random() * Math.PI * 2, (Math.random() - 0.5) * 0.3, 1, h, 1));
      mesh.setColorAt(k, tint.setHSL(0.27 + Math.random() * 0.06, 0.55, 0.45 + Math.random() * 0.2));
    }
  });
  mesh.receiveShadow = true;
  scene.add(mesh);
  return {
    set(level, path) {
      mesh.count = Math.min(spots.length, level * CLUMPS_PER_LEVEL) * BLADES;
      spots.forEach(({ x, z }, n) => {
        const off = path && pathMask(x, z) > 0.1;
        for (let i = 0; i < BLADES; i++) mesh.setMatrixAt(n * BLADES + i, off ? HIDDEN : matrices[n * BLADES + i]);
      });
      mesh.instanceMatrix.needsUpdate = true;
    },
    update(t) {
      sway.uTime.value = t;
    }
  };
}

function leafTuft() {
  const leaves = [];
  for (let i = 0; i < 12; i++) {
    const leaf = new THREE.CircleGeometry(0.0032 + Math.random() * 0.0012, 7);
    leaf.scale(1, 0.85, 1);
    leaf.rotateX(-Math.PI / 2 + (Math.random() - 0.5) * 1.1);
    leaf.rotateZ((Math.random() - 0.5) * 0.9);
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * 0.007;
    leaf.translate(Math.cos(a) * r, 0.002 + Math.random() * 0.007, Math.sin(a) * r);
    leaves.push(leaf);
  }
  const geo = mergeGeometries(leaves);
  return gradient(geo, 0.01, '#1f5216', '#7fc23e');
}

export function buildCarpet(scene) {
  const spots = carpetSpots();
  const mesh = new THREE.InstancedMesh(leafTuft(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, side: THREE.DoubleSide }), spots.length);
  const tint = new THREE.Color();
  const matrices = spots.map(({ x, z, turn }, i) => {
    const puff = fbm(x * 9 + 4, z * 9 + 4, 64, 21);
    const s = 0.9 + puff * 1.4;
    mesh.setColorAt(i, tint.setHSL(0.26 + puff * 0.05, 0.65, 0.34 + fbm(x * 30, z * 30, 64, 5) * 0.18));
    return tallyMatrix(mesh, i, x, sandHeight(x, z) - 0.002 + puff * puff * 0.016, z, 0, turn, 0, s, s * (0.8 + puff * 1.4), s);
  });
  mesh.receiveShadow = true;
  mesh.visible = false;
  scene.add(mesh);
  return {
    set(on, blockers, path) {
      mesh.visible = on;
      if (!on) return;
      spots.forEach(({ x, z }, i) => {
        const off = covered(x, z, blockers, 0.85) || (path && pathMask(x, z) > 0.05);
        mesh.setMatrixAt(i, off ? HIDDEN : matrices[i]);
      });
      mesh.instanceMatrix.needsUpdate = true;
    }
  };
}

const BUSH_COLORS = {
  red: ['#5a2a1e', '#c23b2e', '#ff7a5c'],
  green: ['#1f4a1d', '#4d9a33', '#a9dd63'],
  lime: ['#2d5a1c', '#79b83a', '#d4f07a']
};

function stemLeaf() {
  const geo = new THREE.CircleGeometry(1, 8);
  geo.scale(0.0022, 0.0075, 1);
  geo.translate(0, 0.0072, 0);
  geo.computeVertexNormals();
  return geo;
}

export function buildBushes(scene) {
  const group = new THREE.Group();
  for (const [kind, [base, mid, tip]] of Object.entries(BUSH_COLORS)) {
    const bushes = BUSHES.filter(b => b.kind === kind);
    const places = [];
    for (const b of bushes) {
      for (let s = 0; s < 70; s++) {
        const a = Math.random() * Math.PI * 2;
        const r = Math.sqrt(Math.random());
        const sx = b.x + Math.cos(a) * r * b.rx;
        const sz = b.z + Math.sin(a) * r * b.rz;
        const h = b.h * Math.sqrt(1 - r * r * 0.7) * (0.8 + Math.random() * 0.25);
        const y0 = sandHeight(sx, sz) - 0.004;
        const lean = (Math.random() - 0.5) * 0.3;
        for (let y = 0.01; y < h; y += 0.0065) {
          const t = y / h;
          const cx = sx + Math.cos(a) * lean * y;
          const cz = sz + Math.sin(a) * lean * y * 0.5;
          for (let k = 0; k < 4; k++) places.push({ x: cx, y: y0 + y, z: cz, t, yaw: (k / 4) * Math.PI * 2 + y * 90, s: 0.8 + 0.5 * Math.sin(t * Math.PI) });
        }
      }
    }
    const mesh = new THREE.InstancedMesh(stemLeaf(), new THREE.MeshStandardMaterial({ roughness: 0.6, side: THREE.DoubleSide }), places.length);
    const cBase = new THREE.Color(base);
    const cMid = new THREE.Color(mid);
    const cTip = new THREE.Color(tip);
    const c = new THREE.Color();
    places.forEach((p, i) => {
      tallyMatrix(mesh, i, p.x, p.y, p.z, 0.9 - p.t * 0.5, p.yaw, 0, p.s, p.s, p.s);
      if (p.t < 0.5) c.copy(cBase).lerp(cMid, p.t * 2);
      else c.copy(cMid).lerp(cTip, (p.t - 0.5) * 2);
      mesh.setColorAt(i, c.offsetHSL((Math.random() - 0.5) * 0.02, 0, (Math.random() - 0.5) * 0.08));
    });
    mesh.receiveShadow = true;
    group.add(mesh);
  }
  group.visible = false;
  scene.add(group);
  return {
    set(on) {
      group.visible = on;
    }
  };
}
