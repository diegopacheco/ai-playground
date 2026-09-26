import * as THREE from 'three';
import { TANK } from './catalog.mjs';
import { sandHeight, fbm, cellEdge } from './patterns.mjs';

export const RACK = { width: TANK.width + 0.1, height: 0.78, depth: TANK.depth + 0.1 };
const GLASS = 0.008;
const IW = TANK.width - GLASS * 2;
const ID = TANK.depth - GLASS * 2;

function box(w, h, d, material, x, y, z) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

export function buildRoom(scene) {
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(6, 64),
    new THREE.MeshStandardMaterial({ color: '#221e1b', roughness: 0.92 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -RACK.height;
  floor.receiveShadow = true;
  scene.add(floor);
  const wall = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 6),
    new THREE.MeshStandardMaterial({ color: '#15191b', roughness: 1 })
  );
  wall.position.set(0, 2.2, -0.75);
  wall.receiveShadow = true;
  scene.add(wall);
}

export function buildRack(scene) {
  const material = new THREE.MeshStandardMaterial();
  const metal = new THREE.MeshStandardMaterial({ color: '#c9ccd0', metalness: 1, roughness: 0.25 });
  const shadowy = new THREE.MeshStandardMaterial({ color: '#0d0d0e', roughness: 1 });
  const g = new THREE.Group();
  const { width: w, height: h, depth: d } = RACK;
  const kick = 0.05;
  g.add(box(w, 0.035, d + 0.02, material, 0, -0.0175, 0));
  g.add(box(w - 0.02, h - 0.035 - kick, d, material, 0, -0.035 - (h - 0.035 - kick) / 2, 0));
  g.add(box(w - 0.08, kick, d - 0.06, shadowy, 0, -h + kick / 2, -0.02));
  const doorH = h - 0.035 - kick - 0.05;
  const doorW = (w - 0.02) / 2 - 0.03;
  for (const side of [-1, 1]) {
    const door = box(doorW, doorH, 0.018, material, side * (doorW / 2 + 0.012), -0.06 - doorH / 2, d / 2 + 0.009);
    g.add(door);
    g.add(box(0.012, 0.16, 0.012, metal, side * 0.035, -0.06 - doorH / 2, d / 2 + 0.03));
  }
  scene.add(g);
  return material;
}

function glassPane(w, h, d, material, x, y, z) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  m.position.set(x, y, z);
  m.renderOrder = 3;
  return m;
}

function sandGeometry() {
  const geo = new THREE.PlaneGeometry(IW, ID, 96, 40);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colors = [];
  const light = new THREE.Color('#e3d2ab');
  const dark = new THREE.Color('#a58b62');
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    pos.setY(i, sandHeight(x, z));
    c.copy(dark).lerp(light, 0.35 + 0.65 * fbm(x * 40 + 5, z * 40 + 5, 64, 2));
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

function sandSide(x0, z0, x1, z1, material) {
  const n = 60;
  const verts = [];
  const idx = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const x = x0 + (x1 - x0) * t;
    const z = z0 + (z1 - z0) * t;
    verts.push(x, GLASS, z, x, sandHeight(x, z), z);
    if (i < n) idx.push(i * 2, i * 2 + 2, i * 2 + 1, i * 2 + 1, i * 2 + 2, i * 2 + 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, material);
}

function causticTexture(seed) {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const e = cellEdge(x / size + seed, y / size, 6);
      const v = Math.pow(Math.max(0, 1 - e * 4), 5) * 255;
      const i = (y * size + x) * 4;
      img.data[i] = v * 0.85;
      img.data[i + 1] = v;
      img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 1.3);
  return tex;
}

function buildWaterSurface() {
  const geo = new THREE.PlaneGeometry(IW, ID, 48, 20);
  geo.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: '#bfeff5', transparent: true, opacity: 0.32, roughness: 0.04, metalness: 0.2, side: THREE.DoubleSide, depthWrite: false
  }));
  mesh.position.y = TANK.water;
  mesh.renderOrder = 2;
  const base = Float32Array.from(geo.attributes.position.array);
  return {
    mesh,
    update(t) {
      const p = geo.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const x = base[i * 3];
        const z = base[i * 3 + 2];
        p.setY(i, 0.0025 * Math.sin(x * 22 + t * 1.7) + 0.002 * Math.sin(z * 31 - t * 2.3) + 0.0015 * Math.sin((x + z) * 40 + t * 3.1));
      }
      p.needsUpdate = true;
      geo.computeVertexNormals();
    }
  };
}

function buildBubbles(onSurface) {
  const count = 48;
  const origin = { x: IW / 2 - 0.04, z: -ID / 2 + 0.04 };
  const mesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 10, 8),
    new THREE.MeshStandardMaterial({ color: '#e8fbff', transparent: true, opacity: 0.55, roughness: 0.05, metalness: 0.3 }),
    count
  );
  mesh.renderOrder = 1;
  const items = [];
  const baseY = sandHeight(origin.x, origin.z);
  for (let i = 0; i < count; i++) {
    items.push({ y: baseY + Math.random() * (TANK.water - baseY), phase: Math.random() * 6, size: 0.002 + Math.random() * 0.004, speed: 0.12 + Math.random() * 0.08 });
  }
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const p = new THREE.Vector3();
  return {
    mesh,
    update(dt, t) {
      for (let i = 0; i < count; i++) {
        const b = items[i];
        b.y += b.speed * dt;
        if (b.y > TANK.water) {
          b.y = baseY;
          b.phase = Math.random() * 6;
          onSurface();
        }
        const r = (b.y - baseY) * 0.08;
        p.set(origin.x + Math.sin(t * 3 + b.phase) * r, b.y, origin.z + Math.cos(t * 2.4 + b.phase) * r);
        s.setScalar(b.size * (1 + (b.y - baseY) * 0.8));
        mesh.setMatrixAt(i, m.compose(p, q, s));
      }
      mesh.instanceMatrix.needsUpdate = true;
    }
  };
}

function buildAirline(group) {
  const pipe = new THREE.MeshStandardMaterial({ color: '#2f3a37', roughness: 0.6, transparent: true, opacity: 0.8 });
  const x = IW / 2 - 0.04;
  const z = -ID / 2 + 0.04;
  const y = sandHeight(x, z);
  const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, TANK.height - y, 8), pipe);
  tube.position.set(x + 0.012, y + (TANK.height - y) / 2, z - 0.008);
  const stone = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.014, 0.018, 16), new THREE.MeshStandardMaterial({ color: '#7d8a86', roughness: 1 }));
  stone.position.set(x, y + 0.006, z);
  group.add(tube, stone);
}

function buildLamp(group) {
  const body = new THREE.MeshStandardMaterial({ color: '#1b1d1f', metalness: 0.6, roughness: 0.35 });
  const glow = new THREE.MeshStandardMaterial({ color: '#fff6e2', emissive: '#fff3dc', emissiveIntensity: 2.2 });
  const y = TANK.height + 0.07;
  group.add(box(TANK.width * 0.86, 0.014, 0.1, body, 0, y, 0));
  const strip = new THREE.Mesh(new THREE.PlaneGeometry(TANK.width * 0.84, 0.08), glow);
  strip.rotation.x = Math.PI / 2;
  strip.position.set(0, y - 0.0075, 0);
  group.add(strip);
  for (const side of [-1, 1]) {
    const leg = box(0.008, 0.075, 0.008, body, side * (TANK.width / 2 - 0.01), TANK.height + 0.033, 0);
    leg.rotation.z = side * 0.35;
    leg.position.x -= side * 0.012;
    group.add(leg);
  }
}

export function buildTank(scene, onBubble) {
  const group = new THREE.Group();
  const glass = new THREE.MeshPhysicalMaterial({
    color: '#d9f2f0', transparent: true, opacity: 0.1, roughness: 0.02, metalness: 0, clearcoat: 1, depthWrite: false
  });
  const edge = new THREE.MeshStandardMaterial({ color: '#8fd3c3', transparent: true, opacity: 0.35, roughness: 0.1, depthWrite: false });
  const { width: w, height: h, depth: d } = TANK;
  group.add(glassPane(w, GLASS, d, glass, 0, GLASS / 2, 0));
  group.add(glassPane(w, h, GLASS, glass, 0, h / 2, d / 2 - GLASS / 2));
  group.add(glassPane(w, h, GLASS, glass, 0, h / 2, -d / 2 + GLASS / 2));
  group.add(glassPane(GLASS, h, d - GLASS * 2, glass, -w / 2 + GLASS / 2, h / 2, 0));
  group.add(glassPane(GLASS, h, d - GLASS * 2, glass, w / 2 - GLASS / 2, h / 2, 0));
  for (const x of [-1, 1]) {
    for (const z of [-1, 1]) group.add(glassPane(GLASS * 1.2, h, GLASS * 1.2, edge, x * (w / 2 - GLASS / 2), h / 2, z * (d / 2 - GLASS / 2)));
    group.add(glassPane(GLASS * 1.2, GLASS * 1.2, d, edge, x * (w / 2 - GLASS / 2), h, 0));
  }
  for (const z of [-1, 1]) group.add(glassPane(w, GLASS * 1.2, GLASS * 1.2, edge, 0, h, z * (d / 2 - GLASS / 2)));

  const backdrop = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ color: '#0f3a4a' })
  );
  backdrop.position.set(0, h / 2, -d / 2 - 0.002);
  group.add(backdrop);

  const sandMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 });
  const sand = new THREE.Mesh(sandGeometry(), sandMat);
  sand.receiveShadow = true;
  group.add(sand);
  const sideMat = new THREE.MeshStandardMaterial({ color: '#b89c70', roughness: 1, side: THREE.DoubleSide });
  group.add(sandSide(-IW / 2, ID / 2, IW / 2, ID / 2, sideMat));
  group.add(sandSide(-IW / 2, -ID / 2, IW / 2, -ID / 2, sideMat));
  group.add(sandSide(-IW / 2, -ID / 2, -IW / 2, ID / 2, sideMat));
  group.add(sandSide(IW / 2, -ID / 2, IW / 2, ID / 2, sideMat));

  const caustics = [0, 0.37].map(seed => {
    const tex = causticTexture(seed);
    const mesh = new THREE.Mesh(sand.geometry, new THREE.MeshBasicMaterial({
      map: tex, transparent: true, opacity: 0.11, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    mesh.position.y = 0.0008;
    group.add(mesh);
    return tex;
  });

  const water = new THREE.Mesh(
    new THREE.BoxGeometry(IW - 0.002, TANK.water - GLASS, ID - 0.002),
    new THREE.MeshStandardMaterial({ color: '#2a8fa6', transparent: true, opacity: 0.13, roughness: 0.1, depthWrite: false })
  );
  water.position.y = GLASS + (TANK.water - GLASS) / 2;
  water.renderOrder = 2;
  group.add(water);

  const surface = buildWaterSurface();
  group.add(surface.mesh);
  const bubbles = buildBubbles(onBubble);
  group.add(bubbles.mesh);
  buildAirline(group);
  buildLamp(group);
  scene.add(group);

  return {
    update(dt, t) {
      caustics[0].offset.set(t * 0.012, t * 0.007);
      caustics[1].offset.set(-t * 0.009, t * 0.011);
      surface.update(t);
      bubbles.update(dt, t);
    }
  };
}

export function buildLights(scene) {
  scene.add(new THREE.HemisphereLight('#a8d8ff', '#2b2119', 0.55));
  const sun = new THREE.DirectionalLight('#fff3de', 2.6);
  sun.position.set(0.15, 2.2, 0.35);
  sun.target.position.set(0, 0, 0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -0.9;
  sun.shadow.camera.right = 0.9;
  sun.shadow.camera.top = 0.6;
  sun.shadow.camera.bottom = -0.6;
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 4;
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.01;
  scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight('#ffd9b0', 0.9);
  fill.position.set(1.5, 0.6, 2.5);
  scene.add(fill);
  const glow = new THREE.PointLight('#6fd3e6', 1.6, 2.2, 2);
  glow.position.set(0, 0.3, 0.6);
  scene.add(glow);
}
