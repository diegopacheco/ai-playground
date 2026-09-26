import * as THREE from 'three';
import { TANK, DECOR } from './catalog.mjs';
import { sandHeight, noise } from './patterns.mjs';
import { weathered } from './weather.mjs';
import { grassSpots, CLUMPS_PER_LEVEL } from './layout.mjs';

function mat(color, roughness = 0.85, metalness = 0, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });
}

function mesh(geo, material, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, material);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

function cyl(rt, rb, h, seg = 16) {
  return new THREE.CylinderGeometry(rt, rb, h, seg);
}

function sandMound(rx, rz, h, x = 0, z = 0) {
  const geo = new THREE.SphereGeometry(1, 28, 10, 0, Math.PI * 2, 0, Math.PI / 2);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const k = 0.8 + 0.4 * noise(p.getX(i) * 3 + 5, p.getZ(i) * 3 + 5, 64, 9);
    p.setXYZ(i, p.getX(i) * rx * k, p.getY(i) * h * k, p.getZ(i) * rz * k);
  }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, mat('#cdb991', 1));
  m.position.set(x, -0.004, z);
  m.receiveShadow = true;
  return m;
}

function hullGeometry(L, B, D) {
  const nx = 48;
  const ny = 16;
  const verts = [];
  const idx = [];
  const width = t => B * Math.sqrt(Math.max(0, 1 - Math.pow(Math.abs(t), t > 0 ? 2.2 : 6)));
  const sheer = t => D + 0.022 * t * t + (t < 0 ? 0.012 * t * t : 0);
  for (let i = 0; i <= nx; i++) {
    const t = -1 + (2 * i) / nx;
    const w = width(t);
    const h = sheer(t);
    for (let j = 0; j <= ny; j++) {
      const a = (Math.PI * j) / ny;
      verts.push((t * L) / 2, h * (1 - Math.pow(Math.sin(a), 0.5)), w * Math.cos(a));
    }
  }
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const a = i * (ny + 1) + j;
      const b = a + ny + 1;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const hull = new THREE.BufferGeometry();
  hull.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  hull.setIndex(idx);
  hull.computeVertexNormals();
  const deckVerts = [];
  const deckIdx = [];
  for (let i = 0; i <= nx; i++) {
    const t = -1 + (2 * i) / nx;
    const w = width(t) * 0.96;
    const y = sheer(t) - 0.004;
    deckVerts.push((t * L) / 2, y, w, (t * L) / 2, y, -w);
    if (i < nx) deckIdx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
  }
  const deck = new THREE.BufferGeometry();
  deck.setAttribute('position', new THREE.Float32BufferAttribute(deckVerts, 3));
  deck.setIndex(deckIdx);
  deck.computeVertexNormals();
  return { hull, deck, width, sheer };
}

function tatteredSail(w, h, material) {
  const geo = new THREE.PlaneGeometry(w, h, 10, 8);
  const p = geo.attributes.position;
  for (let i = p.count - 1; i >= 0; i--) {
    const x = p.getX(i);
    const y = p.getY(i);
    p.setZ(i, 0.008 * Math.sin(x * 60) + 0.006 * noise(x * 80 + 3, y * 80, 64, 5));
    if (y < -h / 2 + 0.001) p.setY(i, y + 0.012 * noise(x * 120, 1, 64, 2));
  }
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, material);
}

function ship() {
  const g = new THREE.Group();
  const L = 0.27;
  const { hull, deck, width, sheer } = hullGeometry(L, 0.045, 0.07);
  const wood = weathered('#5d4631', { rust: 0.05, algae: 0.75, planks: 190, roughness: 0.95 });
  const deckWood = weathered('#6e563c', { rust: 0.05, algae: 0.85, roughness: 0.95 });
  const dark = mat('#120d09', 1);
  const hullMesh = mesh(hull, wood);
  hullMesh.material.side = THREE.DoubleSide;
  g.add(hullMesh, mesh(deck, deckWood));
  const sternT = -0.78;
  const sternX = (sternT * L) / 2;
  const castleW = width(sternT) * 1.9;
  g.add(mesh(new THREE.BoxGeometry(0.055, 0.03, castleW), wood, sternX + 0.012, sheer(sternT) + 0.01, 0));
  g.add(mesh(new THREE.BoxGeometry(0.05, 0.004, castleW + 0.006), deckWood, sternX + 0.012, sheer(sternT) + 0.026, 0));
  for (const z of [-1, 1]) {
    for (let i = 0; i < 5; i++) {
      const t = -0.45 + i * 0.2;
      const port = mesh(new THREE.BoxGeometry(0.009, 0.008, 0.002), dark, (t * L) / 2, sheer(t) - 0.016, z * (width(t) + 0.0005));
      g.add(port);
    }
  }
  const hole = mesh(new THREE.CircleGeometry(0.016, 10), dark, 0.035, 0.03, width(0.26) + 0.001);
  hole.scale.set(1.4, 0.8, 1);
  g.add(hole);
  const spar = weathered('#3b2b1d', { rust: 0, algae: 0.6, roughness: 1 });
  const main = mesh(cyl(0.0035, 0.0045, 0.16, 10), spar, 0.0, sheer(0) + 0.08, 0);
  const yard = mesh(cyl(0.0025, 0.0025, 0.09, 8), spar, 0.0, sheer(0) + 0.12, 0);
  yard.rotation.x = Math.PI / 2 - 0.3;
  const fore = mesh(cyl(0.0035, 0.0045, 0.06, 10), spar, 0.075, sheer(0.55) + 0.03, 0);
  fore.rotation.z = -0.2;
  g.add(main, yard, fore);
  const sail = tatteredSail(0.07, 0.05, weathered('#a39576', { rust: 0, algae: 0.5, roughness: 1, extra: { side: THREE.DoubleSide } }));
  sail.position.set(0.004, sheer(0) + 0.095, 0);
  sail.rotation.y = Math.PI / 2 - 0.3;
  g.add(sail);
  const bowsprit = mesh(cyl(0.0025, 0.003, 0.07, 8), spar, L / 2 + 0.01, sheer(1) + 0.012, 0);
  bowsprit.rotation.z = -Math.PI / 2 + 0.35;
  g.add(bowsprit);
  g.rotation.set(0.4, 0.3, 0.05);
  g.position.y = -0.012;
  const root = new THREE.Group();
  const broken = mesh(cyl(0.0035, 0.004, 0.09, 8), spar, -0.02, 0.006, 0.09);
  broken.rotation.set(0, 0.5, Math.PI / 2);
  root.add(g, broken, sandMound(0.15, 0.05, 0.02, 0, 0.035));
  return root;
}

function carBodyShape() {
  const s = new THREE.Shape();
  s.moveTo(-0.1, 0.022);
  s.lineTo(-0.1, 0.042);
  s.quadraticCurveTo(-0.099, 0.053, -0.088, 0.054);
  s.lineTo(-0.052, 0.056);
  s.lineTo(0.046, 0.056);
  s.quadraticCurveTo(0.075, 0.055, 0.092, 0.05);
  s.quadraticCurveTo(0.101, 0.047, 0.101, 0.036);
  s.lineTo(0.1, 0.022);
  s.lineTo(0.085, 0.022);
  s.absarc(0.062, 0.02, 0.023, 0, Math.PI, false);
  s.lineTo(-0.039, 0.022);
  s.absarc(-0.062, 0.02, 0.023, 0, Math.PI, false);
  s.lineTo(-0.1, 0.022);
  return s;
}

function cabinShape(top) {
  const s = new THREE.Shape();
  s.moveTo(-0.05, 0.055);
  s.lineTo(-0.034, top - 0.004);
  s.quadraticCurveTo(-0.029, top, -0.02, top);
  s.lineTo(0.012, top);
  s.quadraticCurveTo(0.02, top, 0.024, top - 0.004);
  s.lineTo(0.044, 0.055);
  return s;
}

function strut(from, to, z, material) {
  const len = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const m = mesh(new THREE.BoxGeometry(0.005, len, 0.004), material, (from[0] + to[0]) / 2, (from[1] + to[1]) / 2, z);
  m.rotation.z = Math.atan2(-(to[0] - from[0]), to[1] - from[1]);
  return m;
}

function tire(material) {
  const pts = [[0.011, -0.007], [0.0165, -0.0072], [0.019, -0.0045], [0.0195, 0], [0.019, 0.0045], [0.0165, 0.0072], [0.011, 0.007]].map(([r, y]) => new THREE.Vector2(r, y));
  const geo = new THREE.LatheGeometry(pts, 28);
  geo.rotateX(Math.PI / 2);
  return mesh(geo, material);
}

function car() {
  const g = new THREE.Group();
  const paint = weathered('#5f8c86', { rust: 0.8, algae: 0.55, roughness: 0.55, metalness: 0.35 });
  const chrome = weathered('#cfcfca', { rust: 0.55, algae: 0.3, roughness: 0.22, metalness: 1 });
  const rubber = weathered('#1b1b1b', { rust: 0, algae: 0.45, roughness: 0.95 });
  const glass = new THREE.MeshStandardMaterial({ color: '#0c1b1f', roughness: 0.06, metalness: 0.9, transparent: true, opacity: 0.88 });
  const depth = 0.07;
  const body = new THREE.ExtrudeGeometry(carBodyShape(), { depth, curveSegments: 20, bevelEnabled: true, bevelThickness: 0.008, bevelSize: 0.006, bevelSegments: 5 });
  body.translate(0, 0, -depth / 2);
  g.add(mesh(body, paint));
  const cabin = new THREE.ExtrudeGeometry(cabinShape(0.088), { depth: 0.058, bevelEnabled: true, bevelThickness: 0.004, bevelSize: 0.003, bevelSegments: 3 });
  cabin.translate(0, 0, -0.029);
  g.add(mesh(cabin, glass));
  const roofShape = new THREE.Shape();
  roofShape.moveTo(-0.036, 0.084);
  roofShape.quadraticCurveTo(-0.03, 0.093, -0.02, 0.093);
  roofShape.lineTo(0.012, 0.093);
  roofShape.quadraticCurveTo(0.022, 0.093, 0.027, 0.084);
  const roof = new THREE.ExtrudeGeometry(roofShape, { depth: 0.064, bevelEnabled: true, bevelThickness: 0.003, bevelSize: 0.002, bevelSegments: 3 });
  roof.translate(0, 0, -0.032);
  g.add(mesh(roof, paint));
  for (const z of [-0.0335, 0.0335]) {
    g.add(strut([0.046, 0.056], [0.025, 0.086], z, paint));
    g.add(strut([-0.006, 0.056], [-0.006, 0.089], z, paint));
    g.add(strut([-0.052, 0.056], [-0.035, 0.086], z, paint));
    g.add(mesh(new THREE.BoxGeometry(0.15, 0.003, 0.002), chrome, 0, 0.045, z * 1.33));
  }
  for (const x of [-0.062, 0.062]) {
    for (const z of [-0.036, 0.036]) {
      const t = tire(rubber);
      t.position.set(x, 0.0195, z);
      const rim = mesh(cyl(0.011, 0.011, 0.012, 20), chrome, x, 0.0195, z);
      rim.rotation.x = Math.PI / 2;
      const cap = mesh(new THREE.SphereGeometry(0.006, 12, 8), chrome, x, 0.0195, z + Math.sign(z) * 0.006);
      cap.scale.z = 0.5;
      g.add(t, rim, cap);
    }
  }
  for (const x of [-0.106, 0.106]) {
    const bumper = mesh(new THREE.CapsuleGeometry(0.0045, 0.07, 6, 12), chrome, x, 0.027, 0);
    bumper.rotation.x = Math.PI / 2;
    g.add(bumper);
  }
  for (const z of [-0.024, 0.024]) {
    const ring = mesh(new THREE.TorusGeometry(0.0075, 0.0018, 8, 20), chrome, 0.1025, 0.043, z);
    ring.rotation.y = Math.PI / 2;
    const lens = mesh(new THREE.SphereGeometry(0.0072, 14, 10, 0, Math.PI), new THREE.MeshStandardMaterial({ color: '#d8e0d0', roughness: 0.1, metalness: 0.2 }), 0.1025, 0.043, z);
    lens.rotation.y = Math.PI / 2;
    lens.scale.z = 0.5;
    const tail = mesh(new THREE.BoxGeometry(0.004, 0.012, 0.01), mat('#6e1a14', 0.4), -0.106, 0.044, z);
    g.add(ring, lens, tail);
  }
  for (let i = 0; i < 4; i++) g.add(mesh(new THREE.BoxGeometry(0.003, 0.0022, 0.03), chrome, 0.1045, 0.028 + i * 0.0045, 0));
  g.rotation.set(0.12, -0.5, 0.07);
  g.position.y = -0.011;
  const root = new THREE.Group();
  root.add(g, sandMound(0.07, 0.05, 0.016, -0.02, 0.03));
  return root;
}

function wingGeometry(half, root, sign, thick = 0.0035) {
  const s = new THREE.Shape();
  const n = 18;
  const chord = t => Math.max(0.006, root * Math.sqrt(1 - t * t));
  s.moveTo(0, root * 0.35);
  for (let i = 1; i <= n; i++) {
    const t = (i / n) * 0.985;
    s.lineTo(sign * t * half, chord(t) * 0.35);
  }
  for (let i = n; i >= 0; i--) {
    const t = (i / n) * 0.985;
    s.lineTo(sign * t * half, -chord(t) * 0.65);
  }
  const geo = new THREE.ExtrudeGeometry(s, { depth: thick, bevelEnabled: true, bevelThickness: 0.0014, bevelSize: 0.0012, bevelSegments: 2 });
  geo.translate(0, 0, -thick / 2);
  geo.rotateX(-Math.PI / 2);
  geo.rotateY(-Math.PI / 2);
  return geo;
}

function roundel(z) {
  const g = new THREE.Group();
  [['#23365e', 0.012], ['#d9d4c3', 0.008], ['#7c231d', 0.004]].forEach(([c, r], i) => {
    const m = new THREE.Mesh(new THREE.CircleGeometry(r, 24), weathered(c, { rust: 0.3, algae: 0.4, roughness: 0.9 }));
    m.rotation.x = -Math.PI / 2;
    m.position.y = 0.0036 + i * 0.0002;
    g.add(m);
  });
  g.position.set(-0.006, 0, z);
  return g;
}

function plane() {
  const g = new THREE.Group();
  const olive = weathered('#556043', { rust: 0.55, algae: 0.7, roughness: 0.7, metalness: 0.35 });
  const metal = weathered('#8d8f8c', { rust: 0.7, algae: 0.4, roughness: 0.45, metalness: 0.8 });
  const glass = new THREE.MeshStandardMaterial({ color: '#16272c', roughness: 0.05, metalness: 0.8, transparent: true, opacity: 0.8 });
  const profile = [[0, -0.125], [0.003, -0.12], [0.007, -0.1], [0.012, -0.06], [0.017, -0.02], [0.02, 0.02], [0.0205, 0.06], [0.019, 0.09], [0.016, 0.104], [0, 0.105]];
  const fuselage = new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), 32);
  fuselage.rotateZ(-Math.PI / 2);
  fuselage.scale(1, 1.12, 0.92);
  g.add(mesh(fuselage, olive));
  const spinner = new THREE.LatheGeometry([[0.0135, 0], [0.012, 0.008], [0.008, 0.016], [0, 0.021]].map(([r, y]) => new THREE.Vector2(r, y)), 24);
  spinner.rotateZ(-Math.PI / 2);
  g.add(mesh(spinner, metal, 0.104, 0, 0));
  const bladeShape = new THREE.Shape();
  bladeShape.moveTo(-0.0035, 0);
  bladeShape.quadraticCurveTo(-0.0075, 0.03, -0.0025, 0.056);
  bladeShape.lineTo(0.0025, 0.056);
  bladeShape.quadraticCurveTo(0.0065, 0.03, 0.0035, 0);
  const bladeGeo = new THREE.ExtrudeGeometry(bladeShape, { depth: 0.0014, bevelEnabled: false });
  bladeGeo.rotateY(Math.PI / 2);
  for (let i = 0; i < 3; i++) {
    const holder = new THREE.Group();
    holder.position.x = 0.112;
    holder.rotation.x = (i * Math.PI * 2) / 3 + 0.4;
    const blade = mesh(bladeGeo, metal);
    blade.rotation.y = 0.35;
    if (i === 1) blade.rotation.z = -0.6;
    holder.add(blade);
    g.add(holder);
  }
  const right = mesh(wingGeometry(0.12, 0.052, 1), olive, 0.03, -0.01, 0.012);
  right.rotation.x = -0.06;
  right.add(roundel(0.07));
  const left = mesh(wingGeometry(0.12, 0.052, -1), olive, 0.03, -0.01, -0.012);
  left.rotation.x = 0.32;
  left.add(roundel(-0.07));
  g.add(right, left);
  g.add(mesh(wingGeometry(0.042, 0.024, 1, 0.002), olive, -0.108, 0.004, 0.002));
  const stub = mesh(wingGeometry(0.022, 0.024, -1, 0.002), olive, -0.108, 0.004, -0.002);
  g.add(stub);
  const finShape = new THREE.Shape();
  finShape.moveTo(-0.094, 0.006);
  finShape.quadraticCurveTo(-0.1, 0.036, -0.116, 0.04);
  finShape.quadraticCurveTo(-0.128, 0.036, -0.127, 0.004);
  const fin = new THREE.ExtrudeGeometry(finShape, { depth: 0.002, bevelEnabled: true, bevelThickness: 0.001, bevelSize: 0.001, bevelSegments: 2 });
  fin.translate(0, 0, -0.001);
  g.add(mesh(fin, olive));
  const canopy = mesh(new THREE.SphereGeometry(0.012, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), glass, 0.02, 0.016, 0);
  canopy.scale.set(2.3, 1.15, 0.85);
  g.add(canopy);
  const frame = mesh(new THREE.TorusGeometry(0.0105, 0.0012, 6, 20, Math.PI), metal, 0.028, 0.016, 0);
  frame.rotation.y = Math.PI / 2;
  g.add(frame);
  for (const z of [-1, 1]) {
    for (let i = 0; i < 3; i++) g.add(mesh(new THREE.BoxGeometry(0.005, 0.003, 0.004), metal, 0.078 - i * 0.008, 0.006, z * 0.0185));
  }
  g.rotation.set(0.12, 0.35, -0.16);
  g.position.y = 0.022;
  const root = new THREE.Group();
  const tailBit = mesh(wingGeometry(0.03, 0.02, 1, 0.002), olive, -0.13, 0.004, 0.07);
  tailBit.rotation.set(0.3, 1.2, 0.1);
  root.add(g, tailBit, sandMound(0.07, 0.05, 0.014, 0.07, 0.01));
  return root;
}

function chest() {
  const g = new THREE.Group();
  const wood = weathered('#5c3b1f', { rust: 0, algae: 0.5, planks: 160, roughness: 0.9 });
  const gold = mat('#e8b93a', 0.25, 1, { emissive: '#4a3208' });
  g.add(mesh(new THREE.BoxGeometry(0.08, 0.042, 0.052), wood, 0, 0.021, 0));
  for (const x of [-0.03, 0.03]) g.add(mesh(new THREE.BoxGeometry(0.006, 0.044, 0.054), gold, x, 0.021, 0));
  for (let i = 0; i < 14; i++) {
    const coin = mesh(cyl(0.006, 0.006, 0.0018, 12), gold, (Math.random() - 0.5) * 0.06, 0.042 + Math.random() * 0.008, (Math.random() - 0.5) * 0.036);
    coin.rotation.set(Math.random(), 0, Math.random());
    g.add(coin);
  }
  for (let i = 0; i < 6; i++) {
    const coin = mesh(cyl(0.006, 0.006, 0.0018, 12), gold, 0.03 + Math.random() * 0.04, 0.003, 0.03 + Math.random() * 0.02);
    coin.rotation.set(0.1, 0, 0.1);
    g.add(coin);
  }
  const hinge = new THREE.Group();
  hinge.position.set(0, 0.042, -0.026);
  const lid = mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.08, 18, 1, false, 0, Math.PI), wood, 0, 0, 0.026);
  lid.rotation.z = Math.PI / 2;
  hinge.add(lid);
  g.add(hinge);
  g.rotation.y = 0.5;
  return {
    group: g,
    update(t) {
      hinge.rotation.x = -(0.35 + 0.45 * Math.max(0, Math.sin(t * 0.6)));
    }
  };
}

function castle() {
  const g = new THREE.Group();
  const stone = weathered('#8f897d', { rust: 0, algae: 0.7, planks: 260, roughness: 0.95 });
  const roof = mat('#9a3a2a', 0.8);
  const dark = mat('#141414', 1);
  g.add(mesh(new THREE.BoxGeometry(0.075, 0.09, 0.055), stone, 0, 0.045, 0));
  for (let i = 0; i < 4; i++) g.add(mesh(new THREE.BoxGeometry(0.012, 0.012, 0.012), stone, -0.03 + i * 0.02, 0.096, 0.022));
  for (const [x, z] of [[-0.045, -0.032], [0.045, -0.032], [-0.045, 0.032], [0.045, 0.032]]) {
    g.add(mesh(cyl(0.017, 0.019, 0.12), stone, x, 0.06, z));
    g.add(mesh(new THREE.ConeGeometry(0.023, 0.045, 16), roof, x, 0.142, z));
    g.add(mesh(new THREE.BoxGeometry(0.006, 0.012, 0.004), dark, x, 0.09, z + Math.sign(z) * 0.018));
  }
  g.add(mesh(cyl(0.012, 0.012, 0.004), dark, 0, 0.018, 0.028));
  const door = mesh(new THREE.BoxGeometry(0.024, 0.03, 0.004), dark, 0, 0.015, 0.028);
  g.add(door);
  g.add(mesh(cyl(0.002, 0.002, 0.05), dark, 0.045, 0.2, 0.032));
  const flag = mesh(new THREE.PlaneGeometry(0.025, 0.014), mat('#d9c34a', 0.8, 0, { side: THREE.DoubleSide }), 0.058, 0.215, 0.032);
  g.add(flag);
  return {
    group: g,
    update(t) {
      flag.rotation.y = Math.sin(t * 2.2) * 0.4;
    }
  };
}

function branch(g, material, x, y, z, len, r, tilt, yaw, depth) {
  const m = mesh(cyl(r * 0.7, r, len, 8), material);
  m.geometry.translate(0, len / 2, 0);
  const holder = new THREE.Group();
  holder.position.set(x, y, z);
  holder.rotation.set(0, yaw, tilt);
  holder.add(m);
  g.add(holder);
  const tip = new THREE.Vector3(0, len, 0).applyEuler(holder.rotation).add(holder.position);
  if (depth > 0) {
    branch(g, material, tip.x, tip.y, tip.z, len * 0.7, r * 0.7, tilt + 0.5, yaw + 0.9, depth - 1);
    branch(g, material, tip.x, tip.y, tip.z, len * 0.7, r * 0.7, tilt - 0.5, yaw - 0.7, depth - 1);
  } else {
    g.add(mesh(new THREE.SphereGeometry(r * 0.9, 8, 6), material, tip.x, tip.y, tip.z));
  }
}

function coral() {
  const g = new THREE.Group();
  const pink = mat('#ff6f91', 0.7);
  const purple = mat('#9b59d0', 0.7);
  branch(g, pink, -0.015, 0, 0, 0.04, 0.006, 0.15, 0, 2);
  branch(g, pink, 0.02, 0, 0.015, 0.03, 0.005, -0.25, 1, 2);
  const brainGeo = new THREE.SphereGeometry(0.024, 32, 20);
  const p = brainGeo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(p, i);
    const k = 1 + 0.08 * Math.sin(v.x * 400 + Math.sin(v.z * 300) * 2);
    p.setXYZ(i, v.x * k, Math.max(0, v.y) * k * 0.8, v.z * k);
  }
  brainGeo.computeVertexNormals();
  g.add(mesh(brainGeo, mat('#f2a65a', 0.8), 0.035, 0, -0.02));
  for (let i = 0; i < 5; i++) {
    const h = 0.025 + Math.random() * 0.03;
    g.add(mesh(cyl(0.0045, 0.004, h, 10), purple, -0.045 + i * 0.008, h / 2, 0.03 + (i % 2) * 0.008));
  }
  const fan = mesh(new THREE.CircleGeometry(0.035, 20, 0, Math.PI), mat('#e8483f', 0.8, 0, { side: THREE.DoubleSide, transparent: true, opacity: 0.9 }), -0.04, 0.0, -0.025);
  fan.rotation.y = 0.4;
  g.add(fan);
  return g;
}

function rockGeometry(r, seed) {
  const geo = new THREE.IcosahedronGeometry(r, 2);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(p, i);
    const n = noise(v.x * 60 + seed, v.z * 60 + v.y * 40, 256, seed);
    v.multiplyScalar(0.75 + 0.45 * n);
    v.y = v.y * 0.7;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

function rock(r, seed) {
  const g = new THREE.Group();
  const stone = mat(seed % 2 ? '#6f6a63' : '#5b5f5c', 0.95, 0, { flatShading: true });
  g.add(mesh(rockGeometry(r, seed), stone, 0, r * 0.25, 0));
  g.add(mesh(rockGeometry(r * 0.4, seed + 3), stone, r * 0.9, 0.004, r * 0.5));
  g.rotation.y = seed;
  return g;
}

function anchor() {
  const g = new THREE.Group();
  const iron = weathered('#3e3b38', { rust: 0.9, algae: 0.5, roughness: 0.7, metalness: 0.6 });
  const shank = mesh(cyl(0.004, 0.004, 0.1), iron, 0, 0.05, 0);
  const ring = mesh(new THREE.TorusGeometry(0.01, 0.0025, 8, 20), iron, 0, 0.108, 0);
  const stock = mesh(cyl(0.003, 0.003, 0.05), iron, 0, 0.088, 0);
  stock.rotation.x = Math.PI / 2;
  const arms = mesh(new THREE.TorusGeometry(0.032, 0.004, 8, 24, Math.PI), iron, 0, 0.032, 0);
  arms.rotation.z = Math.PI;
  g.add(shank, ring, stock, arms);
  for (const side of [-1, 1]) {
    const fluke = mesh(new THREE.ConeGeometry(0.008, 0.016, 4), iron, side * 0.032, 0.036, 0);
    g.add(fluke);
  }
  const holder = new THREE.Group();
  holder.add(g);
  g.rotation.set(0, 0.6, 1.2);
  g.position.set(0.0, 0.03, 0);
  return holder;
}

function helmet() {
  const g = new THREE.Group();
  const brass = weathered('#b58a3c', { rust: 0.25, algae: 0.4, roughness: 0.35, metalness: 1 });
  const glass = mat('#0e1c22', 0.05, 0.7);
  g.add(mesh(new THREE.SphereGeometry(0.03, 28, 20), brass, 0, 0.038, 0));
  g.add(mesh(cyl(0.034, 0.036, 0.014, 24), brass, 0, 0.008, 0));
  const ports = [[0, 0.04, 0.029, 0], [0.027, 0.042, 0.008, Math.PI / 2], [-0.027, 0.042, 0.008, -Math.PI / 2]];
  for (const [x, y, z, ry] of ports) {
    const holder = new THREE.Group();
    holder.position.set(x, y, z);
    holder.rotation.y = ry;
    holder.add(mesh(new THREE.TorusGeometry(0.011, 0.0022, 8, 20), brass));
    holder.add(mesh(new THREE.CircleGeometry(0.011, 20), glass, 0, 0, -0.001));
    g.add(holder);
  }
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * Math.PI * 2;
    g.add(mesh(new THREE.SphereGeometry(0.0022, 6, 5), brass, Math.cos(a) * 0.036, 0.013, Math.sin(a) * 0.036));
  }
  g.rotation.set(-0.12, -0.5, 0.18);
  return g;
}

function buildGrass(scene) {
  const greens = ['#3f8f3a', '#5aa845', '#2f7a42', '#6fbf52', '#4c9a3c'].map(c => mat(c, 0.7, 0, { side: THREE.DoubleSide }));
  const geos = [0.09, 0.14, 0.2, 0.28, 0.36].map(h => {
    const geo = new THREE.PlaneGeometry(0.008, h, 1, 8);
    geo.translate(0, h / 2, 0);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const t = p.getY(i) / h;
      p.setX(i, p.getX(i) * (1 - t * 0.7) + 0.03 * t * t);
    }
    geo.computeVertexNormals();
    return geo;
  });
  const clumps = grassSpots().map(({ x, z }, n) => {
    const clump = new THREE.Group();
    const blades = [];
    const depth = (z + TANK.depth / 2) / TANK.depth;
    const tall = Math.round((1 - depth) * 3);
    for (let i = 0; i < 6; i++) {
      const bx = x + (Math.random() - 0.5) * 0.035;
      const bz = z + (Math.random() - 0.5) * 0.025;
      const b = mesh(geos[Math.min(4, tall + Math.floor(Math.random() * 2))], greens[(n + i) % greens.length], bx, sandHeight(bx, bz) - 0.004, bz);
      b.rotation.y = Math.random() * Math.PI * 2;
      b.castShadow = false;
      blades.push({ m: b, phase: Math.random() * 6 });
      clump.add(b);
    }
    clump.visible = false;
    scene.add(clump);
    return { clump, blades };
  });
  let shown = 0;
  return {
    setLevel(level) {
      shown = Math.min(clumps.length, level * CLUMPS_PER_LEVEL);
      clumps.forEach((c, i) => { c.clump.visible = i < shown; });
    },
    update(t) {
      for (let i = 0; i < shown; i++) {
        for (const b of clumps[i].blades) {
          b.m.rotation.z = Math.sin(t * 1.1 + b.phase) * 0.12;
          b.m.rotation.x = Math.sin(t * 0.8 + b.phase * 1.3) * 0.08;
        }
      }
    }
  };
}

const BUILDERS = { ship, car, plane, chest, castle, coral, anchor, helmet };

export function buildDecor(scene) {
  const items = new Map();
  for (const d of DECOR) {
    const root = new THREE.Group();
    const updates = [];
    if (d.id === 'rocks') {
      d.spots.forEach(([x, z, r], i) => {
        const rk = rock(r, i + 1);
        rk.position.set(x, sandHeight(x, z) - 0.004, z);
        root.add(rk);
      });
    } else {
      const built = BUILDERS[d.id]();
      const group = built.group || built;
      if (built.update) updates.push(built.update);
      const [x, z] = d.spots[0];
      group.position.x = x;
      group.position.z = z;
      group.position.y += sandHeight(x, z);
      root.add(group);
    }
    const [sx, sz] = d.spots[0];
    const silt = sandHeight(sx, sz) + 0.003;
    root.traverse(o => { if (o.material && o.material.userData.weather) o.material.userData.weather.uBury.value = silt; });
    root.visible = false;
    scene.add(root);
    items.set(d.id, { root, updates });
  }
  const grass = buildGrass(scene);
  return {
    show(ids, grassLevel) {
      for (const [id, item] of items) item.root.visible = ids.includes(id);
      grass.setLevel(grassLevel);
    },
    update(t) {
      for (const item of items.values()) if (item.root.visible) item.updates.forEach(u => u(t));
      grass.update(t);
    }
  };
}
