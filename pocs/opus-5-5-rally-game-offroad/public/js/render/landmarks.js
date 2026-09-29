import * as THREE from 'three';
import { toLocal } from '../core/geo.js';
import { mulberry32 } from '../core/math.js';
import { windowGrid, waterfallTexture } from './textures.js';

const INTERNATIONAL_ORANGE = '#b8392a';

function ctxFor(world) {
  const place = world.geo.place;
  const base = world.terrain.base;
  const local = (lat, lon) => toLocal(place, lat, lon);
  const ground = (x, z) => world.geo.elevation(x, z) - base;
  return { local, ground, base };
}

function shadowed(group) {
  group.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  return group;
}

function goldenGate(c) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: INTERNATIONAL_ORANGE, roughness: 0.55, metalness: 0.25 });
  const [sx, sz] = c.local(37.8106, -122.4771);
  const [nx, nz] = c.local(37.8324, -122.481);
  const [cx, cz] = c.local(37.81972, -122.47861);
  const len = Math.hypot(nx - sx, nz - sz);
  const dir = new THREE.Vector2((nx - sx) / len, (nz - sz) / len);
  const sea = -c.base;
  const H = 227;
  const deckY = sea + 67;
  const along = (d) => [cx + dir.x * d, cz + dir.y * d];
  const angle = Math.atan2(dir.x, dir.y);
  const towerGeo = new THREE.BoxGeometry(10, H, 10);
  const beamGeo = new THREE.BoxGeometry(33, 7, 8);
  for (const d of [-640, 640]) {
    const [tx, tz] = along(d);
    const tower = new THREE.Group();
    for (const side of [-1, 1]) {
      const leg = new THREE.Mesh(towerGeo, mat);
      leg.position.set(side * 14, H / 2, 0);
      tower.add(leg);
    }
    for (const y of [deckY - sea + 16, 110, 150, 185, H - 5]) {
      const beam = new THREE.Mesh(beamGeo, mat);
      beam.position.set(0, y, 0);
      tower.add(beam);
    }
    tower.position.set(tx, sea, tz);
    tower.rotation.y = angle;
    g.add(tower);
  }
  const deck = new THREE.Mesh(new THREE.BoxGeometry(27, 7, 2000), mat);
  deck.position.set(cx, deckY, cz);
  deck.rotation.y = angle;
  g.add(deck);
  const lines = [];
  for (const side of [-1, 1]) {
    const pts = [];
    const ox = -dir.y * side * 14;
    const oz = dir.x * side * 14;
    for (let d = -1000; d <= 1000; d += 20) {
      const main = Math.abs(d) <= 640;
      const t = main ? d / 640 : (Math.abs(d) - 640) / 360;
      const y = main ? deckY + 8 + (H - 67 - 8) * t * t : sea + H - (H - 67) * Math.min(1, t) * (2 - Math.min(1, t));
      const [px, pz] = along(d);
      pts.push(new THREE.Vector3(px + ox, y, pz + oz));
      if (Math.abs(d) < 990) lines.push(px + ox, y, pz + oz, px + ox, deckY + 3, pz + oz);
    }
    g.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 200, 0.9, 6), mat));
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  g.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: INTERNATIONAL_ORANGE })));
  return shadowed(g);
}

function tower(c, lat, lon, height, width, shape, color) {
  const [x, z] = c.local(lat, lon);
  const y = c.ground(x, z);
  let geo;
  if (shape === 'pyramid') geo = new THREE.ConeGeometry(width, height, 4);
  else if (shape === 'round') geo = new THREE.CylinderGeometry(width * 0.8, width, height, 28);
  else geo = new THREE.BoxGeometry(width, height, width * 0.8);
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.25, metalness: 0.7 }));
  m.position.set(x, y + height / 2, z);
  if (shape === 'pyramid') m.rotation.y = Math.PI / 4;
  return m;
}

function skyline(c, lat, lon, radius, count, maxH, seed) {
  const g = new THREE.Group();
  const rand = mulberry32(seed);
  const tex = windowGrid(seed);
  const [cx, cz] = c.local(lat, lon);
  for (let k = 0; k < count; k++) {
    const a = rand() * Math.PI * 2;
    const r = Math.sqrt(rand()) * radius;
    const x = cx + Math.cos(a) * r;
    const z = cz + Math.sin(a) * r;
    const h = 25 + rand() * rand() * maxH * (1 - r / radius * 0.6);
    const w = 18 + rand() * 26;
    const t = tex.clone();
    t.repeat.set(w / 16, h / 24);
    t.needsUpdate = true;
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, w * (0.6 + rand() * 0.6)), new THREE.MeshStandardMaterial({ map: t, roughness: 0.35, metalness: 0.5 }));
    b.position.set(x, c.ground(x, z) + h / 2 - 2, z);
    b.rotation.y = rand() * 0.3;
    g.add(b);
  }
  return g;
}

function sfSkyline(c) {
  const g = skyline(c, 37.7915, -122.401, 900, 70, 170, 71);
  g.add(tower(c, 37.7952, -122.4028, 260, 26, 'pyramid', '#e8e4dc'));
  g.add(tower(c, 37.7899, -122.3969, 326, 22, 'round', '#b8c4cc'));
  return g;
}

function laSkyline(c) {
  const g = skyline(c, 34.0505, -118.2555, 800, 60, 220, 83);
  g.add(tower(c, 34.051, -118.2542, 310, 24, 'round', '#9aa8b2'));
  g.add(tower(c, 34.05, -118.2593, 335, 26, 'box', '#8fb3c9'));
  return g;
}

function alcatraz(c) {
  const g = new THREE.Group();
  const [x, z] = c.local(37.8267, -122.4228);
  const y = Math.max(c.ground(x, z), -c.base + 30);
  const cream = new THREE.MeshStandardMaterial({ color: '#d9d2c1', roughness: 0.8 });
  const block = new THREE.Mesh(new THREE.BoxGeometry(150, 16, 32), cream);
  block.position.set(x, y + 8, z);
  block.rotation.y = 0.5;
  const light = new THREE.Mesh(new THREE.CylinderGeometry(3, 4, 30, 12), cream);
  light.position.set(x + 60, y + 15, z + 30);
  g.add(block, light);
  return shadowed(g);
}

function palaceOfFineArts(c) {
  const g = new THREE.Group();
  const [x, z] = c.local(37.8029, -122.4484);
  const y = c.ground(x, z);
  const stone = new THREE.MeshStandardMaterial({ color: '#c79a73', roughness: 0.75 });
  const rotunda = new THREE.Mesh(new THREE.CylinderGeometry(19, 21, 28, 8), stone);
  rotunda.position.set(x, y + 14, z);
  const dome = new THREE.Mesh(new THREE.SphereGeometry(19, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ color: '#b98262', roughness: 0.6 }));
  dome.position.set(x, y + 28, z);
  g.add(rotunda, dome);
  const column = new THREE.CylinderGeometry(1.4, 1.6, 20, 10);
  for (let k = 0; k < 26; k++) {
    const a = -1.2 + (k / 25) * 2.4;
    const m = new THREE.Mesh(column, stone);
    m.position.set(x + Math.sin(a) * 80, y + 10, z - 40 + Math.cos(a) * 80);
    g.add(m);
  }
  return shadowed(g);
}

function letterTexture(ch) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 160;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 170px "Arial Narrow", Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(ch, 64, 150);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function hollywoodSign(c) {
  const g = new THREE.Group();
  const [x, z] = c.local(34.13406, -118.32159);
  const spacing = 12.3;
  const angle = -0.2;
  const panel = new THREE.PlaneGeometry(11, 13.7);
  panel.translate(0, 6.85, 0);
  [...'HOLLYWOOD'].forEach((ch, k) => {
    const mat = new THREE.MeshStandardMaterial({ map: letterTexture(ch), alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.6 });
    const d = (k - 4) * spacing;
    const lx = x + Math.cos(angle) * d;
    const lz = z + Math.sin(angle) * d + 6;
    const m = new THREE.Mesh(panel, mat);
    m.position.set(lx, c.ground(lx, lz) - 1, lz);
    m.rotation.y = -angle;
    g.add(m);
  });
  return shadowed(g);
}

function observatory(c) {
  const g = new THREE.Group();
  const [x, z] = c.local(34.11833, -118.30033);
  const white = new THREE.MeshStandardMaterial({ color: '#efece4', roughness: 0.55 });
  const copper = new THREE.MeshStandardMaterial({ color: '#3f6f63', roughness: 0.35, metalness: 0.7 });
  const main = new THREE.Mesh(new THREE.BoxGeometry(84, 14, 30), white);
  main.position.y = 7;
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(12, 12, 8, 36), white);
  drum.position.y = 16;
  const dome = new THREE.Mesh(new THREE.SphereGeometry(12, 36, 18, 0, Math.PI * 2, 0, Math.PI / 2), copper);
  dome.position.y = 20;
  g.add(main, drum, dome);
  for (const s of [-1, 1]) {
    const d2 = new THREE.Mesh(new THREE.CylinderGeometry(6.5, 6.5, 4, 24), white);
    d2.position.set(s * 36, 16, 0);
    const small = new THREE.Mesh(new THREE.SphereGeometry(6.5, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), copper);
    small.position.set(s * 36, 18, 0);
    g.add(d2, small);
  }
  g.position.set(x, c.ground(x, z) - 1, z);
  g.rotation.y = 0.3;
  return shadowed(g);
}

function fall(c, lat, lon, width) {
  const [x, z] = c.local(lat, lon);
  let top = { y: -Infinity };
  let bottom = { y: Infinity };
  for (let d = -300; d <= 300; d += 10) {
    const px = x;
    const pz = z + d;
    const y = c.ground(px, pz);
    if (y > top.y) top = { x: px, y, z: pz };
    if (y < bottom.y) bottom = { x: px, y, z: pz };
  }
  const tex = waterfallTexture();
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide, fog: true });
  const h = Math.max(40, top.y - bottom.y);
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(width, h), mat);
  plane.position.set((top.x + bottom.x) / 2, bottom.y + h / 2, (top.z + bottom.z) / 2 + 8);
  plane.lookAt(bottom.x, bottom.y + h / 2, bottom.z + 400);
  const g = new THREE.Group();
  g.add(plane);
  const mist = new THREE.Mesh(new THREE.SphereGeometry(width, 16, 8), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.22, depthWrite: false }));
  mist.position.set(bottom.x, bottom.y + width * 0.4, bottom.z + 10);
  g.add(mist);
  g.userData.tex = tex;
  return g;
}

function yosemiteFalls(c) {
  return fall(c, 37.75684, -119.59678, 28);
}

function bridalveilFall(c) {
  return fall(c, 37.7166, -119.6464, 16);
}

function boathouse(c, world) {
  const g = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: '#6b4a31', roughness: 0.85 });
  const roof = new THREE.MeshStandardMaterial({ color: '#3c4a3f', roughness: 0.7 });
  const { cx } = world.track.def.frame;
  const water = world.terrain.water;
  let z = world.track.def.frame.cz;
  while (z > -1500 && world.terrain.raw(cx, z) > water + 0.4) z -= 5;
  const y = water + 0.4;
  const house = new THREE.Mesh(new THREE.BoxGeometry(14, 7, 10), wood);
  house.position.set(cx, c.ground(cx, z + 25) + 3.5, z + 25);
  const top = new THREE.Mesh(new THREE.ConeGeometry(10, 5, 4), roof);
  top.position.set(cx, house.position.y + 6, z + 25);
  top.rotation.y = Math.PI / 4;
  const pier = new THREE.Mesh(new THREE.BoxGeometry(4, 0.6, 80), wood);
  pier.position.set(cx, y + 1.2, z - 30);
  g.add(house, top, pier);
  for (let k = 0; k < 9; k++) {
    const pile = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 5, 6), wood);
    pile.position.set(cx + 2, y - 1, z + 8 - k * 9);
    g.add(pile);
  }
  return shadowed(g);
}

const LANDMARKS = { goldenGate, sfSkyline, alcatraz, palaceOfFineArts, hollywoodSign, observatory, laSkyline, yosemiteFalls, bridalveilFall, boathouse };

export async function buildLandmarks(world) {
  const c = ctxFor(world);
  return Promise.all(world.track.def.landmarks.map((name) => LANDMARKS[name](c, world)));
}
