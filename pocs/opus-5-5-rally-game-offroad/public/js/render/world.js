import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GRID, CELL, HALF_EXTENT, ROAD_OFFSET, slopeAt } from '../core/terrain.js';
import { rightAt } from '../core/tracks.js';
import { FAR, mercatorUV } from '../core/geo.js';
import { paceNotes } from '../core/pacenotes.js';
import { createNoise2D, mulberry32, smoothstep, clamp } from '../core/math.js';
import { mudRoadTextures, groundDetailTextures, waterNormal, grassBlades } from './textures.js';
import { buildLandmarks } from './landmarks.js';

const OVERCAST = { rain: '#8d969c', snow: '#c9d0d6' };

function buildSky(scene, renderer, def, weather) {
  const sun = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - def.sun.elevation), THREE.MathUtils.degToRad(def.sun.azimuth));
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  let sky = null;
  if (weather === 'clear') {
    sky = new Sky();
    sky.scale.setScalar(30000);
    const u = sky.material.uniforms;
    u.turbidity.value = def.id === 'la' ? 9 : 5;
    u.rayleigh.value = def.id === 'sf' ? 2.2 : 1.6;
    u.mieCoefficient.value = 0.006;
    u.mieDirectionalG.value = 0.86;
    u.sunPosition.value.copy(sun);
    scene.add(sky);
    const envSky = new Sky();
    envSky.scale.setScalar(1000);
    envSky.material.uniforms.sunPosition.value.copy(sun);
    envSky.material.uniforms.turbidity.value = u.turbidity.value;
    envScene.add(envSky);
    scene.fog = new THREE.FogExp2(def.fog.color, def.fog.density);
  } else {
    const color = new THREE.Color(OVERCAST[weather]);
    scene.background = color;
    envScene.background = color;
    scene.fog = new THREE.FogExp2(color, weather === 'snow' ? 0.0022 : 0.0017);
  }
  const env = pmrem.fromScene(envScene, 0.02).texture;
  scene.environment = env;
  pmrem.dispose();
  return { sun, sky };
}

function buildLights(scene, def, weather, quality) {
  const clear = weather === 'clear';
  const hemi = new THREE.HemisphereLight(clear ? '#e4ecf2' : '#c4ccd4', def.palette.dirt, clear ? 0.8 : 1.5);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(clear ? (def.sun.elevation < 12 ? '#ffd9a8' : '#fff4e2') : '#dfe6ee', clear ? 3.2 : 0.9);
  sun.castShadow = quality.shadows > 0;
  sun.shadow.mapSize.set(Math.max(quality.shadows, 512), Math.max(quality.shadows, 512));
  const cam = sun.shadow.camera;
  cam.left = -70;
  cam.right = 70;
  cam.top = 70;
  cam.bottom = -70;
  cam.near = 1;
  cam.far = 600;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.04;
  scene.add(sun);
  scene.add(sun.target);
  return sun;
}

export function imagerySampler(place, layer) {
  const { canvas, info } = layer;
  const data = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height).data;
  const color = new THREE.Color();
  return (x, z) => {
    const [u, v] = mercatorUV(place, x, z, info);
    const px = clamp(Math.floor(u * canvas.width), 0, canvas.width - 1);
    const py = clamp(Math.floor((1 - v) * canvas.height), 0, canvas.height - 1);
    const o = (py * canvas.width + px) * 4;
    return color.setRGB(data[o] / 255, data[o + 1] / 255, data[o + 2] / 255, THREE.SRGBColorSpace);
  };
}

const tint = new THREE.Color();

export function coverOf(s) {
  const lum = (s.r + s.g + s.b) / 3;
  const green = s.g / Math.max(1e-4, (s.r + s.b) / 2);
  const forest = smoothstep(0.075, 0.035, lum) * smoothstep(0.9, 1.15, green);
  const meadow = (1 - forest) * smoothstep(0.95, 1.2, green);
  const bright = (1 - forest) * smoothstep(0.16, 0.32, lum);
  return { forest, meadow, bright, lum };
}

function landCover(c, s, pal, n) {
  const k = coverOf(s);
  c.copy(pal.dry).lerp(pal.grass, clamp(k.meadow + n, 0, 1));
  c.lerp(pal.sand, k.bright * 0.8);
  c.lerp(pal.forest, k.forest);
  tint.copy(s).multiplyScalar(0.18 / Math.max(k.lum, 0.02));
  c.lerp(tint, 0.08).multiplyScalar(1 + n * 0.5);
  return c;
}

function terrainColors(simWorld, def, weather, sat) {
  const { terrain, track } = simWorld;
  const noise = createNoise2D(def.seed + 44);
  const pal = Object.fromEntries(Object.entries(def.palette).map(([k, v]) => [k, new THREE.Color(v)]));
  const colors = new Float32Array(GRID * GRID * 3);
  const c = new THREE.Color();
  for (let gz = 0; gz < GRID; gz++) {
    for (let gx = 0; gx < GRID; gx++) {
      const i = gz * GRID + gx;
      const x = -HALF_EXTENT + gx * CELL;
      const z = -HALF_EXTENT + gz * CELL;
      const slope = slopeAt(terrain, x, z);
      const n = noise(x / 40, z / 40) * 0.3 + noise(x / 9, z / 9) * 0.15;
      landCover(c, sat(x, z), pal, n);
      c.lerp(pal.rock, smoothstep(0.45, 0.9, slope));
      const d = terrain.roadDist[i];
      c.lerp(pal.dirt, 1 - smoothstep(track.halfWidth - 1, track.halfWidth + 6 + n * 12, d));
      if (weather === 'snow') c.lerp(pal.snow, clamp(0.8 - slope * 0.9 - (d < track.halfWidth + 2 ? 0.45 : 0) + n, 0, 0.92));
      if (weather === 'rain') c.multiplyScalar(0.8);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
  }
  return colors;
}

function buildTerrainMesh(simWorld, def, weather, sat) {
  const { terrain } = simWorld;
  const positions = new Float32Array(GRID * GRID * 3);
  const uvs = new Float32Array(GRID * GRID * 2);
  for (let gz = 0; gz < GRID; gz++) {
    for (let gx = 0; gx < GRID; gx++) {
      const i = gz * GRID + gx;
      positions[i * 3] = -HALF_EXTENT + gx * CELL;
      positions[i * 3 + 1] = terrain.heights[i];
      positions[i * 3 + 2] = -HALF_EXTENT + gz * CELL;
      uvs[i * 2] = gx * CELL / 7;
      uvs[i * 2 + 1] = gz * CELL / 7;
    }
  }
  const index = [];
  for (let gz = 0; gz < GRID - 1; gz++) {
    for (let gx = 0; gx < GRID - 1; gx++) {
      const a = gz * GRID + gx;
      const b = a + 1;
      const c = a + GRID;
      const d = c + 1;
      index.push(a, c, b, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(terrainColors(simWorld, def, weather, sat), 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  const detail = groundDetailTextures(def.seed);
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, map: detail.map, normalMap: detail.normalMap, normalScale: new THREE.Vector2(0.9, 0.9), roughness: weather === 'rain' ? 0.6 : 0.96, envMapIntensity: 0.4 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  return mesh;
}

function buildRoad(simWorld, def, weather) {
  const { track, terrain } = simWorld;
  const across = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1];
  const n = track.count;
  const positions = new Float32Array((n + 1) * across.length * 3);
  const uvs = new Float32Array((n + 1) * across.length * 2);
  for (let k = 0; k <= n; k++) {
    const i = k % n;
    const [rx, rz] = rightAt(track, i);
    across.forEach((a, j) => {
      const x = track.xs[i] + rx * a * track.halfWidth;
      const z = track.zs[i] + rz * a * track.halfWidth;
      const v = (k * across.length + j) * 3;
      positions[v] = x;
      positions[v + 1] = terrain.heightAt(x, z) + ROAD_OFFSET;
      positions[v + 2] = z;
      uvs[(k * across.length + j) * 2] = (a + 1) / 2;
      uvs[(k * across.length + j) * 2 + 1] = (k * track.spacing) / 14;
    });
  }
  const index = [];
  const w = across.length;
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < w - 1; j++) {
      const a = k * w + j;
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      index.push(a, b, c, b, d, c);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(index);
  geo.computeVertexNormals();
  const tex = mudRoadTextures(def.seed, weather);
  const mat = new THREE.MeshStandardMaterial({
    ...tex,
    normalScale: new THREE.Vector2(1.4, 1.4),
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    envMapIntensity: weather === 'clear' ? 0.5 : 1.1,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  return mesh;
}

function buildPuddles(simWorld, weather) {
  const group = new THREE.Group();
  const mat = new THREE.MeshPhysicalMaterial({ color: weather === 'snow' ? '#6d6660' : '#3b2d20', roughness: weather === 'snow' ? 0.35 : 0.03, metalness: 0.1, clearcoat: 1, transparent: true, opacity: 0.92, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
  const geo = new THREE.CircleGeometry(1, 28);
  for (const p of simWorld.track.puddles) {
    const m = new THREE.Mesh(geo, mat);
    m.rotation.set(-Math.PI / 2, 0, p.angle);
    m.scale.set(p.r, p.r * p.stretch, 1);
    m.position.set(p.x, simWorld.terrain.heightAt(p.x, p.z) + ROAD_OFFSET + 0.02, p.z);
    m.receiveShadow = true;
    group.add(m);
  }
  return group;
}

function buildWater(def, level) {
  if (level === null) return null;
  const normal = waterNormal(def.seed);
  normal.repeat.set(2000, 2000);
  const mat = new THREE.MeshPhysicalMaterial({ color: def.id === 'tahoe' ? '#0f4e6c' : '#2b4a57', roughness: 0.05, metalness: 0.15, normalMap: normal, normalScale: new THREE.Vector2(0.3, 0.3), clearcoat: 0.6 });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = level + (def.id === 'tahoe' ? 0.3 : 0);
  mesh.receiveShadow = true;
  return mesh;
}

function snowCanvas(source) {
  const c = document.createElement('canvas');
  c.width = source.width;
  c.height = source.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(source, 0, 0);
  const img = ctx.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < img.data.length; i += 4) {
    const lum = (img.data[i] + img.data[i + 1] + img.data[i + 2]) / 3;
    const t = lum < 45 ? 0.25 : 0.72;
    for (let k = 0; k < 3; k++) img.data[i + k] = img.data[i + k] * (1 - t) + 240 * t;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function buildFar(simWorld, imagery, weather) {
  const { geo, terrain } = simWorld;
  const n = FAR.n;
  const cell = (2 * FAR.half) / (n - 1);
  const positions = new Float32Array(n * n * 3);
  const uvs = new Float32Array(n * n * 2);
  const inner = HALF_EXTENT - 12;
  for (let gz = 0; gz < n; gz++) {
    for (let gx = 0; gx < n; gx++) {
      const i = gz * n + gx;
      const x = -FAR.half + gx * cell;
      const z = -FAR.half + gz * cell;
      const inside = Math.abs(x) < inner && Math.abs(z) < inner;
      const edge = gx === 0 || gz === 0 || gx === n - 1 || gz === n - 1;
      positions[i * 3] = x;
      positions[i * 3 + 1] = geo.far.heights[i] - terrain.base - (inside ? 8 : 0) - (edge ? 400 : 0);
      positions[i * 3 + 2] = z;
      const [u, v] = mercatorUV(geo.place, x, z, imagery.far.info);
      uvs[i * 2] = u;
      uvs[i * 2 + 1] = v;
    }
  }
  const index = [];
  for (let gz = 0; gz < n - 1; gz++) {
    for (let gx = 0; gx < n - 1; gx++) {
      const a = gz * n + gx;
      index.push(a, a + n, a + 1, a + 1, a + n, a + n + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  g.setIndex(index);
  g.computeVertexNormals();
  const tex = new THREE.CanvasTexture(weather === 'snow' ? snowCanvas(imagery.far.canvas) : imagery.far.canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 1, color: weather === 'rain' ? '#b8b8b8' : '#ffffff', envMapIntensity: 0.4 });
  const mesh = new THREE.Mesh(g, mat);
  mesh.receiveShadow = true;
  return mesh;
}

function chevronTexture(dir) {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 96;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#a3120b';
  ctx.fillRect(0, 0, 256, 96);
  ctx.fillStyle = '#ffffff';
  for (let k = 0; k < 3; k++) {
    const x = 40 + k * 70;
    ctx.beginPath();
    if (dir === 'LEFT') {
      ctx.moveTo(x + 30, 12);
      ctx.lineTo(x - 5, 48);
      ctx.lineTo(x + 30, 84);
      ctx.lineTo(x + 50, 84);
      ctx.lineTo(x + 15, 48);
      ctx.lineTo(x + 50, 12);
    } else {
      ctx.moveTo(x, 12);
      ctx.lineTo(x + 35, 48);
      ctx.lineTo(x, 84);
      ctx.lineTo(x + 20, 84);
      ctx.lineTo(x + 55, 48);
      ctx.lineTo(x + 20, 12);
    }
    ctx.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildChevrons(simWorld) {
  const { track, terrain } = simWorld;
  const group = new THREE.Group();
  const notes = paceNotes(track).filter((n) => n.severity <= 4);
  const board = new THREE.PlaneGeometry(2.6, 0.95);
  const post = new THREE.CylinderGeometry(0.06, 0.06, 1.6, 6);
  post.translate(0, 0.8, 0);
  const postMat = new THREE.MeshStandardMaterial({ color: '#333333', roughness: 0.6 });
  for (const dir of ['LEFT', 'RIGHT']) {
    const spots = [];
    for (const note of notes.filter((n) => n.dir === dir)) {
      const steps = Math.max(2, Math.round(note.length / 18));
      for (let k = 0; k < steps; k++) {
        const i = (note.i + Math.round((k * note.length) / steps / track.spacing)) % track.count;
        const [rx, rz] = rightAt(track, i);
        const side = dir === 'LEFT' ? 1 : -1;
        const x = track.xs[i] + rx * side * (track.halfWidth + 2.5);
        const z = track.zs[i] + rz * side * (track.halfWidth + 2.5);
        spots.push({ x, z, y: terrain.heightAt(x, z), heading: track.heading[i] });
      }
    }
    if (!spots.length) continue;
    const mat = new THREE.MeshStandardMaterial({ map: chevronTexture(dir), roughness: 0.5, side: THREE.DoubleSide });
    group.add(instanced(board, mat, spots, (sp, p, e, s) => {
      p.set(sp.x, sp.y + 1.9, sp.z);
      e.set(0, sp.heading + Math.PI, 0);
      s.set(1, 1, 1);
      return false;
    }));
    group.add(instanced(post, postMat, spots, (sp, p, e, s) => {
      p.set(sp.x, sp.y, sp.z);
      e.set(0, 0, 0);
      s.set(1, 1, 1);
      return false;
    }));
  }
  return group;
}

function colored(geo, color) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  const c = new THREE.Color(color);
  const arr = new Float32Array(g.attributes.position.count * 3);
  for (let i = 0; i < arr.length; i += 3) {
    arr[i] = c.r;
    arr[i + 1] = c.g;
    arr[i + 2] = c.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
  if (g.attributes.uv) g.deleteAttribute('uv');
  return g;
}

function ragged(geo, amount, seed) {
  const noise = createNoise2D(seed);
  const p = geo.attributes.position;
  for (let v = 0; v < p.count; v++) {
    const x = p.getX(v);
    const z = p.getZ(v);
    const r = Math.hypot(x, z);
    if (r < 1e-3) continue;
    const k = 1 + noise(Math.atan2(z, x) * 2.5, p.getY(v) * 0.7) * amount;
    p.setX(v, x * k);
    p.setZ(v, z * k);
  }
  geo.computeVertexNormals();
  return geo;
}

function liteConifer() {
  const trunk = colored(new THREE.CylinderGeometry(0.3, 0.4, 6, 5).translate(0, 3, 0), '#5b3f2c');
  const low = colored(ragged(new THREE.ConeGeometry(3.2, 8, 7), 0.2, 5).translate(0, 7, 0), '#223a23');
  const high = colored(ragged(new THREE.ConeGeometry(2, 6, 7), 0.2, 6).translate(0, 11.5, 0), '#2d4829');
  return mergeGeometries([trunk, low, high]);
}

function liteBroadleaf() {
  const trunk = colored(new THREE.CylinderGeometry(0.3, 0.45, 4, 5).translate(0, 2, 0), '#5a4330');
  const crown = ragged(new THREE.IcosahedronGeometry(3.4, 1), 0.25, 8);
  crown.scale(1.2, 0.75, 1.1);
  return mergeGeometries([trunk, colored(crown.translate(0, 5.5, 0), '#34472a')]);
}

function treeGeometry(kind) {
  const parts = [];
  const t = (geo, color, x, y, z, rx = 0, rz = 0) => {
    geo.rotateX(rx);
    geo.rotateZ(rz);
    geo.translate(x, y, z);
    parts.push(colored(geo, color));
  };
  if (kind === 'pine') {
    t(new THREE.CylinderGeometry(0.2, 0.42, 9, 8), '#5b3f2c', 0, 4.5, 0);
    const layers = 8;
    for (let k = 0; k < layers; k++) {
      const f = k / (layers - 1);
      const r = 3.4 * (1 - f * 0.85);
      const shade = ['#1e3320', '#243a24', '#2b4428', '#324d2c'][k % 4];
      t(ragged(new THREE.ConeGeometry(r, 3.2, 12, 2), 0.28, k + 3), shade, 0, 3.6 + k * 1.45, 0);
    }
  } else if (kind === 'cypress') {
    t(new THREE.CylinderGeometry(0.35, 0.6, 5.5, 8), '#4e3a2b', 0, 2.7, 0, 0, 0.1);
    const blobs = [[3.8, 0, 6.8, 0], [3.1, 2.6, 7.6, 1.3], [3, -2.7, 7.2, -0.9], [2.6, 0.9, 9, 1.9], [2.4, -1.5, 8.6, -2], [2.2, 3.4, 6.2, -1.6]];
    blobs.forEach(([r, x, y, z], k) => {
      const g = ragged(new THREE.IcosahedronGeometry(r, 2), 0.22, k + 11);
      g.scale(1.35, 0.5, 1.15);
      t(g, k % 2 ? '#2b3d24' : '#34482b', x, y, z);
    });
  } else {
    const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.6, 7, 0), new THREE.Vector3(1.4, 14, 0));
    t(new THREE.TubeGeometry(curve, 12, 0.28, 7), '#7a6248', 0, 0, 0);
    for (let k = 0; k < 12; k++) {
      const frond = new THREE.PlaneGeometry(0.9, 4.6, 1, 4);
      const pos = frond.attributes.position;
      for (let v = 0; v < pos.count; v++) {
        const yy = pos.getY(v) + 2.3;
        pos.setZ(v, -yy * yy * 0.09);
        pos.setY(v, yy);
      }
      frond.rotateX(-1.05 + (k % 3) * 0.15);
      frond.rotateY((k / 12) * Math.PI * 2);
      t(frond, k % 2 ? '#3f6a2e' : '#4d7a33', 1.4, 14, 0);
    }
  }
  return mergeGeometries(parts);
}

function instanced(geo, mat, items, place) {
  const mesh = new THREE.InstancedMesh(geo, mat, items.length);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const p = new THREE.Vector3();
  const e = new THREE.Euler();
  const color = new THREE.Color();
  items.forEach((it, k) => {
    const shade = place(it, p, e, s, color);
    q.setFromEuler(e);
    m.compose(p, q, s);
    mesh.setMatrixAt(k, m);
    if (shade) mesh.setColorAt(k, color);
  });
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildProps(simWorld, def, weather, quality) {
  const group = new THREE.Group();
  const { props } = simWorld;
  const snow = weather === 'snow';
  const rand = mulberry32(def.seed + 3);
  const treeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, envMapIntensity: 0.25, side: def.trees.kind === 'palm' ? THREE.DoubleSide : THREE.FrontSide });
  group.add(instanced(treeGeometry(def.trees.kind), treeMat, props.trees, (t, p, e, s, color) => {
    p.set(t.x, t.y - 0.3, t.z);
    e.set((rand() - 0.5) * 0.06, t.rot, (rand() - 0.5) * 0.06);
    s.setScalar(t.scale * (def.id === 'yosemite' ? 1.7 : 1));
    color.setHSL(0, 0, 0.75 + rand() * 0.4);
    if (snow) color.lerp(new THREE.Color('#ffffff'), 0.35);
    return true;
  }));
  const fillGeo = def.trees.kind === 'pine' ? liteConifer() : liteBroadleaf();
  const fillMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, envMapIntensity: 0.25 });
  const fill = instanced(fillGeo, fillMat, props.forest, (t, p, e, s, color) => {
    p.set(t.x, t.y - 0.3, t.z);
    e.set(0, t.rot, 0);
    s.setScalar(t.scale * (def.id === 'yosemite' ? 1.7 : 1));
    color.setHSL(0, 0, 0.7 + rand() * 0.4);
    if (snow) color.lerp(new THREE.Color('#ffffff'), 0.35);
    return true;
  });
  fill.castShadow = false;
  group.add(fill);
  const rockGeo = new THREE.IcosahedronGeometry(1, 1);
  const noise = createNoise2D(def.seed + 8);
  const rp = rockGeo.attributes.position;
  for (let v = 0; v < rp.count; v++) {
    const k = 1 + noise(rp.getX(v) * 1.7, rp.getZ(v) * 1.7 + rp.getY(v)) * 0.35;
    rp.setXYZ(v, rp.getX(v) * k, rp.getY(v) * k * 0.7, rp.getZ(v) * k);
  }
  rockGeo.computeVertexNormals();
  const rockMat = new THREE.MeshStandardMaterial({ color: def.id === 'yosemite' ? '#a8a398' : def.palette.rock, roughness: 0.93, flatShading: true });
  group.add(instanced(rockGeo, rockMat, props.rocks, (r, p, e, s, color) => {
    p.set(r.x, r.y - r.scale * 0.2, r.z);
    e.set(0, r.rot, 0);
    s.set(r.scale, r.scale, r.scale * 0.9);
    color.setHSL(0, 0, 0.8 + rand() * 0.3);
    if (snow) color.lerp(new THREE.Color('#ffffff'), 0.3);
    return true;
  }));
  const grassCount = Math.floor(props.grass.length * quality.grass * (snow ? 0.3 : 1));
  if (grassCount > 0) {
    const blade = new THREE.PlaneGeometry(1.4, 0.9);
    blade.translate(0, 0.45, 0);
    const blade2 = blade.clone().rotateY(Math.PI / 2);
    const grassGeo = mergeGeometries([blade, blade2]);
    const grassMat = new THREE.MeshStandardMaterial({ map: grassBlades(snow ? '#9aa08a' : def.palette.grass), alphaTest: 0.45, side: THREE.DoubleSide, roughness: 0.9, envMapIntensity: 0.25 });
    const grass = instanced(grassGeo, grassMat, props.grass.slice(0, grassCount), (g, p, e, s) => {
      p.set(g.x, g.y - 0.05, g.z);
      e.set(0, g.rot, 0);
      s.setScalar(g.scale);
      return false;
    });
    grass.castShadow = false;
    group.add(grass);
  }
  const post = new THREE.CylinderGeometry(0.07, 0.07, 1.2, 8);
  post.translate(0, 0.6, 0);
  const postMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.5 });
  group.add(instanced(post, postMat, props.markers, (mk, p, e, s, color) => {
    p.set(mk.x, mk.y, mk.z);
    e.set(0, 0, 0);
    s.set(1, 1, 1);
    color.set(Math.round(mk.x + mk.z) % 2 === 0 ? '#e0401b' : '#f2f2f2');
    return true;
  }));
  return group;
}

function textBanner(text) {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 128;
  const ctx = c.getContext('2d');
  for (let x = 0; x < 1024; x += 32) {
    for (let y = 0; y < 128; y += 32) {
      ctx.fillStyle = ((x + y) / 32) % 2 === 0 ? '#111' : '#f4f4f4';
      ctx.fillRect(x, y, 32, 32);
    }
  }
  ctx.fillStyle = 'rgba(12,12,12,0.86)';
  ctx.fillRect(120, 18, 784, 92);
  ctx.fillStyle = '#ffb31a';
  ctx.font = 'bold 60px Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 512, 66);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildStartArch(simWorld) {
  const { track, terrain } = simWorld;
  const group = new THREE.Group();
  const [rx, rz] = rightAt(track, 0);
  const span = track.halfWidth + 2.2;
  const mat = new THREE.MeshStandardMaterial({ color: '#2a2d31', roughness: 0.4, metalness: 0.7 });
  const y0 = terrain.heightAt(track.xs[0], track.zs[0]);
  for (const s of [-1, 1]) {
    const x = track.xs[0] + rx * span * s;
    const z = track.zs[0] + rz * span * s;
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.5, 7, 0.5), mat);
    leg.position.set(x, terrain.heightAt(x, z) + 3.5, z);
    leg.castShadow = true;
    group.add(leg);
  }
  const banner = new THREE.Mesh(new THREE.BoxGeometry(span * 2 + 0.5, 1.6, 0.2), [mat, mat, mat, mat, new THREE.MeshStandardMaterial({ map: textBanner('CALIFORNIA OFFROAD RALLY'), roughness: 0.6 }), new THREE.MeshStandardMaterial({ map: textBanner('FINISH'), roughness: 0.6 })]);
  banner.position.set(track.xs[0], y0 + 6.6, track.zs[0]);
  banner.rotation.y = track.heading[0] + Math.PI;
  banner.castShadow = true;
  group.add(banner);
  const line = new THREE.Mesh(new THREE.PlaneGeometry(track.halfWidth * 2, 1.2), new THREE.MeshStandardMaterial({ map: textBanner(''), polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6 }));
  line.rotation.set(-Math.PI / 2, 0, track.heading[0] + Math.PI / 2);
  line.position.set(track.xs[0], y0 + ROAD_OFFSET + 0.03, track.zs[0]);
  group.add(line);
  return group;
}

function buildClouds(def, weather) {
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(512, 512);
  const noise = createNoise2D(def.seed + 90);
  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      let v = 0;
      let amp = 1;
      let f = 1;
      for (let o = 0; o < 5; o++) {
        const nx = Math.cos((x / 512) * Math.PI * 2) * f;
        const ny = Math.sin((x / 512) * Math.PI * 2) * f;
        v += noise(nx * 1.5 + ny, (y / 512) * 6 * f) * amp;
        amp *= 0.5;
        f *= 2;
      }
      const a = clamp((v - (weather === 'clear' ? 0.12 : -0.2)) * 2.2, 0, 1);
      const i = (y * 512 + x) * 4;
      const shade = weather === 'clear' ? 255 : 170;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = shade;
      img.data[i + 3] = a * 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(50000, 50000), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, fog: false, opacity: weather === 'clear' ? 0.75 : 0.95, side: THREE.DoubleSide }));
  mesh.rotation.x = Math.PI / 2;
  mesh.position.y = 3200;
  mesh.renderOrder = -1;
  return mesh;
}

export async function buildWorld(scene, renderer, simWorld, weather, quality, imagery) {
  const def = simWorld.track.def;
  const { sun: sunDir } = buildSky(scene, renderer, def, weather);
  const sun = buildLights(scene, def, weather, quality);
  const sat = imagerySampler(simWorld.geo.place, imagery.near);
  scene.add(buildTerrainMesh(simWorld, def, weather, sat));
  scene.add(buildFar(simWorld, imagery, weather));
  scene.add(buildRoad(simWorld, def, weather));
  scene.add(buildPuddles(simWorld, weather));
  const water = buildWater(def, simWorld.terrain.water);
  if (water) scene.add(water);
  scene.add(buildProps(simWorld, def, weather, quality));
  scene.add(buildChevrons(simWorld));
  scene.add(buildStartArch(simWorld));
  const landmarks = await buildLandmarks(simWorld);
  landmarks.forEach((l) => scene.add(l));
  const clouds = buildClouds(def, weather);
  scene.add(clouds);
  const falls = landmarks.filter((l) => l.userData.tex);
  let t = 0;
  return {
    sun,
    update(dt, target) {
      t += dt;
      sun.position.set(target.x + sunDir.x * 300, target.y + sunDir.y * 300, target.z + sunDir.z * 300);
      sun.target.position.copy(target);
      if (water) water.material.normalMap.offset.set(t * 0.004, t * 0.003);
      clouds.material.map.offset.x = t * 0.0015;
      for (const f of falls) f.userData.tex.offset.y = t * 0.6;
    },
    setShadows(size) {
      sun.castShadow = size > 0;
      if (size > 0 && sun.shadow.mapSize.x !== size) {
        sun.shadow.mapSize.set(size, size);
        sun.shadow.map?.dispose();
        sun.shadow.map = null;
      }
    },
  };
}
