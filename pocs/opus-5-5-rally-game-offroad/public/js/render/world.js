import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { GRID, CELL, HALF_EXTENT, ROAD_OFFSET, slopeAt } from '../core/terrain.js';
import { rightAt } from '../core/tracks.js';
import { createNoise2D, mulberry32, smoothstep, clamp } from '../core/math.js';
import { mudRoadTextures, groundDetailTextures, waterNormal, grassBlades, windowGrid, waterfallTexture } from './textures.js';

const OVERCAST = { rain: '#8d969c', snow: '#c9d0d6' };

function buildSky(scene, renderer, def, weather) {
  const sun = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - def.sun.elevation), THREE.MathUtils.degToRad(def.sun.azimuth));
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  let sky = null;
  if (weather === 'clear') {
    sky = new Sky();
    sky.scale.setScalar(20000);
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
    scene.fog = new THREE.FogExp2(color, weather === 'snow' ? 0.0065 : 0.0052);
  }
  const env = pmrem.fromScene(envScene, 0.02).texture;
  scene.environment = env;
  pmrem.dispose();
  return { sun, sky };
}

function buildLights(scene, def, weather, quality) {
  const clear = weather === 'clear';
  const hemi = new THREE.HemisphereLight(clear ? '#cfe3ff' : '#c4ccd4', def.palette.dirt, clear ? 0.9 : 1.5);
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

function terrainColors(simWorld, def, weather) {
  const { terrain, track } = simWorld;
  const noise = createNoise2D(def.seed + 44);
  const pal = Object.fromEntries(Object.entries(def.palette).map(([k, v]) => [k, new THREE.Color(v)]));
  const sand = new THREE.Color('#b9a57e');
  const colors = new Float32Array(GRID * GRID * 3);
  const c = new THREE.Color();
  for (let gz = 0; gz < GRID; gz++) {
    for (let gx = 0; gx < GRID; gx++) {
      const i = gz * GRID + gx;
      const x = -HALF_EXTENT + gx * CELL;
      const z = -HALF_EXTENT + gz * CELL;
      const y = terrain.heights[i];
      const slope = slopeAt(terrain, x, z);
      const n = noise(x / 60, z / 60) * 0.5 + noise(x / 13, z / 13) * 0.25;
      c.copy(pal.grass).lerp(pal.dry, clamp(0.5 + n * 1.3, 0, 1));
      c.lerp(pal.rock, smoothstep(0.35, 0.8, slope));
      const d = terrain.roadDist[i];
      c.lerp(pal.dirt, 1 - smoothstep(track.halfWidth - 1, track.halfWidth + 7 + n * 6, d));
      if (terrain.water !== null) c.lerp(sand, 1 - smoothstep(terrain.water + 0.5, terrain.water + 3, y));
      if (weather === 'snow') c.lerp(pal.snow, clamp(0.85 - slope * 0.9 - (d < track.halfWidth + 2 ? 0.45 : 0) + n * 0.2, 0, 0.95));
      if (weather === 'rain') c.multiplyScalar(0.78);
      if (y > 150) c.lerp(pal.snow, smoothstep(150, 220, y) * (1 - smoothstep(0.9, 1.4, slope)));
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
  }
  return colors;
}

function buildTerrainMesh(simWorld, def, weather) {
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
  geo.setAttribute('color', new THREE.BufferAttribute(terrainColors(simWorld, def, weather), 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  const detail = groundDetailTextures(def.seed);
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, map: detail.map, normalMap: detail.normalMap, normalScale: new THREE.Vector2(0.9, 0.9), roughness: weather === 'rain' ? 0.6 : 0.96 });
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

function buildWater(def) {
  if (def.water === null) return null;
  const normal = waterNormal(def.seed);
  normal.repeat.set(300, 300);
  const mat = new THREE.MeshPhysicalMaterial({ color: def.id === 'tahoe' ? '#12506b' : '#2d4b58', roughness: 0.06, metalness: 0.2, normalMap: normal, normalScale: new THREE.Vector2(0.35, 0.35), clearcoat: 0.6 });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = def.water;
  mesh.receiveShadow = true;
  return mesh;
}

function buildHorizon(simWorld, def, weather) {
  const raw = simWorld.terrain.raw;
  const rings = 56;
  const segs = 240;
  const positions = [];
  const colors = [];
  const pal = Object.fromEntries(Object.entries(def.palette).map(([k, v]) => [k, new THREE.Color(v)]));
  const c = new THREE.Color();
  for (let r = 0; r <= rings; r++) {
    const radius = 560 + Math.pow(r / rings, 1.8) * 5200;
    for (let s = 0; s <= segs; s++) {
      const a = (s / segs) * Math.PI * 2;
      const x = Math.cos(a) * radius;
      const z = Math.sin(a) * radius;
      const inside = Math.max(Math.abs(x), Math.abs(z)) < HALF_EXTENT;
      const y = raw(x, z) - (inside ? 4 : 0);
      positions.push(x, y, z);
      c.copy(pal.grass).lerp(pal.rock, smoothstep(40, 160, y));
      c.lerp(pal.snow, smoothstep(weather === 'snow' ? 30 : 160, weather === 'snow' ? 90 : 240, y));
      colors.push(c.r, c.g, c.b);
    }
  }
  const index = [];
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segs; s++) {
      const a = r * (segs + 1) + s;
      const b = a + 1;
      const cc = a + segs + 1;
      const d = cc + 1;
      index.push(a, b, cc, b, d, cc);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }));
  mesh.receiveShadow = true;
  return mesh;
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

function treeGeometry(kind) {
  const parts = [];
  const t = (geo, color, x, y, z, rx = 0, rz = 0) => {
    geo.rotateX(rx);
    geo.rotateZ(rz);
    geo.translate(x, y, z);
    parts.push(colored(geo, color));
  };
  if (kind === 'pine') {
    t(new THREE.CylinderGeometry(0.22, 0.38, 5, 7), '#4a3526', 0, 2.5, 0);
    [[3.4, 5, 3.6], [2.8, 4.4, 6.4], [2.1, 3.8, 9], [1.3, 3.2, 11.4], [0.7, 2.4, 13.2]].forEach(([r, h, y]) => t(new THREE.ConeGeometry(r, h, 9), '#27402a', 0, y, 0));
  } else if (kind === 'sequoia') {
    t(new THREE.CylinderGeometry(0.9, 1.7, 22, 10), '#8a4a2c', 0, 11, 0);
    [[4.2, 7, 19], [3.6, 6, 23.5], [2.8, 5.5, 27.5], [1.8, 5, 31]].forEach(([r, h, y]) => t(new THREE.ConeGeometry(r, h, 9), '#2e4a2c', 0, y, 0));
  } else if (kind === 'cypress') {
    t(new THREE.CylinderGeometry(0.3, 0.55, 5, 7), '#4e3a2b', 0, 2.5, 0, 0, 0.12);
    [[3.6, 0, 6.6, 0], [3, 2.2, 7.6, 1.2], [2.8, -2.4, 7.2, -0.8], [2.4, 0.6, 9.2, 1.8]].forEach(([r, x, y, z]) => {
      const g = new THREE.IcosahedronGeometry(r, 1);
      g.scale(1.3, 0.55, 1.1);
      t(g, '#2b3d25', x, y, z);
    });
  } else {
    const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.6, 6, 0), new THREE.Vector3(1.4, 12, 0));
    t(new THREE.TubeGeometry(curve, 10, 0.26, 6), '#7a6248', 0, 0, 0);
    for (let k = 0; k < 9; k++) {
      const frond = new THREE.PlaneGeometry(0.8, 4.2, 1, 3);
      const pos = frond.attributes.position;
      for (let v = 0; v < pos.count; v++) {
        const yy = pos.getY(v) + 2.1;
        pos.setZ(v, -yy * yy * 0.09);
        pos.setY(v, yy);
      }
      frond.rotateX(-1.1);
      frond.rotateY((k / 9) * Math.PI * 2);
      t(frond, '#3f6a2e', 1.4, 12, 0);
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
  const treeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: def.trees.kind === 'palm' ? THREE.DoubleSide : THREE.FrontSide });
  group.add(instanced(treeGeometry(def.trees.kind), treeMat, props.trees, (t, p, e, s, color) => {
    p.set(t.x, t.y - 0.3, t.z);
    e.set((rand() - 0.5) * 0.06, t.rot, (rand() - 0.5) * 0.06);
    s.setScalar(t.scale);
    color.setHSL(0, 0, 0.75 + rand() * 0.4);
    if (snow) color.lerp(new THREE.Color('#ffffff'), 0.35);
    return true;
  }));
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
    const grassMat = new THREE.MeshStandardMaterial({ map: grassBlades(snow ? '#9aa08a' : def.palette.grass), alphaTest: 0.45, side: THREE.DoubleSide, roughness: 0.9 });
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

const INTERNATIONAL_ORANGE = '#b8392a';

function goldenGate() {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: INTERNATIONAL_ORANGE, roughness: 0.55, metalness: 0.3 });
  const towers = [-300, 470];
  const H = 150;
  const deckY = 46;
  for (const tx of towers) {
    for (const s of [-1, 1]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(9, H, 11), mat);
      leg.position.set(tx, H / 2, s * 13);
      g.add(leg);
    }
    for (const y of [deckY + 18, 88, 118, H - 4]) {
      const beam = new THREE.Mesh(new THREE.BoxGeometry(7, 6, 26), mat);
      beam.position.set(tx, y, 0);
      g.add(beam);
    }
  }
  const deck = new THREE.Mesh(new THREE.BoxGeometry(1700, 5, 30), mat);
  deck.position.set(85, deckY, 0);
  g.add(deck);
  const catenary = (x0, y0, x1, y1, sag) => {
    const pts = [];
    for (let k = 0; k <= 30; k++) {
      const t = k / 30;
      pts.push(new THREE.Vector3(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t - Math.sin(Math.PI * t) * sag, 0));
    }
    return pts;
  };
  const sections = [[-760, deckY + 4, -300, H, 20], [-300, H, 470, H, 92], [470, H, 930, deckY + 4, 20]];
  const lines = [];
  for (const s of [-1, 1]) {
    for (const [x0, y0, x1, y1, sag] of sections) {
      const pts = catenary(x0, y0, x1, y1, sag).map((p) => p.setZ(s * 13));
      const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 40, 1.1, 6), mat);
      g.add(tube);
      for (let k = 1; k < pts.length - 1; k++) lines.push(pts[k].x, pts[k].y, pts[k].z, pts[k].x, deckY + 2, pts[k].z);
    }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  g.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: INTERNATIONAL_ORANGE })));
  g.position.set(-150, 0, -1320);
  g.rotation.y = 0.08;
  return g;
}

function skyline(raw, seed, x, z, count, spread, maxH, extras) {
  const g = new THREE.Group();
  const rand = mulberry32(seed);
  const tex = windowGrid(seed);
  for (let k = 0; k < count; k++) {
    const h = 30 + rand() * rand() * maxH;
    const w = 14 + rand() * 22;
    const t = tex.clone();
    t.repeat.set(w / 16, h / 24);
    t.needsUpdate = true;
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, w * (0.7 + rand() * 0.6)), new THREE.MeshStandardMaterial({ map: t, roughness: 0.3, metalness: 0.5 }));
    const bx = x + (rand() - 0.5) * spread;
    const bz = z + (rand() - 0.5) * spread;
    b.position.set(bx, raw(bx, bz) + h / 2 - 4, bz);
    g.add(b);
  }
  extras(g);
  return g;
}

function sfSkyline(raw) {
  return skyline(raw, 71, 1900, 250, 40, 380, 170, (g) => {
    const base = raw(1900, 250);
    const pyramid = new THREE.Mesh(new THREE.ConeGeometry(22, 190, 4), new THREE.MeshStandardMaterial({ color: '#e8e4dc', roughness: 0.5 }));
    pyramid.position.set(1850, base + 90, 180);
    pyramid.rotation.y = Math.PI / 4;
    g.add(pyramid);
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(16, 22, 240, 24), new THREE.MeshStandardMaterial({ color: '#b8c4cc', roughness: 0.2, metalness: 0.8 }));
    tower.position.set(1960, base + 116, 300);
    g.add(tower);
  });
}

function laSkyline(raw) {
  return skyline(raw, 83, 1800, -1500, 34, 320, 230, (g) => {
    const wilshire = new THREE.Mesh(new THREE.CylinderGeometry(10, 18, 320, 4), new THREE.MeshStandardMaterial({ color: '#9fb4c4', roughness: 0.15, metalness: 0.9 }));
    wilshire.position.set(1790, raw(1790, -1480) + 156, -1480);
    g.add(wilshire);
  });
}

function observatory(raw) {
  const g = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: '#efece4', roughness: 0.6 });
  const copper = new THREE.MeshStandardMaterial({ color: '#4b7a6a', roughness: 0.4, metalness: 0.6 });
  const main = new THREE.Mesh(new THREE.BoxGeometry(70, 16, 26), white);
  main.position.y = 8;
  g.add(main);
  const dome = new THREE.Mesh(new THREE.SphereGeometry(14, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), copper);
  dome.position.y = 18;
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(14, 14, 6, 32), white);
  drum.position.y = 16;
  g.add(drum, dome);
  for (const s of [-1, 1]) {
    const small = new THREE.Mesh(new THREE.SphereGeometry(7, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), copper);
    small.position.set(s * 32, 16, 0);
    const d2 = new THREE.Mesh(new THREE.CylinderGeometry(7, 7, 4, 24), white);
    d2.position.set(s * 32, 14, 0);
    g.add(small, d2);
  }
  const x = -520;
  const z = 980;
  g.position.set(x, raw(x, z) - 2, z);
  g.rotation.y = 0.4;
  return g;
}

function granite() {
  return new THREE.MeshStandardMaterial({ color: '#c3beb3', roughness: 0.85, flatShading: true });
}

function halfDome(raw) {
  const g = new THREE.Group();
  const geo = new THREE.SphereGeometry(360, 48, 24, Math.PI / 2, Math.PI, 0, Math.PI / 2);
  const face = new THREE.CircleGeometry(360, 48, 0, Math.PI);
  face.rotateY(Math.PI / 2);
  const noise = createNoise2D(9);
  const p = geo.attributes.position;
  for (let v = 0; v < p.count; v++) {
    const k = 1 + noise(p.getX(v) / 80, p.getY(v) / 80 + p.getZ(v) / 80) * 0.05;
    p.setXYZ(v, p.getX(v) * k, p.getY(v) * k, p.getZ(v) * k);
  }
  geo.computeVertexNormals();
  g.add(new THREE.Mesh(geo, granite()), new THREE.Mesh(face, new THREE.MeshStandardMaterial({ color: '#b3ada2', roughness: 0.9 })));
  g.scale.set(1, 1.35, 0.85);
  g.position.set(1650, raw(1650, 120) - 30, 120);
  g.rotation.y = Math.PI;
  return g;
}

function elCapitan(raw) {
  const geo = new THREE.BoxGeometry(380, 520, 260, 12, 16, 8);
  const p = geo.attributes.position;
  const noise = createNoise2D(21);
  for (let v = 0; v < p.count; v++) {
    const x = p.getX(v);
    const y = p.getY(v);
    const z = p.getZ(v);
    const top = (y + 260) / 520;
    const taper = 1 - top * 0.25;
    const bulge = noise(x / 90, y / 90) * 18 + noise(y / 40, z / 40) * 6;
    p.setXYZ(v, x * taper, y + (top > 0.98 ? noise(x / 60, z / 60) * 30 : 0), z * taper + (z > 0 ? bulge : 0));
  }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, granite());
  m.position.set(-380, raw(-380, -800) + 60, -800);
  return m;
}

function yosemiteFalls(raw) {
  const g = new THREE.Group();
  const tex = waterfallTexture();
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide });
  const x = 260;
  const z = -486;
  const top = raw(x, -650);
  const fall = new THREE.Mesh(new THREE.PlaneGeometry(26, top - 12), mat);
  fall.position.set(x, (top + 12) / 2, z + 1);
  g.add(fall);
  const mist = new THREE.Mesh(new THREE.SphereGeometry(22, 16, 8), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.25, depthWrite: false }));
  mist.position.set(x, 14, z + 10);
  g.add(mist);
  g.userData.tex = tex;
  return g;
}

function boathouse(raw) {
  const g = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: '#6b4a31', roughness: 0.85 });
  const roof = new THREE.MeshStandardMaterial({ color: '#3c4a3f', roughness: 0.7 });
  let x = 480;
  while (x < 800 && raw(x, 60) > 1) x += 4;
  const house = new THREE.Mesh(new THREE.BoxGeometry(14, 7, 10), wood);
  house.position.set(x - 10, raw(x - 10, 60) + 3.5, 60);
  const top = new THREE.Mesh(new THREE.ConeGeometry(10, 5, 4), roof);
  top.position.set(x - 10, raw(x - 10, 60) + 9.5, 60);
  top.rotation.y = Math.PI / 4;
  const pier = new THREE.Mesh(new THREE.BoxGeometry(60, 0.6, 4), wood);
  pier.position.set(x + 24, 1.2, 60);
  g.add(house, top, pier);
  for (let k = 0; k < 8; k++) {
    const pile = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 4, 6), wood);
    pile.position.set(x + k * 7.5, -0.6, 58);
    g.add(pile);
  }
  g.traverse((o) => {
    if (o.isMesh) o.castShadow = true;
  });
  return g;
}

const LANDMARKS = { goldenGate, sfSkyline, laSkyline, observatory, halfDome, elCapitan, yosemiteFalls, boathouse };

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
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(12000, 12000), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, fog: false, opacity: weather === 'clear' ? 0.75 : 0.95, side: THREE.DoubleSide }));
  mesh.rotation.x = Math.PI / 2;
  mesh.position.y = 700;
  mesh.renderOrder = -1;
  return mesh;
}

export function buildWorld(scene, renderer, simWorld, weather, quality) {
  const def = simWorld.track.def;
  const { sun: sunDir } = buildSky(scene, renderer, def, weather);
  const sun = buildLights(scene, def, weather, quality);
  scene.add(buildTerrainMesh(simWorld, def, weather));
  scene.add(buildRoad(simWorld, def, weather));
  scene.add(buildPuddles(simWorld, weather));
  const water = buildWater(def);
  if (water) scene.add(water);
  scene.add(buildHorizon(simWorld, def, weather));
  scene.add(buildProps(simWorld, def, weather, quality));
  scene.add(buildStartArch(simWorld));
  const landmarks = def.landmarks.map((name) => LANDMARKS[name](simWorld.terrain.raw));
  landmarks.forEach((l) => scene.add(l));
  const clouds = buildClouds(def, weather);
  scene.add(clouds);
  const falls = landmarks.find((l) => l.userData.tex);
  let t = 0;
  return {
    sun,
    update(dt, target) {
      t += dt;
      sun.position.set(target.x + sunDir.x * 300, target.y + sunDir.y * 300, target.z + sunDir.z * 300);
      sun.target.position.copy(target);
      if (water) water.material.normalMap.offset.set(t * 0.004, t * 0.003);
      clouds.material.map.offset.x = t * 0.0015;
      if (falls) falls.userData.tex.offset.y = t * 0.6;
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
