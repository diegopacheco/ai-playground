import { mulberry32 } from './math.js';
import { HALF_EXTENT, roadDistanceAt, slopeAt } from './terrain.js';

const OBSTACLE_CELL = 10;

function scatter(rand, count, tries, accept) {
  const out = [];
  for (let t = 0; t < tries && out.length < count; t++) {
    const x = (rand() * 2 - 1) * (HALF_EXTENT - 10);
    const z = (rand() * 2 - 1) * (HALF_EXTENT - 10);
    const item = accept(x, z);
    if (item) out.push(item);
  }
  return out;
}

function clusterForest(track) {
  const seed = track.def.seed;
  return (x, z) => (Math.sin(x * 0.013 + seed) * Math.cos(z * 0.011 - seed) > -0.15 ? 0.9 : 0.2);
}

export function placeProps(track, terrain, forestAt = clusterForest(track)) {
  const rand = mulberry32(track.def.seed * 31 + 7);
  const half = track.halfWidth;
  const aboveWater = (y) => terrain.water === null || y > terrain.water + 1.2;
  const NEAR_BAND = 180;
  const trees = scatter(rand, track.def.trees.count, track.def.trees.count * 14, (x, z) => {
    const d = roadDistanceAt(terrain, x, z);
    if (d < half + 9 || d > NEAR_BAND) return null;
    const y = terrain.heightAt(x, z);
    if (!aboveWater(y) || slopeAt(terrain, x, z) > 0.75) return null;
    if (rand() > forestAt(x, z)) return null;
    const scale = 0.75 + rand() * 0.6;
    return { x, y, z, scale, rot: rand() * Math.PI * 2, r: 0.5 * scale + 0.3 };
  });
  const forest = scatter(rand, track.def.trees.fill, track.def.trees.fill * 6, (x, z) => {
    const d = roadDistanceAt(terrain, x, z);
    if (d <= NEAR_BAND) return null;
    const y = terrain.heightAt(x, z);
    if (!aboveWater(y) || slopeAt(terrain, x, z) > 0.9) return null;
    if (rand() > forestAt(x, z)) return null;
    return { x, y, z, scale: 0.8 + rand() * 0.6, rot: rand() * Math.PI * 2 };
  });
  const rocks = scatter(rand, 420, 5000, (x, z) => {
    const d = roadDistanceAt(terrain, x, z);
    if (d < half + 6) return null;
    const y = terrain.heightAt(x, z);
    if (!aboveWater(y)) return null;
    const scale = 0.6 + rand() * rand() * 3.5;
    return { x, y, z, scale, rot: rand() * Math.PI * 2, r: scale * 0.8 };
  });
  const grass = scatter(rand, 5000, 30000, (x, z) => {
    const d = roadDistanceAt(terrain, x, z);
    if (d < half + 0.8 || d > half + 40) return null;
    const y = terrain.heightAt(x, z);
    if (!aboveWater(y)) return null;
    return { x, y, z, scale: 0.6 + rand() * 0.9, rot: rand() * Math.PI };
  });
  const markers = [];
  const every = Math.round(24 / track.spacing);
  for (let i = 0; i < track.count; i += every) {
    const h = track.heading[i];
    const rx = -Math.cos(h);
    const rz = Math.sin(h);
    for (const side of [-1, 1]) {
      const x = track.xs[i] + rx * side * (half + 1.3);
      const z = track.zs[i] + rz * side * (half + 1.3);
      markers.push({ x, z, y: terrain.heightAt(x, z), heading: h });
    }
  }
  const obstacles = [...trees, ...rocks.filter((r) => r.scale > 1.1)];
  return { trees, forest, rocks, grass, markers, obstacles, obstacleHash: hashObstacles(obstacles) };
}

function hashObstacles(list) {
  const map = new Map();
  for (const o of list) {
    const key = `${Math.floor(o.x / OBSTACLE_CELL)},${Math.floor(o.z / OBSTACLE_CELL)}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(o);
  }
  return map;
}

export function obstaclesNear(props, x, z) {
  const cx = Math.floor(x / OBSTACLE_CELL);
  const cz = Math.floor(z / OBSTACLE_CELL);
  const out = [];
  for (let a = -1; a <= 1; a++) {
    for (let b = -1; b <= 1; b++) {
      const list = props.obstacleHash.get(`${cx + a},${cz + b}`);
      if (list) out.push(...list);
    }
  }
  return out;
}
