import { createNoise2D, lerp, smoothstep, clamp } from './math.js';
import { nearbyDistance } from './tracks.js';
import { NEAR } from './geo.js';

export const HALF_EXTENT = NEAR.half;
export const CELL = 4;
export const GRID = HALF_EXTENT * 2 / CELL + 1;
export const ROAD_OFFSET = 0.06;

export function rawHeightFn(def, geo) {
  const detail = createNoise2D(def.seed + 101);
  const base = Math.round(geo.elevation(0, 0));
  const lake = def.water === null ? -Infinity : def.water + 0.25;
  const fn = (x, z) => {
    const e = geo.elevation(x, z);
    return e - base + detail(x / 14, z / 14) * 0.3 - (e <= lake ? 2.5 : 0);
  };
  fn.base = base;
  return fn;
}

function roadProfile(track, raw, water) {
  const n = track.count;
  let ys = new Float32Array(n);
  for (let i = 0; i < n; i++) ys[i] = raw(track.xs[i], track.zs[i]);
  for (let pass = 0; pass < 3; pass++) {
    const out = new Float32Array(n);
    const r = 12;
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += ys[(i + k + n) % n];
      out[i] = sum / (r * 2 + 1);
    }
    ys = out;
  }
  const floor = water === null ? -Infinity : water + 1.2;
  const rampLen = 18 / track.spacing;
  for (let i = 0; i < n; i++) {
    ys[i] = Math.max(ys[i], floor);
    for (const c of track.jumps) {
      let d = i - c;
      if (d > n / 2) d -= n;
      if (d < -n / 2) d += n;
      if (d > -rampLen && d < rampLen * 0.35) {
        const t = d < 0 ? (d + rampLen) / rampLen : 1 - d / (rampLen * 0.35);
        ys[i] += t * t * 2.1;
      }
    }
  }
  return ys;
}

export function buildTerrain(track, geo) {
  const raw = rawHeightFn(track.def, geo);
  const water = track.def.water === null ? null : track.def.water - raw.base;
  const roadY = roadProfile(track, raw, water);
  const heights = new Float32Array(GRID * GRID);
  const roadDist = new Float32Array(GRID * GRID);
  const half = track.halfWidth;
  for (let gz = 0; gz < GRID; gz++) {
    const z = -HALF_EXTENT + gz * CELL;
    for (let gx = 0; gx < GRID; gx++) {
      const x = -HALF_EXTENT + gx * CELL;
      const { dist, index } = nearbyDistance(track, x, z);
      const r = raw(x, z);
      const w = index < 0 ? 1 : smoothstep(half + 1.5, half + 14, dist);
      const idx = gz * GRID + gx;
      heights[idx] = index < 0 ? r : lerp(roadY[index], r, w);
      roadDist[idx] = dist;
    }
  }
  const terrain = { raw, base: raw.base, roadY, heights, roadDist, water };
  terrain.heightAt = (x, z) => sampleGrid(terrain, heights, x, z, raw);
  return terrain;
}

function sampleGrid(terrain, field, x, z, fallback) {
  const fx = (x + HALF_EXTENT) / CELL;
  const fz = (z + HALF_EXTENT) / CELL;
  if (fx < 0 || fz < 0 || fx >= GRID - 1 || fz >= GRID - 1) return fallback(x, z);
  const x0 = Math.floor(fx);
  const z0 = Math.floor(fz);
  const tx = fx - x0;
  const tz = fz - z0;
  const i = z0 * GRID + x0;
  const a = field[i];
  const b = field[i + 1];
  const c = field[i + GRID];
  const d = field[i + GRID + 1];
  if (tx + tz <= 1) return a + (b - a) * tx + (c - a) * tz;
  return d + (c - d) * (1 - tx) + (b - d) * (1 - tz);
}

export function roadDistanceAt(terrain, x, z) {
  return sampleGrid(terrain, terrain.roadDist, x, z, () => Infinity);
}

export function slopeAt(terrain, x, z) {
  const e = CELL;
  const dx = (terrain.heightAt(x + e, z) - terrain.heightAt(x - e, z)) / (2 * e);
  const dz = (terrain.heightAt(x, z + e) - terrain.heightAt(x, z - e)) / (2 * e);
  return clamp(Math.hypot(dx, dz), 0, 10);
}
