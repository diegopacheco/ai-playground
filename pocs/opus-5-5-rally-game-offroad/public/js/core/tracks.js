import { wrapAngle, mulberry32 } from './math.js';

const TWISTY = [[-1, 0], [-0.85, 0.8], [-0.45, 0.95], [-0.15, 0.45], [0.15, 0.95], [0.55, 0.9], [0.92, 0.6], [1, -0.1], [0.8, -0.85], [0.35, -0.95], [0.05, -0.45], [-0.3, -0.95], [-0.7, -0.9]];
const FLOWING = [[-1, 0], [-0.9, 0.75], [-0.6, 1], [-0.2, 0.85], [0.25, 1], [0.65, 0.95], [0.93, 0.7], [1, 0], [0.93, -0.7], [0.6, -1], [0.2, -0.75], [-0.25, -1], [-0.65, -0.95], [-0.93, -0.7]];
const SHORE = [[-1, 0], [-0.85, 0.85], [-0.4, 1], [0, 0.6], [0.35, 1], [0.8, 0.9], [1, 0.2], [0.85, -0.55], [0.5, -0.3], [0.2, -0.95], [-0.35, -0.95], [-0.8, -0.8]];

export const TRACKS = [
  {
    id: 'sf',
    name: 'Crissy Field Mud Run',
    city: 'San Francisco',
    blurb: 'Bayfront flats below the Presidio, the Golden Gate Bridge towering to the northwest and Alcatraz out in the bay.',
    seed: 11,
    width: 16,
    water: 0,
    frame: { cx: 70, cz: -105, angle: 0, hu: 560, hv: 100 },
    shape: FLOWING,
    jumps: [0.2, 0.63],
    puddles: 26,
    sun: { elevation: 9, azimuth: 250 },
    fog: { color: '#b4c0c8', density: 0.00022 },
    palette: { grass: '#5c7d3c', dry: '#9e8f5e', forest: '#4a3f2e', sand: '#cdb892', rock: '#6d675e', dirt: '#5b4431', snow: '#eef2f5' },
    trees: { kind: 'cypress', count: 1800, fill: 9000 },
    landmarks: ['goldenGate', 'sfSkyline', 'alcatraz', 'palaceOfFineArts'],
  },
  {
    id: 'la',
    name: 'Griffith Park Rally',
    city: 'Los Angeles',
    blurb: 'Dusty flats of Griffith Park under the Hollywood Sign hills and the Observatory, downtown towers on the southern horizon.',
    seed: 23,
    width: 16,
    water: null,
    frame: { cx: 330, cz: -60, angle: 90, hu: 520, hv: 230 },
    shape: TWISTY,
    jumps: [0.34, 0.8],
    puddles: 14,
    sun: { elevation: 7, azimuth: 255 },
    fog: { color: '#d8c6a6', density: 0.00016 },
    palette: { grass: '#6f7340', dry: '#a88f5c', forest: '#5a4a34', sand: '#c2a87a', rock: '#8a7560', dirt: '#6e4f33', snow: '#f1f1ee' },
    trees: { kind: 'palm', count: 700, fill: 5000 },
    landmarks: ['hollywoodSign', 'observatory', 'laSkyline'],
  },
  {
    id: 'tahoe',
    name: 'Pope Beach Trail',
    city: 'Lake Tahoe',
    blurb: 'Pine forest on the south shore right beside the lake, Mount Tallac rising behind and the Sierra ringing the water.',
    seed: 37,
    width: 16,
    water: 1897.5,
    frame: { cx: 0, cz: 150, angle: 13, hu: 600, hv: 160 },
    shape: SHORE,
    jumps: [0.27, 0.71],
    puddles: 30,
    sun: { elevation: 24, azimuth: 215 },
    fog: { color: '#b8c8d6', density: 0.00012 },
    palette: { grass: '#4f6b35', dry: '#7b7449', forest: '#3d4a2a', sand: '#c9b894', rock: '#6a6d70', dirt: '#4e3b2b', snow: '#f3f6fa' },
    trees: { kind: 'pine', count: 6000, fill: 22000 },
    landmarks: ['boathouse'],
  },
  {
    id: 'yosemite',
    name: 'Yosemite Valley Floor',
    city: 'Yosemite',
    blurb: 'Meadows on the valley floor between El Capitan and Yosemite Falls, Half Dome at the head of the valley.',
    seed: 53,
    width: 16,
    water: null,
    frame: { cx: 120, cz: 120, angle: -45, hu: 560, hv: 150 },
    shape: TWISTY,
    jumps: [0.15, 0.55],
    puddles: 22,
    sun: { elevation: 30, azimuth: 200 },
    fog: { color: '#c3cfd8', density: 0.00009 },
    palette: { grass: '#62803a', dry: '#8c8250', forest: '#3e4a2b', sand: '#b9ab8c', rock: '#a9a49a', dirt: '#5a4632', snow: '#f4f7fa' },
    trees: { kind: 'pine', count: 6000, fill: 20000 },
    landmarks: ['yosemiteFalls', 'bridalveilFall'],
  },
];

export function trackPoints(def) {
  const { cx, cz, angle, hu, hv } = def.frame;
  const a = (angle * Math.PI) / 180;
  return def.shape.map(([u, v]) => [cx + u * hu * Math.cos(a) - v * hv * Math.sin(a), cz + u * hu * Math.sin(a) + v * hv * Math.cos(a)]);
}

export const SPACING = 2;
const HASH_CELL = 24;

function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
  return [f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])];
}

function denseLoop(points) {
  const out = [];
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const p0 = points[(i - 1 + n) % n];
    const p1 = points[i];
    const p2 = points[(i + 1) % n];
    const p3 = points[(i + 2) % n];
    for (let k = 0; k < 60; k++) out.push(catmull(p0, p1, p2, p3, k / 60));
  }
  return out;
}

function resample(dense) {
  const n = dense.length;
  const cum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const a = dense[i];
    const b = dense[(i + 1) % n];
    cum[i + 1] = cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  const total = cum[n];
  const count = Math.round(total / SPACING);
  const xs = new Float32Array(count);
  const zs = new Float32Array(count);
  let j = 0;
  for (let i = 0; i < count; i++) {
    const s = (i / count) * total;
    while (cum[j + 1] < s) j++;
    const t = (s - cum[j]) / (cum[j + 1] - cum[j] || 1);
    const a = dense[j];
    const b = dense[(j + 1) % n];
    xs[i] = a[0] + (b[0] - a[0]) * t;
    zs[i] = a[1] + (b[1] - a[1]) * t;
  }
  return { xs, zs, length: total, count };
}

function smoothLoop(values, radius) {
  const n = values.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += values[(i + k + n) % n];
    out[i] = sum / (radius * 2 + 1);
  }
  return out;
}

export function buildTrack(def) {
  const { xs, zs, length, count } = resample(denseLoop(trackPoints(def)));
  const heading = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const a = (i - 1 + count) % count;
    const b = (i + 1) % count;
    heading[i] = Math.atan2(xs[b] - xs[a], zs[b] - zs[a]);
  }
  const rawCurv = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const b = (i + 2) % count;
    const a = (i - 2 + count) % count;
    rawCurv[i] = wrapAngle(heading[b] - heading[a]) / (SPACING * 4);
  }
  const curvature = smoothLoop(rawCurv, 4);
  const vmaxUnit = new Float32Array(count);
  for (let i = 0; i < count; i++) vmaxUnit[i] = Math.min(60, Math.sqrt(9.81 / Math.max(Math.abs(curvature[i]), 1e-4)));

  const hash = new Map();
  for (let i = 0; i < count; i++) {
    const key = cellKey(Math.floor(xs[i] / HASH_CELL), Math.floor(zs[i] / HASH_CELL));
    if (!hash.has(key)) hash.set(key, []);
    hash.get(key).push(i);
  }

  const track = { def, xs, zs, heading, curvature, vmaxUnit, length, count, spacing: length / count, halfWidth: def.width / 2, hash };
  track.jumps = def.jumps.map((f) => straightestNear(track, Math.round(f * count)));
  track.puddles = placePuddles(track);
  return track;
}

function straightestNear(track, center) {
  let best = center;
  let bestScore = Infinity;
  for (let d = -150; d <= 150; d += 3) {
    const c = (center + d + track.count) % track.count;
    let score = 0;
    for (let k = -30; k <= 90; k++) score = Math.max(score, Math.abs(track.curvature[(c + k + track.count) % track.count]));
    if (score < bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

function cellKey(cx, cz) {
  return (cx + 1000) * 4096 + (cz + 1000);
}

export function rightAt(track, i) {
  const h = track.heading[i];
  return [-Math.cos(h), Math.sin(h)];
}

function scan(track, x, z, from, to, best) {
  for (let k = from; k <= to; k++) {
    const i = ((k % track.count) + track.count) % track.count;
    const dx = x - track.xs[i];
    const dz = z - track.zs[i];
    const d = dx * dx + dz * dz;
    if (d < best.d) {
      best.d = d;
      best.i = i;
    }
  }
}

export function nearestIndex(track, x, z, hint = -1) {
  const best = { d: Infinity, i: -1 };
  if (hint >= 0) {
    scan(track, x, z, hint - 50, hint + 50, best);
    if (best.d < 30 * 30) return best.i;
  }
  const cx = Math.floor(x / HASH_CELL);
  const cz = Math.floor(z / HASH_CELL);
  for (let a = -1; a <= 1; a++) {
    for (let b = -1; b <= 1; b++) {
      const list = track.hash.get(cellKey(cx + a, cz + b));
      if (!list) continue;
      for (const i of list) {
        const dx = x - track.xs[i];
        const dz = z - track.zs[i];
        const d = dx * dx + dz * dz;
        if (d < best.d) {
          best.d = d;
          best.i = i;
        }
      }
    }
  }
  if (best.i >= 0 && best.d < HASH_CELL * HASH_CELL) return best.i;
  scan(track, x, z, 0, track.count - 1, best);
  return best.i;
}

export function nearbyDistance(track, x, z) {
  const cx = Math.floor(x / HASH_CELL);
  const cz = Math.floor(z / HASH_CELL);
  let best = Infinity;
  let index = -1;
  for (let a = -1; a <= 1; a++) {
    for (let b = -1; b <= 1; b++) {
      const list = track.hash.get(cellKey(cx + a, cz + b));
      if (!list) continue;
      for (const i of list) {
        const dx = x - track.xs[i];
        const dz = z - track.zs[i];
        const d = dx * dx + dz * dz;
        if (d < best) {
          best = d;
          index = i;
        }
      }
    }
  }
  return { dist: Math.sqrt(best), index };
}

export function project(track, x, z, hint = -1) {
  const i = nearestIndex(track, x, z, hint);
  const [rx, rz] = rightAt(track, i);
  const dx = x - track.xs[i];
  const dz = z - track.zs[i];
  const lateral = dx * rx + dz * rz;
  return { i, s: i * track.spacing, lateral, dist: Math.abs(lateral) };
}

function placePuddles(track) {
  const rand = mulberry32(track.def.seed * 7 + 1);
  const out = [];
  for (let k = 0; k < track.def.puddles; k++) {
    let i = Math.floor((0.06 + rand() * 0.9) * track.count);
    if (track.jumps.some((j) => Math.abs(i - j) < 30)) i = (i + 60) % track.count;
    const lateral = (rand() - 0.5) * track.def.width * 0.6;
    const [rx, rz] = rightAt(track, i);
    out.push({ i, x: track.xs[i] + rx * lateral, z: track.zs[i] + rz * lateral, r: 1.6 + rand() * 2.4, stretch: 1.2 + rand() * 1.4, angle: track.heading[i] });
  }
  return out;
}

export function puddleAt(track, x, z) {
  for (const p of track.puddles) {
    const dx = x - p.x;
    const dz = z - p.z;
    if (dx * dx + dz * dz < p.r * p.r * p.stretch) return p;
  }
  return null;
}

export function gridSlots(track, cars) {
  const slots = [];
  for (let k = 0; k < cars; k++) {
    const i = Math.round((34 - k * 8) / track.spacing);
    const [rx, rz] = rightAt(track, i);
    const lateral = (k % 2 === 0 ? -1 : 1) * track.def.width * 0.22;
    slots.push({ x: track.xs[i] + rx * lateral, z: track.zs[i] + rz * lateral, heading: track.heading[i], i });
  }
  return slots;
}
