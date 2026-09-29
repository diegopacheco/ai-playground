import { smoothstep, wrapAngle, mulberry32 } from './math.js';

const edge = (x, z, from, to) => smoothstep(from, to, Math.max(Math.abs(x), Math.abs(z)));

export const TRACKS = [
  {
    id: 'sf',
    name: 'Presidio Mud Run',
    city: 'San Francisco',
    blurb: 'Foggy cypress ridges above the bay, Golden Gate on the horizon.',
    seed: 11,
    width: 13,
    hills: 30,
    hillScale: 260,
    water: 0,
    jumps: [0.2, 0.63],
    puddles: 26,
    sun: { elevation: 9, azimuth: 235 },
    fog: { color: '#aeb9c2', density: 0.0034 },
    palette: { grass: '#56663a', dry: '#8b7d52', rock: '#6d675e', dirt: '#5b4431', snow: '#eef2f5' },
    trees: { kind: 'cypress', count: 1500 },
    landmarks: ['goldenGate', 'sfSkyline'],
    points: [[-380, 130], [-300, 320], [-110, 380], [40, 280], [140, 380], [320, 340], [410, 170], [300, 20], [390, -150], [270, -330], [80, -250], [-60, -380], [-260, -350], [-380, -190], [-250, -30]],
    macro: (x, z) => 18 - smoothstep(-560, -760, z) * 36 + edge(x, z, 620, 900) * 120 * (1 - smoothstep(-450, -560, z)) * (1 - 0.8 * smoothstep(400, 800, x)) + smoothstep(-1250, -1550, z) * 190,
  },
  {
    id: 'la',
    name: 'Griffith Canyon Rally',
    city: 'Los Angeles',
    blurb: 'Dusty chaparral canyons, palms and the downtown skyline at golden hour.',
    seed: 23,
    width: 13,
    hills: 38,
    hillScale: 300,
    water: null,
    jumps: [0.34, 0.8],
    puddles: 14,
    sun: { elevation: 6, azimuth: 250 },
    fog: { color: '#d9c3a0', density: 0.0022 },
    palette: { grass: '#7a7440', dry: '#a58a58', rock: '#8a7560', dirt: '#6e4f33', snow: '#f1f1ee' },
    trees: { kind: 'palm', count: 900 },
    landmarks: ['observatory', 'laSkyline'],
    points: [[-400, -60], [-370, 200], [-210, 390], [0, 300], [150, 400], [360, 310], [410, 90], [250, -40], [390, -220], [230, -390], [20, -310], [-160, -400], [-350, -300]],
    macro: (x, z) => 20 + edge(x, z, 560, 900) * 160 + smoothstep(300, 900, x) * 25,
  },
  {
    id: 'tahoe',
    name: 'Emerald Bay Trail',
    city: 'Lake Tahoe',
    blurb: 'Alpine pine forest on the shore of a sapphire lake ringed by Sierra peaks.',
    seed: 37,
    width: 12,
    hills: 26,
    hillScale: 220,
    water: 0,
    jumps: [0.27, 0.71],
    puddles: 30,
    sun: { elevation: 22, azimuth: 210 },
    fog: { color: '#b8c8d6', density: 0.0018 },
    palette: { grass: '#3f5a32', dry: '#6b6a45', rock: '#6a6d70', dirt: '#4e3b2b', snow: '#f3f6fa' },
    trees: { kind: 'pine', count: 2600 },
    landmarks: ['boathouse'],
    points: [[-420, 0], [-330, 280], [-80, 400], [160, 320], [360, 380], [430, 150], [300, -40], [420, -280], [180, -420], [-40, -300], [-220, -410], [-400, -240]],
    macro: (x, z) => 16 - smoothstep(520, 700, x) * 40 + edge(x, z, 600, 900) * (x > 520 ? 0 : 210) + smoothstep(2200, 2900, x) * 420,
  },
  {
    id: 'yosemite',
    name: 'Yosemite Valley Floor',
    city: 'Yosemite',
    blurb: 'Meadows and giant sequoias under El Capitan and Half Dome granite walls.',
    seed: 53,
    width: 12,
    hills: 16,
    hillScale: 200,
    water: null,
    jumps: [0.15, 0.55],
    puddles: 22,
    sun: { elevation: 28, azimuth: 160 },
    fog: { color: '#c3cfd8', density: 0.0014 },
    palette: { grass: '#4f6a34', dry: '#86804f', rock: '#9c978d', dirt: '#5a4632', snow: '#f4f7fa' },
    trees: { kind: 'sequoia', count: 1800 },
    landmarks: ['halfDome', 'elCapitan', 'yosemiteFalls'],
    points: [[-440, -120], [-420, 160], [-250, 300], [-60, 200], [120, 330], [330, 260], [440, 60], [380, -200], [190, -140], [40, -330], [-150, -240], [-300, -350]],
    macro: (x, z) => 10 + smoothstep(470, 640, Math.abs(z)) * 260 + smoothstep(560, 760, Math.abs(x)) * 140,
  },
];

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
  const { xs, zs, length, count } = resample(denseLoop(def.points));
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
