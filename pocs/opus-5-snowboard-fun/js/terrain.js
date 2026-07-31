export const GRADE = 0.34;
export const COURSE_LENGTH = 2600;
export const PISTE_HALF = 52;
export const START_Z = -30;

const KICKERS = [
  { z: 240, x: 0, len: 22, w: 14, h: 3.4 },
  { z: 520, x: -18, len: 26, w: 12, h: 4.6 },
  { z: 760, x: 20, len: 24, w: 12, h: 4.0 },
  { z: 1480, x: 0, len: 30, w: 18, h: 6.2 },
  { z: 1820, x: -22, len: 24, w: 12, h: 4.4 },
  { z: 2080, x: 16, len: 28, w: 14, h: 5.4 },
  { z: 2380, x: 0, len: 32, w: 20, h: 7.0 }
];

const MOGUL_ZONES = [
  { from: 980, to: 1320 },
  { from: 1980, to: 2180 }
];

function hash(a, b) {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function smooth(t) {
  return t * t * (3 - 2 * t);
}

function zoneAmount(z, from, to) {
  const fade = 40;
  if (z < from - fade || z > to + fade) return 0;
  if (z < from) return smooth((z - from + fade) / fade);
  if (z > to) return smooth((to + fade - z) / fade);
  return 1;
}

export function height(x, z) {
  let y = -z * GRADE;

  y += Math.sin(z * 0.017) * 4.2 + Math.sin(z * 0.0061 + 1.7) * 7.5;
  y += Math.cos(x * 0.021 + z * 0.004) * 2.6;
  y += Math.sin(x * 0.083 + z * 0.037) * 0.42;
  y += Math.sin(z * 0.14) * 0.16;

  for (let i = 0; i < MOGUL_ZONES.length; i++) {
    const amount = zoneAmount(z, MOGUL_ZONES[i].from, MOGUL_ZONES[i].to);
    if (amount > 0) {
      y += amount * 0.3 * (1 + Math.sin(x * 0.78 + 0.4) * Math.sin(z * 0.5));
    }
  }

  const edge = Math.abs(x) - PISTE_HALF;
  if (edge > 0) y += Math.min(edge * edge * 0.014, 46);

  for (let i = 0; i < KICKERS.length; i++) {
    const k = KICKERS[i];
    const dx = Math.abs(x - k.x);
    if (dx > k.w) continue;
    const lateral = 0.5 * (1 + Math.cos((Math.PI * dx) / k.w));
    const t = (z - k.z) / k.len;
    if (t >= 0 && t <= 1) {
      y += k.h * lateral * (t * t * (0.35 + 0.65 * t));
    } else if (t > 1 && t < 3.4) {
      y -= k.h * lateral * 0.5 * smooth(1 - (t - 1) / 2.4);
    }
  }

  return y;
}

export function normal(x, z, out) {
  const e = 0.45;
  const hl = height(x - e, z);
  const hr = height(x + e, z);
  const hb = height(x, z - e);
  const hf = height(x, z + e);
  let nx = hl - hr;
  let ny = 2 * e;
  let nz = hb - hf;
  const inv = 1 / Math.hypot(nx, ny, nz);
  out.x = nx * inv;
  out.y = ny * inv;
  out.z = nz * inv;
  return out;
}

export function buildScenery() {
  const trees = [];
  const rocks = [];
  const banners = [];

  for (let z = -60; z < COURSE_LENGTH + 160; z += 7) {
    for (let side = -1; side <= 1; side += 2) {
      const r = hash(z, side * 31);
      if (r > 0.68) continue;
      const spread = 12 + hash(z, side * 77) * 78;
      const x = side * (PISTE_HALF + 6 + spread);
      const zz = z + hash(z, side * 13) * 6;
      trees.push({
        x,
        z: zz,
        y: height(x, zz),
        h: 6 + hash(z, side * 91) * 9,
        w: 1.5 + hash(z, side * 53) * 1.4,
        tone: hash(z, side * 17)
      });
    }
  }

  for (let z = 0; z < COURSE_LENGTH; z += 23) {
    const r = hash(z, 5001);
    if (r > 0.45) continue;
    const x = (r - 0.22) * 4 * PISTE_HALF;
    if (Math.abs(x) < 16) continue;
    rocks.push({ x, z, y: height(x, z), r: 0.7 + hash(z, 71) * 1.5 });
  }

  for (let z = 120; z < COURSE_LENGTH; z += 160) {
    banners.push({ z, y: height(0, z) });
  }

  const props = [];
  for (const t of trees) props.push({ kind: 0, z: t.z, o: t });
  for (const r of rocks) props.push({ kind: 1, z: r.z, o: r });
  for (const b of banners) props.push({ kind: 2, z: b.z, o: b });
  props.sort((a, b) => a.z - b.z);

  return { trees, rocks, banners, props };
}

export const kickers = KICKERS;
