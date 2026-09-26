import { TANK } from './catalog.mjs';

function hash(x, y, seed) {
  let h = (x * 374761393 + y * 668265263 + seed * 144269504) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function smooth(t) {
  return t * t * (3 - 2 * t);
}

export function noise(x, y, period = 64, seed = 1) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = smooth(x - xi);
  const fy = smooth(y - yi);
  const m = v => ((v % period) + period) % period;
  const a = hash(m(xi), m(yi), seed);
  const b = hash(m(xi + 1), m(yi), seed);
  const c = hash(m(xi), m(yi + 1), seed);
  const d = hash(m(xi + 1), m(yi + 1), seed);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

export function fbm(x, y, period = 8, seed = 1) {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  for (let o = 0; o < 4; o++) {
    sum += amp * noise(x, y, period, seed + o);
    norm += amp;
    x *= 2;
    y *= 2;
    period *= 2;
    amp *= 0.5;
  }
  return sum / norm;
}

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

function fract(v) {
  return v - Math.floor(v);
}

function cells(u, v, n) {
  const x = u * n;
  const y = v * n;
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  let d1 = 9;
  let d2 = 9;
  let id = 0;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const cx = ((xi + i) % n + n) % n;
      const cy = ((yi + j) % n + n) % n;
      const px = xi + i + hash(cx, cy, 7);
      const py = yi + j + hash(cx, cy, 11);
      const d = Math.hypot(px - x, py - y);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        id = hash(cx, cy, 3);
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  return { edge: d2 - d1, id };
}

const SHADES = {
  wood(u, v) {
    const warp = fbm(u * 2, v * 8, 2) * 2.2;
    const ring = fract(v * 14 + warp);
    const grain = noise(u * 180, v * 6, 180, 5);
    return clamp01(0.58 + 0.18 * Math.pow(ring, 3) - 0.14 * grain + 0.1 * fbm(u * 4, v * 4, 4, 9));
  },
  brushed(u, v) {
    return clamp01(0.72 + 0.16 * noise(u * 3, v * 320, 320, 2) + 0.08 * fbm(u * 2, v * 2, 2, 4));
  },
  bricks(u, v) {
    const rows = 10;
    const cols = 5;
    const row = Math.floor(v * rows);
    const x = u * cols + (row % 2) * 0.5;
    const bx = fract(x);
    const by = fract(v * rows);
    const mortar = 0.06;
    if (bx < mortar * 0.6 || by < mortar) return 0.22 + 0.1 * noise(u * 90, v * 90, 90, 3);
    const tone = hash(Math.floor(x) % cols, row, 13);
    return clamp01(0.5 + 0.25 * tone + 0.2 * fbm(u * 16, v * 16, 16, 6) - 0.1);
  },
  marble(u, v) {
    const turb = fbm(u * 4, v * 4, 4, 21) * 6;
    const vein = Math.pow(Math.abs(Math.sin((u + v) * 5 * Math.PI + turb)), 0.25);
    return clamp01(0.45 + 0.5 * vein + 0.05 * noise(u * 60, v * 60, 60, 8));
  },
  concrete(u, v) {
    const pit = noise(u * 140, v * 140, 140, 4) > 0.86 ? -0.25 : 0;
    return clamp01(0.55 + 0.3 * fbm(u * 8, v * 8, 8, 12) + pit);
  },
  bamboo(u, v) {
    const stalks = 6;
    const s = fract(u * stalks);
    const round = Math.sin(s * Math.PI);
    const id = Math.floor(u * stalks);
    const node = Math.abs(fract(v * 3 + hash(id, 0, 5)) - 0.5) < 0.012 ? -0.35 : 0;
    const gap = s < 0.03 || s > 0.97 ? -0.4 : 0;
    return clamp01(0.35 + 0.45 * round + node + gap + 0.08 * noise(u * 40, v * 300, 40, 2));
  },
  stone(u, v) {
    const c = cells(u, v, 5);
    if (c.edge < 0.06) return 0.18;
    return clamp01(0.45 + 0.35 * c.id + 0.2 * fbm(u * 12, v * 12, 12, 17) - 0.1);
  },
  carbon(u, v) {
    const n = 32;
    const x = u * n;
    const y = v * n;
    const cx = Math.floor(x);
    const cy = Math.floor(y);
    const twill = ((cx + Math.floor(cy / 2)) % 2 + 2) % 2;
    const along = twill ? fract(x) : fract(y);
    return clamp01(0.3 + 0.45 * Math.sin(along * Math.PI) * (twill ? 1 : 0.8));
  },
  leather(u, v) {
    const c = cells(u, v, 40);
    return clamp01(0.5 + 0.35 * Math.min(1, c.edge * 3) + 0.15 * fbm(u * 6, v * 6, 6, 30) - 0.15);
  }
};

export const PATTERNS = Object.keys(SHADES);

export function cellEdge(u, v, n) {
  return cells(fract(u), fract(v), n).edge;
}

export function shade(pattern, u, v) {
  const fn = SHADES[pattern];
  if (!fn) throw new Error(`unknown pattern ${pattern}`);
  return fn(fract(u), fract(v));
}

export function fishColor(fish, nx, ny) {
  const [base, a = base, b = a] = fish.colors;
  switch (fish.pattern) {
    case 'neon':
      if (ny > 0.02 && ny < 0.34 && nx > -0.8) return a;
      if (ny <= 0.02 && nx < 0.15) return b;
      return base;
    case 'clown':
      for (const c of [0.45, -0.1, -0.72]) {
        const d = Math.abs(nx - c);
        if (d < 0.09) return a;
        if (d < 0.13) return b;
      }
      return base;
    case 'tang':
      if (nx < -0.8) return b;
      if (ny > -0.05 && ny < 0.55 - 0.3 * Math.abs(nx + 0.1) && nx > -0.75 && nx < 0.45) return a;
      return base;
    case 'belly':
      return ny < -0.3 ? a : base;
    case 'bands':
      for (const c of [0.42, 0.02, -0.42]) if (Math.abs(nx - c) < 0.06) return a;
      return base;
    case 'fade':
      return nx > 0 ? base : a;
    case 'guppy':
      if (nx > -0.15) return base;
      return noise(nx * 9, ny * 9, 64, 44) > 0.62 ? b : a;
    case 'wavy':
      return Math.sin(ny * 20 + Math.sin(nx * 5) * 1.6) > 0.55 ? a : base;
    case 'psy':
      return Math.sin(nx * 14 + Math.sin(ny * 6) * 2.2) > 0.3 ? a : base;
    case 'shark':
      return ny < -0.2 ? '#eeeeea' : '#6d7a85';
    default:
      return base;
  }
}

export function sandHeight(x, z) {
  const back = (-z / TANK.depth + 0.5) * 0.045;
  return 0.03 + back + 0.012 * (fbm(x * 6 + 10, z * 6 + 10, 64, 3) - 0.5);
}

export function pathCenter(z) {
  return 0.03 + 0.07 * Math.sin(z * 9 + 0.6);
}

export function pathMask(x, z) {
  const t = z / TANK.depth + 0.5;
  const w = 0.022 + 0.03 * t + 0.006 * (noise(z * 60 + 3, 1, 64, 12) - 0.5);
  const d = Math.abs(x - pathCenter(z));
  return clamp01((w - d) / (w * 0.35));
}
