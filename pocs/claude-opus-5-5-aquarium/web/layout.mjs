import { TANK, DECOR, STONES, MAX_GRASS } from './catalog.mjs';
import { rng } from './swim.mjs';

export const CLUMPS_PER_LEVEL = 8;
export const AIR_STONE = [TANK.width / 2 - 0.05, -TANK.depth / 2 + 0.05, 0.035];
export const CARPET_STEP = 0.009;

export function covered(x, z, spots, shrink = 1) {
  return spots.some(([bx, bz, r]) => Math.hypot(x - bx, z - bz) < r * shrink);
}

export function grassSpots() {
  const random = rng(42);
  const blocked = [...DECOR.flatMap(d => d.spots), ...STONES, AIR_STONE].map(([x, z, r]) => [x, z, r + 0.025]);
  const out = [];
  let tries = 0;
  while (out.length < MAX_GRASS * CLUMPS_PER_LEVEL && tries < 20000) {
    tries++;
    const x = (random() - 0.5) * (TANK.width - 0.08);
    const z = (random() - 0.5) * (TANK.depth - 0.08);
    if (covered(x, z, blocked)) continue;
    out.push({ x, z, order: z + random() * 0.12 });
  }
  return out.sort((a, b) => a.order - b.order).map(({ x, z }) => ({ x, z }));
}

export function carpetSpots() {
  const random = rng(7);
  const out = [];
  const hx = TANK.width / 2 - 0.014;
  const hz = TANK.depth / 2 - 0.014;
  for (let x = -hx; x <= hx; x += CARPET_STEP) {
    for (let z = -hz; z <= hz; z += CARPET_STEP) {
      const px = x + (random() - 0.5) * CARPET_STEP;
      const pz = z + (random() - 0.5) * CARPET_STEP;
      if (Math.abs(px) > hx || Math.abs(pz) > hz || covered(px, pz, [AIR_STONE])) continue;
      out.push({ x: px, z: pz, turn: random() * Math.PI * 2 });
    }
  }
  return out;
}
