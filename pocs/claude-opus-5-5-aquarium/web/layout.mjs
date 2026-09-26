import { TANK, DECOR, MAX_GRASS } from './catalog.mjs';
import { rng } from './swim.mjs';

export const CLUMPS_PER_LEVEL = 8;
export const AIR_STONE = [TANK.width / 2 - 0.05, -TANK.depth / 2 + 0.05, 0.035];

export function grassSpots() {
  const random = rng(42);
  const blocked = [...DECOR.flatMap(d => d.spots), AIR_STONE];
  const out = [];
  let tries = 0;
  while (out.length < MAX_GRASS * CLUMPS_PER_LEVEL && tries < 20000) {
    tries++;
    const x = (random() - 0.5) * (TANK.width - 0.08);
    const z = (random() - 0.5) * (TANK.depth - 0.08);
    if (blocked.some(([bx, bz, r]) => Math.hypot(x - bx, z - bz) < r + 0.025)) continue;
    out.push({ x, z, order: z + random() * 0.12 });
  }
  return out.sort((a, b) => a.order - b.order).map(({ x, z }) => ({ x, z }));
}
