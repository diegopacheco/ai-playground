import { FOLD_TIME, UNFOLD_TIME, clamp01 } from "./timeline";

export const DEPART_TIME = FOLD_TIME;
export const ARRIVE_TIME = UNFOLD_TIME;
export const LEAF_START = 0.12;
export const LEAF_END = 0.88;

function ease(t: number): number {
  const x = clamp01(t);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

export function departOpenness(t: number): number {
  return 1 - ease(t / FOLD_TIME);
}

export function leafAt(flight: number): number {
  return ease((flight - LEAF_START) / (LEAF_END - LEAF_START));
}

export function arriveOpenness(t: number): number {
  return clamp01(t / UNFOLD_TIME);
}

export function bank(flight: number, direction: number): { roll: number; pitch: number; lift: number } {
  const swing = Math.sin(Math.PI * clamp01(flight));
  return { roll: swing * 0.18 * direction, pitch: -swing * 0.12, lift: swing * 0.6 };
}
