import { clamp01, loop, smooth } from "../engine/timeline";

export const RABBIT_LOOP = 8;
export const HOLE_X = 2.6;

export function rabbitX(time: number): number {
  return -5.5 + (HOLE_X + 5.5) * smooth(loop(time, RABBIT_LOOP), 0, 5);
}

export function rabbitDive(time: number): number {
  return smooth(loop(time, RABBIT_LOOP), 5, 5.8);
}

export const SIZE_LOOP = 12;

export function shrinkSet(time: number): number {
  const t = loop(time, SIZE_LOOP);
  return 1 + 1.4 * (smooth(t, 2.5, 5) - smooth(t, 9, 11.5));
}

export function growAlice(time: number): number {
  const t = loop(time, SIZE_LOOP);
  return 1 + 1.4 * (smooth(t, 2.5, 5.5) - smooth(t, 9, 11.5));
}

export function shrinkRoom(time: number): number {
  const t = loop(time, SIZE_LOOP);
  return 1 - 0.4 * (smooth(t, 2.5, 5.5) - smooth(t, 9, 11.5));
}

export const CAT_LOOP = 11;

export function catBody(time: number): number {
  const t = loop(time, CAT_LOOP);
  return 1 - smooth(t, 3, 5.5) + smooth(t, 8.5, 10.5);
}

export function catGrin(time: number): number {
  const t = loop(time, CAT_LOOP);
  return 1 - smooth(t, 6.5, 7.5) + smooth(t, 8, 8.8);
}

export const SEATS = 6;
export const MOVE_EVERY = 3.2;

export function seatShift(time: number): number {
  const hop = Math.floor(time / MOVE_EVERY);
  const phase = clamp01((loop(time, MOVE_EVERY) - (MOVE_EVERY - 0.8)) / 0.8);
  return hop + phase;
}

export const ROSE_LOOP = 12;

export function roseRed(time: number, index: number, count: number): number {
  const t = loop(time, ROSE_LOOP);
  const start = 1 + (index / count) * 7;
  return smooth(t, start, start + 0.8) * (1 - smooth(t, 11, 11.8));
}
