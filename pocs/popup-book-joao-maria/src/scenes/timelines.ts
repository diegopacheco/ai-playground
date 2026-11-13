export function smooth(t: number, a: number, b: number): number {
  const x = Math.min(Math.max((t - a) / (b - a), 0), 1);
  return x * x * (3 - 2 * x);
}

export const CRUMB_LOOP = 10;
export const WALK_START = -3.4;
export const WALK_END = 3.2;
export const BIRD_END = 2.2;
export const CRUMBS = Array.from({ length: 12 }, (_, i) => -3.0 + i * 0.4);

export function joaoX(c: number): number {
  return WALK_START + smooth(c, 0.3, 6.5) * (WALK_END - WALK_START);
}

export function birdX(c: number): number {
  return WALK_START - 1 + smooth(c, 2.2, 9.0) * (BIRD_END - WALK_START + 1);
}

export function crumbDropped(c: number, x: number): boolean {
  return joaoX(c) >= x;
}

export function crumbEaten(c: number, x: number): boolean {
  return birdX(c) >= x;
}

export function crumbShowing(c: number, x: number): boolean {
  return crumbDropped(c, x) && !crumbEaten(c, x);
}

export const OVEN_LOOP = 8;
export const DOOR_SHUT = 4.1;

export function witchLean(c: number): number {
  return smooth(c, 0.5, 1.5);
}

export function mariaRun(c: number): number {
  return smooth(c, 1.6, 3.0);
}

export function push(c: number): number {
  return smooth(c, 3.0, 3.8);
}

export function doorClose(c: number): number {
  return smooth(c, 3.85, 4.4) * (1 - smooth(c, 7.2, 7.9));
}

export function witchShowing(c: number): boolean {
  return c < DOOR_SHUT;
}
