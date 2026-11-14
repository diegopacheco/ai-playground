export interface Vec {
  x: number;
  y: number;
  z: number;
}

export interface Shot {
  pos: Vec;
  look: Vec;
}

export const MIN_FLIGHT = 1.6;
export const MAX_FLIGHT = 5.0;

export function distance(a: Vec, b: Vec): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

export function flightTime(from: Shot, to: Shot): number {
  return Math.min(Math.max(distance(from.pos, to.pos) / 7, MIN_FLIGHT), MAX_FLIGHT);
}

export function ease(t: number): number {
  const x = Math.min(Math.max(t, 0), 1);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

function mix(a: Vec, b: Vec, t: number): Vec {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t };
}

export function arc(from: Shot, to: Shot): number {
  return Math.min(distance(from.pos, to.pos) * 0.12, 5);
}

export function flightShot(from: Shot, to: Shot, progress: number): Shot {
  const t = ease(progress);
  const pos = mix(from.pos, to.pos, t);
  const swing = Math.sin(Math.PI * t) * arc(from, to);
  pos.y += swing;
  pos.z += swing * 1.6;
  return { pos, look: mix(from.look, to.look, t) };
}
