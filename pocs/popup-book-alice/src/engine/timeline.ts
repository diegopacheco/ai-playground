export const SCENE_SPACING = 16;
export const SET_DEPTH = 7;
export const VIEW_DISTANCE = 9.2;
export const TRAVEL_TIME = 2.6;
export const OPEN_START = 1.1;
export const OPEN_TIME = 1.5;

export interface TravelFrame {
  progress: number;
  nextOpenness: number;
  arrived: boolean;
}

export function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function easeOutBack(t: number): number {
  const c1 = 1.6;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

export function smooth(t: number, a: number, b: number): number {
  const x = clamp01((t - a) / (b - a));
  return x * x * (3 - 2 * x);
}

export function loop(t: number, length: number): number {
  return ((t % length) + length) % length;
}

export function sceneZ(index: number): number {
  return -index * SCENE_SPACING;
}

export function stationZ(index: number): number {
  return sceneZ(index) + VIEW_DISTANCE;
}

export function travelFrame(elapsed: number): TravelFrame {
  return {
    progress: easeInOutCubic(clamp01(elapsed / TRAVEL_TIME)),
    nextOpenness: clamp01((elapsed - OPEN_START) / OPEN_TIME),
    arrived: elapsed >= Math.max(TRAVEL_TIME, OPEN_START + OPEN_TIME),
  };
}

export function layerFade(distanceAhead: number): number {
  return smooth(distanceAhead, 0.25, 1.6);
}

export function accordionZ(baseZ: number, openness: number): number {
  const spread = easeOutBack(clamp01(openness));
  return -SET_DEPTH + (baseZ + SET_DEPTH) * spread;
}

export function fitDistance(aspect: number, halfWidth: number, halfHeight: number, fovDeg: number): number {
  const t = Math.tan((fovDeg * Math.PI) / 360);
  return Math.max(halfWidth / (t * aspect), halfHeight / t);
}
