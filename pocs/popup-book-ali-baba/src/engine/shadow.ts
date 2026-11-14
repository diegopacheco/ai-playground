export const SCREEN_W = 8;
export const SCREEN_H = 4.8;
export const SCREEN_LEFT = -SCREEN_W / 2;
export const SCREEN_BOTTOM = 0;

export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export interface Frustum {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export function shadowScale(lampDistance: number, depth: number): number {
  return lampDistance / Math.max(lampDistance - depth, 0.001);
}

export function projectToScreen(point: Point3, lamp: Point3): { x: number; y: number } {
  const t = -lamp.z / Math.max(point.z - lamp.z, 0.0001);
  return { x: lamp.x + (point.x - lamp.x) * t, y: lamp.y + (point.y - lamp.y) * t };
}

export function penumbra(lampDistance: number, depth: number, lampSize: number): number {
  return (lampSize * depth) / Math.max(lampDistance - depth, 0.001);
}

export function lampFrustum(lamp: Point3, near: number): Frustum {
  const d = Math.max(-lamp.z, 0.001);
  const k = near / d;
  const x0 = SCREEN_LEFT - lamp.x;
  const x1 = SCREEN_LEFT + SCREEN_W - lamp.x;
  return {
    left: -x1 * k,
    right: -x0 * k,
    top: (SCREEN_BOTTOM + SCREEN_H - lamp.y) * k,
    bottom: (SCREEN_BOTTOM - lamp.y) * k,
  };
}
