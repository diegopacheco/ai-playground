export const DIM_TIME = 0.9;
export const DARK_TIME = 0.35;
export const LIGHT_TIME = 1.1;
export const RISE_TIME = 1.6;
export const TOTAL_CHANGE_TIME = DIM_TIME + DARK_TIME + RISE_TIME + 0.2;
export const FLARE = 0.3;

export interface ChangeFrame {
  level: number;
  oldOpenness: number;
  newOpenness: number;
  swapped: boolean;
  done: boolean;
}

export function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function easeOutBack(t: number): number {
  const c1 = 1.7;
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

export function lampLevel(elapsed: number): number {
  if (elapsed < DIM_TIME) {
    const t = elapsed / DIM_TIME;
    const gutter = 1 - t * t;
    const sputter = t > 0.45 ? 0.18 * Math.abs(Math.sin(t * 38)) * (1 - t) : 0;
    return clamp01(gutter - sputter);
  }
  if (elapsed < DIM_TIME + DARK_TIME) return 0;
  const t = clamp01((elapsed - DIM_TIME - DARK_TIME) / LIGHT_TIME);
  const grow = 1 - Math.pow(1 - t, 3);
  const flare = FLARE * Math.sin(Math.PI * t) ** 2;
  return grow + flare;
}

export function changeFrame(elapsed: number): ChangeFrame {
  const lit = elapsed - DIM_TIME - DARK_TIME;
  return {
    level: lampLevel(elapsed),
    oldOpenness: 1 - easeInOutCubic(clamp01(elapsed / DIM_TIME)),
    newOpenness: clamp01(lit / RISE_TIME),
    swapped: elapsed >= DIM_TIME + DARK_TIME * 0.5,
    done: elapsed >= TOTAL_CHANGE_TIME,
  };
}

export function pieceProgress(openness: number, delay: number): number {
  const span = 1 - delay;
  return span <= 0 ? (openness >= 1 ? 1 : 0) : clamp01((openness - delay) / span);
}

export function foldAngle(openness: number, delay: number): number {
  return (1 - easeOutBack(pieceProgress(openness, delay))) * (Math.PI / 2);
}
