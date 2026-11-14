export const CLOSE_TIME = 1.1;
export const HOLD_TIME = 0.45;
export const OPEN_TIME = 1.6;
export const RISE_TIME = 1.7;
export const TOTAL_CHANGE_TIME = CLOSE_TIME + HOLD_TIME + OPEN_TIME + 0.5;

export interface ChangeFrame {
  curtain: number;
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
  const c1 = 1.9;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

export function curtainAt(elapsed: number): number {
  if (elapsed < CLOSE_TIME) return easeInOutCubic(clamp01(elapsed / CLOSE_TIME));
  if (elapsed < CLOSE_TIME + HOLD_TIME) return 1;
  return 1 - easeInOutCubic(clamp01((elapsed - CLOSE_TIME - HOLD_TIME) / OPEN_TIME));
}

export function changeFrame(elapsed: number): ChangeFrame {
  const opening = elapsed - CLOSE_TIME - HOLD_TIME - OPEN_TIME * 0.2;
  return {
    curtain: curtainAt(elapsed),
    oldOpenness: 1 - easeInOutCubic(clamp01(elapsed / (CLOSE_TIME * 0.85))),
    newOpenness: clamp01(opening / RISE_TIME),
    swapped: elapsed >= CLOSE_TIME,
    done: elapsed >= TOTAL_CHANGE_TIME,
  };
}

export function pieceProgress(openness: number, delay: number): number {
  const span = 1 - delay;
  return span <= 0 ? (openness >= 1 ? 1 : 0) : clamp01((openness - delay) / span);
}

export function foldAngle(openness: number, delay: number): number {
  const p = pieceProgress(openness, delay);
  return (1 - easeOutBack(p)) * (Math.PI / 2);
}

export function smooth(t: number, a: number, b: number): number {
  const x = clamp01((t - a) / (b - a));
  return x * x * (3 - 2 * x);
}

export function loop(t: number, length: number): number {
  return ((t % length) + length) % length;
}
