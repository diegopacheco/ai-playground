export const FOLD_TIME = 0.7;
export const UNFOLD_TIME = 1.6;

export function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

export function easeOutBack(t: number): number {
  const c1 = 1.9;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

export function pieceProgress(openness: number, delay: number): number {
  const span = 1 - delay;
  return span <= 0 ? (openness >= 1 ? 1 : 0) : clamp01((openness - delay) / span);
}

export function foldAngle(openness: number, delay: number): number {
  const p = pieceProgress(openness, delay);
  return (1 - easeOutBack(p)) * (Math.PI / 2);
}
