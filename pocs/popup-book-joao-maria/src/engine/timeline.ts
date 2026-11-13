export const FOLD_TIME = 0.7;
export const TURN_TIME = 1.25;
export const UNFOLD_TIME = 1.6;
export const TOTAL_TURN_TIME = FOLD_TIME + TURN_TIME + UNFOLD_TIME;

export interface TurnFrame {
  oldOpenness: number;
  leaf: number;
  newOpenness: number;
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

export function turnFrame(elapsed: number): TurnFrame {
  const fold = clamp01(elapsed / FOLD_TIME);
  const turn = clamp01((elapsed - FOLD_TIME) / TURN_TIME);
  const unfold = clamp01((elapsed - FOLD_TIME - TURN_TIME) / UNFOLD_TIME);
  return {
    oldOpenness: 1 - easeInOutCubic(fold),
    leaf: easeInOutCubic(turn),
    newOpenness: unfold,
    done: elapsed >= TOTAL_TURN_TIME,
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
