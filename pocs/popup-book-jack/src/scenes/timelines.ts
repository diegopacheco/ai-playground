export function smooth(t: number, a: number, b: number): number {
  const x = Math.min(Math.max((t - a) / (b - a), 0), 1);
  return x * x * (3 - 2 * x);
}

export const GROW_TIME = 7;
export const STALK_HEIGHT = 34;
export const LEAF_SPOTS = [0.1, 0.21, 0.32, 0.43, 0.54, 0.65, 0.76, 0.86];

export function growth(since: number): number {
  return smooth(since, 1.0, GROW_TIME);
}

export function leafOpen(amount: number, spot: number): number {
  const reached = amount - spot;
  return reached <= 0 ? 0 : Math.min(reached / 0.12, 1);
}

export const EGG_LOOP = 10;
export const LAY_TIMES = [1.5, 3.8, 6.1];
export const EGG_FALL = 0.5;

export function eggsLaid(c: number): number {
  return LAY_TIMES.filter((time) => c >= time).length;
}

export function eggDrop(c: number, egg: number): number {
  const time = LAY_TIMES[egg];
  if (time === undefined || c < time) return 0;
  return Math.min((c - time) / EGG_FALL, 1);
}

export function henSquat(c: number): number {
  return LAY_TIMES.reduce((most, time) => Math.max(most, smooth(c, time - 0.5, time - 0.1) * (1 - smooth(c, time, time + 0.4))), 0);
}

export const CHASE_LOOP = 8;
export const TOP = 3.2;
export const BOTTOM = 0.2;
export const GAP = 0.9;

export function jackHeight(c: number): number {
  return TOP - smooth(c, 0.3, 6.0) * (TOP - BOTTOM);
}

export function giantHeight(c: number): number {
  return TOP + 0.6 - smooth(c, 1.2, 7.4) * (TOP + 0.6 - BOTTOM - GAP);
}

export const CHOP_TIMES = [2.2, 3.0, 3.8];
export const FALL_START = 4.2;
export const FALL_TIME = 2.2;

export function chopsDone(c: number): number {
  return CHOP_TIMES.filter((time) => c >= time).length;
}

export function stalkFall(since: number): number {
  const t = Math.min(Math.max((since - FALL_START) / FALL_TIME, 0), 1);
  return t * t;
}

export function crashed(since: number): boolean {
  return stalkFall(since) >= 1;
}

export function axeSwing(c: number): number {
  return CHOP_TIMES.reduce((most, time) => Math.max(most, smooth(c, time - 0.35, time) * (1 - smooth(c, time, time + 0.25))), 0);
}
