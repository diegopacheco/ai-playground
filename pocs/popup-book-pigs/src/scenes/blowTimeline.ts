import { smooth } from "./kit";

export const BLOW_LOOP = 7.5;
export const REBUILD_START = 6.8;

export function collapse(c: number): number {
  return smooth(c, 2.2, 3.4) * (1 - smooth(c, REBUILD_START, 7.4));
}

export function escape(c: number, pig: number): number {
  return smooth(c, 3.0 + pig * 0.3, 4.8) * (1 - smooth(c, REBUILD_START, 7.4));
}

export function pigShowing(c: number, pig: number): boolean {
  return c < REBUILD_START && escape(c, pig) < 0.98;
}
