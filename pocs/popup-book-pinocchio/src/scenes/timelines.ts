import { smooth } from "../engine/timeline";

export const NOSE_LOOP = 10;
export const NOSE_SHORT = 0.22;
export const NOSE_LONG = 1.9;

export function noseLength(c: number): number {
  return NOSE_SHORT + (smooth(c, 1.4, 4.0) - smooth(c, 7.6, 9.2)) * (NOSE_LONG - NOSE_SHORT);
}

export function birdsPerched(c: number): boolean {
  return c >= 4.4 && c < 7.3;
}

export const EARS_LOOP = 8;

export function earGrowth(c: number): number {
  return smooth(c, 1.5, 3.0) * (1 - smooth(c, 6.6, 7.6));
}

export function heeHaw(c: number): boolean {
  return c >= 3.1 && c < 6.4;
}

export const SHARK_LOOP = 10;
export const SWIM_START = -3.6;
export const MOUTH_X = 0.2;

export function jawOpen(c: number): number {
  return smooth(c, 0.6, 2.0) * (1 - smooth(c, 4.6, 5.1));
}

export function swimX(c: number): number {
  return SWIM_START + smooth(c, 0.4, 4.6) * (MOUTH_X - SWIM_START);
}

export function swallowed(c: number): boolean {
  return c >= 4.85;
}

export const FIELD_LOOP = 12;
export const HOLE_X = -0.4;
export const FOX_START = 4.6;

export function buried(c: number): boolean {
  return c >= 2.2;
}

export function dreamTree(c: number): number {
  return smooth(c, 2.6, 4.2) * (1 - smooth(c, 6.0, 6.8));
}

export function foxX(c: number): number {
  const sneak = smooth(c, 7.0, 9.0);
  const flee = smooth(c, 9.8, 11.6);
  return FOX_START + sneak * (HOLE_X + 0.5 - FOX_START) + flee * (FOX_START + 1.5 - HOLE_X - 0.5);
}

export function coinsStolen(c: number): boolean {
  return c >= 9.2;
}

export const SNEEZE_LOOP = 6;
export const SNEEZE_AT = 3.2;

export function sneezeSwell(c: number): number {
  return smooth(c, 1.6, SNEEZE_AT) * (1 - smooth(c, SNEEZE_AT, SNEEZE_AT + 0.25));
}

export function sneezing(c: number): boolean {
  return c >= SNEEZE_AT && c < SNEEZE_AT + 1.2;
}
