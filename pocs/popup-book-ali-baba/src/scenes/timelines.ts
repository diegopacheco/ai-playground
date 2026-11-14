import { clamp01, loop, smooth } from "../engine/timeline";

export const STREET_LOOP = 16;
export const DOORS = [-3, -1.5, 0, 1.5, 3];
export const ALI_DOOR = 2;
export const THIEF_MARK_AT = 3;
export const MORGIANA_ENTER = 4.5;
export const MORGIANA_SPEED = 0.75;

export function thiefX(time: number): number {
  const t = loop(time, STREET_LOOP);
  if (t < THIEF_MARK_AT) return -5 + (5 + (DOORS[ALI_DOOR] ?? 0) - 0.4) * smooth(t, 0, THIEF_MARK_AT);
  return (DOORS[ALI_DOOR] ?? 0) - 0.4 + Math.max(0, t - THIEF_MARK_AT - 0.8) * 1.6;
}

export function morgianaX(time: number): number {
  const t = loop(time, STREET_LOOP);
  return -4.6 + Math.max(0, t - MORGIANA_ENTER) * MORGIANA_SPEED;
}

export function doorMarks(time: number): boolean[] {
  const t = loop(time, STREET_LOOP);
  const her = morgianaX(time);
  return DOORS.map((x, i) => (i === ALI_DOOR && t >= THIEF_MARK_AT) || (t > MORGIANA_ENTER && her >= x));
}

export const RIDERS = 8;
export const RIDE_LOOP = 9;

export function riderX(time: number, index: number): number {
  const t = loop(time + index * 0.9, RIDE_LOOP);
  return 7 - (t / RIDE_LOOP) * 14;
}

export const SESAME_LOOP = 12;

export function shoutOn(time: number): boolean {
  const t = loop(time, SESAME_LOOP);
  return t > 1.5 && t < 5;
}

export function doorOpen(time: number): number {
  const t = loop(time, SESAME_LOOP);
  return smooth(t, 3, 5.5) * (1 - smooth(t, 10.5, 11.8));
}

export const WRONG_WORDS = ["Open, Barley!", "Open, Wheat!", "Open, Rice!", "Open... Bean?"];

export function wrongWord(time: number): number {
  return Math.floor(loop(time, WRONG_WORDS.length * 2.2) / 2.2);
}

export const JAR_X = [-2.6, -1.3, 0, 1.3, 2.6];
export const JAR_LOOP = 15;

export function lanternX(time: number): number {
  const t = loop(time, JAR_LOOP);
  return -4.4 + 8.8 * smooth(t, 1, 13);
}

export function lidLift(time: number, index: number): number {
  const t = loop(time, JAR_LOOP);
  const x = JAR_X[index] ?? 0;
  const sealed = lanternX(time) > x + 0.5;
  if (sealed) return 0;
  const cycle = loop(t * 1.4 + index * 1.7, 4);
  return clamp01(Math.sin(cycle * Math.PI * 0.5) * 1.2) * (cycle < 2 ? 1 : 0);
}

export function danceOrbit(time: number): { x: number; z: number } {
  return { x: Math.sin(time * 0.55) * 1.6, z: Math.cos(time * 0.55) * 0.9 - 0.3 };
}

export const DAGGER_LOOP = 10;

export function daggerShown(time: number): boolean {
  const t = loop(time, DAGGER_LOOP);
  return t > 7 && t < 9.5;
}
