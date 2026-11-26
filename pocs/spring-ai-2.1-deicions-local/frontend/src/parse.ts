import { ApiError } from "./http";

export type Fields = Readonly<Record<string, unknown>>;

function fail(what: string): never {
  throw new ApiError(422, `invalid payload: ${what}`);
}

export function object(raw: unknown, what: string): Fields {
  if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) return Object.fromEntries(Object.entries(raw));
  return fail(`${what} is not an object`);
}

export function num(o: Fields, key: string): number {
  const v = o[key];
  return typeof v === "number" && Number.isFinite(v) ? v : fail(`${key} is not a number`);
}

export function str(o: Fields, key: string): string {
  const v = o[key];
  return typeof v === "string" ? v : fail(`${key} is not a string`);
}

export function bool(o: Fields, key: string): boolean {
  const v = o[key];
  return typeof v === "boolean" ? v : fail(`${key} is not a boolean`);
}

export function list<T>(o: Fields, key: string, item: (raw: unknown, what: string) => T): readonly T[] {
  const v = o[key];
  return Array.isArray(v) ? v.map((x, i) => item(x, `${key}[${i}]`)) : fail(`${key} is not a list`);
}

export function numbers(o: Fields, key: string): Readonly<Record<string, number>> {
  const entries = Object.entries(object(o[key], key));
  return Object.fromEntries(entries.map(([k, v]) => [k, typeof v === "number" ? v : fail(`${key}.${k} is not a number`)]));
}

export function oneOf<const T extends readonly string[]>(o: Fields, key: string, allowed: T): T[number] {
  const v = str(o, key);
  const match = allowed.find((a) => a === v);
  return match ?? fail(`${key} has unknown value ${v}`);
}

export const asNumber = (raw: unknown, what: string): number =>
  typeof raw === "number" && Number.isFinite(raw) ? raw : fail(`${what} is not a number`);

export const asString = (raw: unknown, what: string): string =>
  typeof raw === "string" ? raw : fail(`${what} is not a string`);
