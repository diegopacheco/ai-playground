import { ApiError } from "./http";

export type Obj = Readonly<Record<string, unknown>>;

function invalid(what: string): never {
  throw new ApiError(422, `invalid payload: ${what}`, null);
}

export function obj(raw: unknown, what: string): Obj {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) invalid(what);
  return Object.fromEntries(Object.entries(raw));
}

export function str(o: Obj, key: string): string {
  const v = o[key];
  return typeof v === "string" ? v : invalid(key);
}

export function num(o: Obj, key: string): number {
  const v = o[key];
  return typeof v === "number" ? v : invalid(key);
}

export function bool(o: Obj, key: string): boolean {
  const v = o[key];
  return typeof v === "boolean" ? v : invalid(key);
}

export function list<T>(raw: unknown, parse: (item: unknown) => T, what: string): readonly T[] {
  if (!Array.isArray(raw)) invalid(what);
  return raw.map(parse);
}

export function strings(o: Obj, key: string): readonly string[] {
  return list(o[key] ?? [], (v) => (typeof v === "string" ? v : invalid(key)), key);
}

export function date(o: Obj, key: string): Date {
  const d = new Date(str(o, key));
  return Number.isNaN(d.getTime()) ? invalid(key) : d;
}
