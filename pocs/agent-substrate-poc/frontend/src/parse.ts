import { ApiError } from "./http";

export type Obj = Readonly<Record<string, unknown>>;

export function invalid(what: string): never {
  throw new ApiError(422, `invalid ${what} payload`);
}

export function obj(raw: unknown, what: string): Obj {
  if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
    return Object.fromEntries(Object.entries(raw));
  }
  return invalid(what);
}

export function str(o: Obj, key: string, what: string): string {
  const v = o[key];
  return typeof v === "string" ? v : invalid(what);
}

export function num(o: Obj, key: string, what: string): number {
  const v = o[key];
  return typeof v === "number" ? v : invalid(what);
}

export function list<T>(raw: unknown, what: string, item: (v: unknown) => T): readonly T[] {
  return Array.isArray(raw) ? raw.map(item) : invalid(what);
}

export function oneOf<T extends string>(o: Obj, key: string, values: readonly T[], what: string): T {
  const v = o[key];
  return values.find((x) => x === v) ?? invalid(what);
}
