export function call(fn: (a: number) => number): number {
  return Reflect.apply(fn, undefined, [1]);
}
