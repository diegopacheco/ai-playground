export function nextSeed(seed: number): number {
  return (seed + 0x6d2b79f5) >>> 0;
}

export function random(seed: number): number {
  let value = nextSeed(seed);
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}
