import type { Clock, IdGenerator } from "../src/clock.ts";
import type { Logger } from "../src/logger.ts";

export function manualClock(start = new Date("2026-11-24T10:00:00Z")): Clock & { readonly advance: (ms: number) => void } {
  let now = start.getTime();
  return { now: () => new Date(now), advance: (ms) => (now += ms) };
}

export function sequentialIds(): IdGenerator {
  let n = 0;
  return () => `task-${++n}`;
}

export const silentLogger: Logger = { info: () => {}, warn: () => {}, error: () => {} };

export function gate(): { readonly promise: Promise<void>; readonly open: () => void } {
  let open = () => {};
  const promise = new Promise<void>((resolve) => (open = resolve));
  return { promise, open };
}
