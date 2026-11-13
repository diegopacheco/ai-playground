import { describe, expect, test } from "bun:test";
import { BLOW_LOOP, REBUILD_START, collapse, escape, pigShowing } from "./blowTimeline";

const PIGS = [0, 1];
const frames = Array.from({ length: Math.ceil(BLOW_LOOP / 0.05) }, (_, i) => i * 0.05);

describe("pigs hiding in a house the wolf blows down", () => {
  test("the pigs stay hidden inside while the walls still stand, so they only run once the house falls", () => {
    for (const c of frames) {
      if (c < REBUILD_START && collapse(c) < 0.5) for (const pig of PIGS) expect(escape(c, pig)).toBe(0);
    }
  });

  test("every pig is out of the house before it rebuilds for the next loop", () => {
    for (const pig of PIGS) expect(escape(REBUILD_START, pig)).toBeGreaterThan(0.98);
  });

  test("the pigs never slide back into the house where the reader can see it", () => {
    for (const c of frames.filter((c) => c >= REBUILD_START)) {
      for (const pig of PIGS) expect(pigShowing(c, pig)).toBe(false);
    }
  });

  test("the pigs burst out one after another instead of as a single block", () => {
    for (const c of frames) expect(escape(c, 0)).toBeGreaterThanOrEqual(escape(c, 1));
    expect(escape(3.6, 0)).toBeGreaterThan(escape(3.6, 1));
  });

  test("the house is standing again when the loop starts over", () => {
    expect(collapse(0)).toBe(0);
    expect(collapse(BLOW_LOOP - 0.01)).toBeLessThan(0.01);
  });
});
