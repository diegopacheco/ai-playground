import { describe, expect, test } from "bun:test";
import { changeFrame, DARK_TIME, DIM_TIME, lampLevel, TOTAL_CHANGE_TIME } from "./timeline";

const step = 0.01;
const frames = Array.from({ length: Math.ceil(TOTAL_CHANGE_TIME / step) + 1 }, (_, i) => ({ t: i * step, f: changeFrame(i * step) }));

describe("scene change by lamp", () => {
  test("the scenery is only swapped while the lamp is out, so the audience never sees the stagehands", () => {
    const swap = frames.find(({ f }) => f.swapped);
    expect(swap).toBeDefined();
    expect(swap!.f.level).toBe(0);
  });

  test("the old shadows fold flat before the swap so nothing pops out of existence", () => {
    const before = frames.filter(({ f }) => !f.swapped).at(-1)!;
    expect(before.f.oldOpenness).toBeLessThan(0.01);
  });

  test("the new scene does not rise before the swap", () => {
    for (const { f } of frames) if (!f.swapped) expect(f.newOpenness).toBe(0);
  });

  test("the lamp is relit with a flare that settles back to full light", () => {
    const relit = frames.filter(({ t }) => t > DIM_TIME + DARK_TIME);
    expect(Math.max(...relit.map(({ f }) => f.level))).toBeGreaterThan(1);
    expect(lampLevel(TOTAL_CHANGE_TIME + 5)).toBeCloseTo(1, 2);
  });

  test("the lamp only dims while it gutters out, it never brightens before going dark", () => {
    expect(lampLevel(0)).toBe(1);
    expect(lampLevel(DIM_TIME * 0.9)).toBeLessThan(lampLevel(DIM_TIME * 0.2));
    expect(changeFrame(TOTAL_CHANGE_TIME).done).toBe(true);
  });
});
