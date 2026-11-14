import { describe, expect, test } from "bun:test";
import { CLOSE_TIME, HOLD_TIME, OPEN_TIME, TOTAL_CHANGE_TIME, changeFrame, curtainAt, foldAngle, pieceProgress } from "./timeline";

const frames = Array.from({ length: Math.ceil(TOTAL_CHANGE_TIME / 0.02) + 1 }, (_, i) => i * 0.02);

describe("scene change behind the curtains", () => {
  test("the curtains start open and end open, so every scene is seen whole", () => {
    expect(curtainAt(0)).toBe(0);
    expect(curtainAt(TOTAL_CHANGE_TIME)).toBeCloseTo(0, 9);
  });

  test("the scenery is only swapped while the curtains are fully shut, so the audience never sees the switch", () => {
    for (const t of frames) {
      const frame = changeFrame(t);
      if (frame.swapped && t < CLOSE_TIME + HOLD_TIME) expect(frame.curtain).toBe(1);
    }
    expect(changeFrame(CLOSE_TIME - 0.01).swapped).toBe(false);
    expect(changeFrame(CLOSE_TIME).curtain).toBe(1);
  });

  test("the old pop-ups have folded flat before the curtains meet", () => {
    expect(changeFrame(CLOSE_TIME * 0.85).oldOpenness).toBeCloseTo(0, 9);
    expect(changeFrame(CLOSE_TIME * 0.85).curtain).toBeLessThan(1);
  });

  test("the new scene stays flat until the curtains have started to part, then rises fully", () => {
    for (const t of frames) if (t <= CLOSE_TIME + HOLD_TIME) expect(changeFrame(t).newOpenness).toBe(0);
    expect(changeFrame(CLOSE_TIME + HOLD_TIME + OPEN_TIME / 2).newOpenness).toBeGreaterThan(0);
    expect(changeFrame(TOTAL_CHANGE_TIME).newOpenness).toBe(1);
  });

  test("the curtains never flutter: they only close, wait, then only open", () => {
    let last = 0;
    for (const t of frames.filter((f) => f <= CLOSE_TIME + HOLD_TIME)) {
      expect(curtainAt(t)).toBeGreaterThanOrEqual(last);
      last = curtainAt(t);
    }
    for (const t of frames.filter((f) => f > CLOSE_TIME + HOLD_TIME)) {
      expect(curtainAt(t)).toBeLessThanOrEqual(last);
      last = curtainAt(t);
    }
  });
});

describe("pop-up pieces", () => {
  test("a delayed piece waits for its turn, which makes the set rise in layers", () => {
    expect(pieceProgress(0.3, 0.4)).toBe(0);
    expect(pieceProgress(0.7, 0.4)).toBeGreaterThan(0);
  });

  test("a closed scene lays every flat down and an open one stands it upright", () => {
    expect(foldAngle(0, 0.2)).toBeCloseTo(Math.PI / 2, 5);
    expect(foldAngle(1, 0.2)).toBeCloseTo(0, 5);
  });

  test("flats overshoot upright while rising, giving the cardboard a springy bounce", () => {
    const angles = Array.from({ length: 50 }, (_, i) => foldAngle(i / 49, 0));
    expect(Math.min(...angles)).toBeLessThan(0);
  });
});
