import { describe, expect, test } from "bun:test";
import { FOLD_TIME, TOTAL_TURN_TIME, TURN_TIME, foldAngle, pieceProgress, turnFrame } from "./timeline";

describe("page turn timeline", () => {
  test("the old pop-ups are fully flat before the leaf starts moving, so the leaf never cuts through them", () => {
    const atLeafStart = turnFrame(FOLD_TIME);
    expect(atLeafStart.oldOpenness).toBeCloseTo(0, 5);
    expect(atLeafStart.leaf).toBe(0);
  });

  test("the new pop-ups stay flat until the leaf has landed", () => {
    const midTurn = turnFrame(FOLD_TIME + TURN_TIME / 2);
    expect(midTurn.newOpenness).toBe(0);
    const landed = turnFrame(FOLD_TIME + TURN_TIME);
    expect(landed.leaf).toBeCloseTo(1, 5);
  });

  test("the leaf only ever moves forward during a turn", () => {
    let last = -1;
    for (let t = 0; t <= TOTAL_TURN_TIME; t += 0.05) {
      const { leaf } = turnFrame(t);
      expect(leaf).toBeGreaterThanOrEqual(last);
      last = leaf;
    }
  });

  test("the turn starts with the scene standing and ends with the new scene standing", () => {
    expect(turnFrame(0).oldOpenness).toBe(1);
    expect(turnFrame(0).done).toBe(false);
    const end = turnFrame(TOTAL_TURN_TIME);
    expect(end.newOpenness).toBeCloseTo(1, 9);
    expect(end.done).toBe(true);
  });
});

describe("pop-up pieces", () => {
  test("a delayed piece waits for its turn, which makes the scene rise in layers", () => {
    expect(pieceProgress(0.3, 0.4)).toBe(0);
    expect(pieceProgress(0.7, 0.4)).toBeGreaterThan(0);
  });

  test("a closed page lays every piece flat and an open page stands it upright", () => {
    expect(foldAngle(0, 0.2)).toBeCloseTo(Math.PI / 2, 5);
    expect(foldAngle(1, 0.2)).toBeCloseTo(0, 5);
  });

  test("pieces overshoot upright while rising, giving the paper a springy bounce", () => {
    const angles = Array.from({ length: 50 }, (_, i) => foldAngle(i / 49, 0));
    expect(Math.min(...angles)).toBeLessThan(0);
  });
});
