import { describe, expect, test } from "bun:test";
import { foldAngle, pieceProgress } from "./timeline";

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
