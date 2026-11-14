import { describe, expect, test } from "bun:test";
import { catBody, catGrin, CAT_LOOP, growAlice, HOLE_X, MOVE_EVERY, rabbitDive, rabbitX, roseRed, ROSE_LOOP, seatShift, shrinkRoom, shrinkSet, SIZE_LOOP } from "./timelines";

const sample = (length: number, step = 0.05) => Array.from({ length: Math.floor(length / step) }, (_, i) => i * step);

describe("Wonderland rules", () => {
  test("the rabbit only dives once he reaches the hole", () => {
    for (const t of sample(8)) if (rabbitDive(t) > 0) expect(rabbitX(t)).toBeCloseTo(HOLE_X, 1);
  });

  test("drinking makes the world bigger around Alice, which means she shrinks, and she always comes back", () => {
    expect(Math.max(...sample(SIZE_LOOP).map(shrinkSet))).toBeGreaterThan(2);
    for (const t of sample(SIZE_LOOP)) expect(shrinkSet(t)).toBeGreaterThanOrEqual(1);
    expect(shrinkSet(0)).toBe(1);
  });

  test("eating makes Alice tall enough to hit the roof while the room shrinks", () => {
    const peak = sample(SIZE_LOOP).reduce((best, t) => (growAlice(t) > growAlice(best) ? t : best), 0);
    expect(growAlice(peak) * 2.6).toBeGreaterThan(6);
    expect(shrinkRoom(peak)).toBeLessThan(1);
  });

  test("the Cheshire Cat vanishes before its grin, so the grin is ever left alone", () => {
    let grinAlone = false;
    for (const t of sample(CAT_LOOP)) {
      if (catBody(t) < 0.05) expect(catGrin(t) >= 0).toBe(true);
      if (catBody(t) < 0.05 && catGrin(t) > 0.95) grinAlone = true;
      if (catGrin(t) < 0.5) expect(catBody(t)).toBeLessThan(0.05);
    }
    expect(grinAlone).toBe(true);
  });

  test("at the tea party everyone moves exactly one place on at each move", () => {
    expect(seatShift(MOVE_EVERY - 0.9)).toBe(0);
    expect(seatShift(MOVE_EVERY)).toBeCloseTo(1);
    expect(seatShift(MOVE_EVERY * 2)).toBeCloseTo(2);
  });

  test("the gardeners paint the roses one after another, never all at once", () => {
    const t = 3;
    const reds = Array.from({ length: 7 }, (_, i) => roseRed(t, i, 7));
    expect(reds[0]).toBeGreaterThan(reds[6]!);
    expect(roseRed(ROSE_LOOP - 4, 6, 7)).toBe(1);
  });
});
