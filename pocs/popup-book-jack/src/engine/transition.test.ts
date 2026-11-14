import { describe, expect, test } from "bun:test";
import { ARRIVE_TIME, DEPART_TIME, arriveOpenness, bank, departOpenness, leafAt } from "./transition";

const steps = (length: number) => Array.from({ length: Math.ceil(length / 0.01) + 1 }, (_, i) => i * 0.01);

describe("one book that travels: fold, fly while the page turns, land, unfold", () => {
  test("the pop-up is fully flat when the book takes off, so nothing stands while the page turns", () => {
    expect(departOpenness(0)).toBe(1);
    expect(departOpenness(DEPART_TIME)).toBe(0);
  });

  test("the page lies flat at takeoff and has fully turned before the landing, so the book lands on the new spread", () => {
    expect(leafAt(0)).toBe(0);
    expect(leafAt(0.95)).toBe(1);
    expect(leafAt(1)).toBe(1);
  });

  test("the page only ever moves forward during a flight", () => {
    let last = 0;
    for (const p of steps(1)) {
      expect(leafAt(p)).toBeGreaterThanOrEqual(last);
      last = leafAt(p);
    }
  });

  test("the new pop-up starts flat on landing and ends fully standing", () => {
    expect(arriveOpenness(0)).toBe(0);
    expect(arriveOpenness(ARRIVE_TIME)).toBe(1);
  });

  test("the book banks into the turn and levels out to land flat on its spot", () => {
    expect(bank(0, 1)).toEqual({ roll: 0, pitch: -0, lift: 0 });
    expect(Math.abs(bank(1, 1).roll)).toBeLessThan(1e-9);
    expect(bank(0.5, 1).roll).toBeGreaterThan(0);
    expect(bank(0.5, -1).roll).toBeLessThan(0);
  });
});
