import { describe, expect, test } from "bun:test";
import { CRUMBS, CRUMB_LOOP, OVEN_LOOP, birdX, crumbDropped, crumbEaten, doorClose, joaoX, mariaRun, push, witchShowing } from "./timelines";

const frames = (length: number) => Array.from({ length: Math.ceil(length / 0.02) }, (_, i) => i * 0.02);

describe("the bread crumb trail", () => {
  test("a bird never eats a crumb João has not dropped yet", () => {
    for (const c of frames(CRUMB_LOOP)) for (const x of CRUMBS) if (crumbEaten(c, x)) expect(crumbDropped(c, x)).toBe(true);
  });

  test("the birds eat every crumb before the loop ends, so the way home is lost", () => {
    for (const x of CRUMBS) expect(crumbEaten(CRUMB_LOOP - 0.01, x)).toBe(true);
  });

  test("the birds follow behind João and never pass him", () => {
    for (const c of frames(CRUMB_LOOP)) expect(birdX(c)).toBeLessThan(joaoX(c));
  });

  test("the trail is visible for a while before the birds reach it, so the reader sees the crumbs", () => {
    for (const x of CRUMBS) {
      const dropped = frames(CRUMB_LOOP).find((c) => crumbDropped(c, x)) ?? 0;
      const eaten = frames(CRUMB_LOOP).find((c) => crumbEaten(c, x)) ?? 0;
      expect(eaten - dropped).toBeGreaterThan(1);
    }
  });
});

describe("the oven", () => {
  test("Maria reaches the witch before she pushes her", () => {
    for (const c of frames(OVEN_LOOP)) if (push(c) > 0) expect(mariaRun(c)).toBe(1);
  });

  test("the door only starts closing once the witch is all the way inside", () => {
    for (const c of frames(OVEN_LOOP)) if (doorClose(c) > 0 && c < OVEN_LOOP / 2 + 1) expect(push(c)).toBe(1);
  });

  test("the witch is never seen while the door is shut nor after it opens again", () => {
    for (const c of frames(OVEN_LOOP)) if (doorClose(c) > 0.5 || c > OVEN_LOOP / 2 + 1) expect(witchShowing(c)).toBe(false);
  });

  test("the door is open when the loop starts over, so the scene can play again", () => {
    expect(doorClose(0)).toBe(0);
    expect(doorClose(OVEN_LOOP - 0.01)).toBeLessThan(0.01);
  });
});
