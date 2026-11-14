import { describe, expect, test } from "bun:test";
import {
  BOTTOM,
  CHASE_LOOP,
  CHOP_TIMES,
  FALL_START,
  FALL_TIME,
  EGG_LOOP,
  GROW_TIME,
  LAY_TIMES,
  LEAF_SPOTS,
  axeSwing,
  chopsDone,
  crashed,
  eggDrop,
  eggsLaid,
  giantHeight,
  growth,
  henSquat,
  jackHeight,
  leafOpen,
  stalkFall,
} from "./timelines";

const frames = (length: number) => Array.from({ length: Math.ceil(length / 0.02) }, (_, i) => i * 0.02);

describe("the beanstalk grows", () => {
  test("a leaf never unfolds before the stalk has grown past it", () => {
    for (const c of frames(GROW_TIME + 2)) for (const spot of LEAF_SPOTS) if (leafOpen(growth(c), spot) > 0) expect(growth(c)).toBeGreaterThan(spot);
  });

  test("the stalk only ever grows taller, it never shrinks back", () => {
    let last = 0;
    for (const c of frames(GROW_TIME + 2)) {
      expect(growth(c)).toBeGreaterThanOrEqual(last);
      last = growth(c);
    }
  });

  test("the camera arrives on a bare bean, so the reader sees the stalk burst out of the ground", () => {
    expect(growth(0)).toBe(0);
  });

  test("the stalk reaches the sky with every leaf open and stays there", () => {
    expect(growth(GROW_TIME)).toBe(1);
    expect(growth(GROW_TIME + 60)).toBe(1);
    for (const spot of LEAF_SPOTS) expect(leafOpen(1, spot)).toBe(1);
  });
});

describe("the golden hen", () => {
  test("an egg only shows up after the hen lays it", () => {
    for (const c of frames(EGG_LOOP)) for (let egg = 0; egg < LAY_TIMES.length; egg++) if (eggDrop(c, egg) > 0) expect(eggsLaid(c)).toBeGreaterThan(egg);
  });

  test("the hen squats right before each egg drops, so the reader sees where it came from", () => {
    for (const time of LAY_TIMES) expect(henSquat(time - 0.1)).toBeGreaterThan(0.9);
  });

  test("eggs pile up one by one and never disappear in the middle of the loop", () => {
    let last = 0;
    for (const c of frames(EGG_LOOP)) {
      expect(eggsLaid(c)).toBeGreaterThanOrEqual(last);
      last = eggsLaid(c);
    }
    expect(eggsLaid(EGG_LOOP - 0.01)).toBe(LAY_TIMES.length);
  });

  test("the pile starts empty, so the loop plays the laying again", () => {
    expect(eggsLaid(0)).toBe(0);
  });
});

describe("the chase", () => {
  test("the giant is always above Jack, so Jack is never caught", () => {
    for (const c of frames(CHASE_LOOP)) expect(giantHeight(c)).toBeGreaterThan(jackHeight(c));
  });

  test("Jack reaches the ground before the loop ends, so he has time to grab the axe", () => {
    expect(jackHeight(CHASE_LOOP - 0.5)).toBeCloseTo(BOTTOM, 6);
  });

  test("Jack climbs down, never back up", () => {
    let last = Number.POSITIVE_INFINITY;
    for (const c of frames(CHASE_LOOP)) {
      expect(jackHeight(c)).toBeLessThanOrEqual(last);
      last = jackHeight(c);
    }
  });
});

describe("chopping the beanstalk", () => {
  test("the beanstalk only falls after the last chop", () => {
    for (const c of frames(10)) if (stalkFall(c) > 0) expect(chopsDone(c)).toBe(CHOP_TIMES.length);
  });

  test("the axe hits the stalk at every chop", () => {
    for (const time of CHOP_TIMES) expect(axeSwing(time)).toBe(1);
  });

  test("the stalk speeds up as it falls, like a real tree", () => {
    const step = 0.2;
    const early = stalkFall(FALL_START + step) - stalkFall(FALL_START);
    const late = stalkFall(FALL_START + FALL_TIME) - stalkFall(FALL_START + FALL_TIME - step);
    expect(late).toBeGreaterThan(early);
  });

  test("once the stalk crashes it stays down, so the giant can never climb back", () => {
    expect(stalkFall(0)).toBe(0);
    expect(crashed(FALL_START + FALL_TIME)).toBe(true);
    for (const c of frames(30)) if (c >= FALL_START + FALL_TIME) expect(stalkFall(c)).toBe(1);
  });
});
