import { describe, expect, test } from "bun:test";
import {
  EARS_LOOP,
  FIELD_LOOP,
  FOX_START,
  HOLE_X,
  MOUTH_X,
  NOSE_LONG,
  NOSE_LOOP,
  NOSE_SHORT,
  SHARK_LOOP,
  SNEEZE_AT,
  SNEEZE_LOOP,
  birdsPerched,
  buried,
  coinsStolen,
  dreamTree,
  earGrowth,
  foxX,
  heeHaw,
  jawOpen,
  noseLength,
  sneezeSwell,
  sneezing,
  swallowed,
  swimX,
} from "./timelines";

const frames = (length: number) => Array.from({ length: Math.ceil(length / 0.02) }, (_, i) => i * 0.02);

describe("the growing nose", () => {
  test("the birds only perch once the lie has made the nose long enough to hold them", () => {
    for (const c of frames(NOSE_LOOP)) if (birdsPerched(c)) expect(noseLength(c)).toBeGreaterThan(NOSE_LONG * 0.95);
  });

  test("the nose is short again when the loop restarts, so there is no jump", () => {
    expect(noseLength(0)).toBeCloseTo(NOSE_SHORT, 5);
    expect(noseLength(NOSE_LOOP - 0.01)).toBeCloseTo(NOSE_SHORT, 5);
  });

  test("the nose never shrinks below its normal size", () => {
    for (const c of frames(NOSE_LOOP)) expect(noseLength(c)).toBeGreaterThanOrEqual(NOSE_SHORT - 1e-9);
  });
});

describe("the donkey ears", () => {
  test("HEE-HAW only shows once the ears have fully grown", () => {
    for (const c of frames(EARS_LOOP)) if (heeHaw(c)) expect(earGrowth(c)).toBeGreaterThan(0.99);
  });

  test("the ears are gone when the loop restarts", () => {
    expect(earGrowth(0)).toBe(0);
    expect(earGrowth(EARS_LOOP - 0.01)).toBe(0);
  });
});

describe("the shark", () => {
  test("Pinocchio only vanishes once he has swum all the way into the mouth", () => {
    for (const c of frames(SHARK_LOOP)) if (swallowed(c)) expect(swimX(c)).toBeCloseTo(MOUTH_X, 5);
  });

  test("the jaws are wide open while he swims in, so he is not eaten through closed teeth", () => {
    for (const c of frames(SHARK_LOOP)) if (swimX(c) > MOUTH_X - 1 && !swallowed(c)) expect(jawOpen(c)).toBeGreaterThan(0.4);
  });

  test("the jaws snap shut after the swallow and Pinocchio stays inside until the loop restarts", () => {
    expect(jawOpen(5.2)).toBe(0);
    for (const c of frames(SHARK_LOOP)) if (c > 5) expect(swallowed(c)).toBe(true);
    expect(swallowed(0)).toBe(false);
  });
});

describe("the field of miracles", () => {
  test("the coins are buried before Pinocchio dreams of the golden tree", () => {
    for (const c of frames(FIELD_LOOP)) if (dreamTree(c) > 0) expect(buried(c)).toBe(true);
  });

  test("the dream fades before the Fox sneaks up, so the theft wakes him from it", () => {
    for (const c of frames(FIELD_LOOP)) if (foxX(c) < FOX_START - 0.01) expect(dreamTree(c)).toBe(0);
  });

  test("the coins are only stolen once the Fox has reached the hole", () => {
    for (const c of frames(FIELD_LOOP)) if (coinsStolen(c) && c < 9.8) expect(foxX(c)).toBeCloseTo(HOLE_X + 0.5, 3);
  });

  test("the Fox runs off the stage with the coins at the end", () => {
    expect(foxX(FIELD_LOOP - 0.01)).toBeGreaterThan(FOX_START);
  });
});

describe("the Fire-Eater's sneeze", () => {
  test("he swells up before the sneeze and relaxes after it", () => {
    expect(sneezeSwell(SNEEZE_AT - 0.01)).toBeGreaterThan(0.95);
    expect(sneezeSwell(SNEEZE_AT + 0.3)).toBe(0);
    expect(sneezeSwell(0)).toBe(0);
  });

  test("the sneeze happens once per loop", () => {
    const starts = frames(SNEEZE_LOOP).filter((c, i, all) => sneezing(c) && !sneezing(all[i - 1] ?? -1));
    expect(starts.length).toBe(1);
  });
});
