import { describe, expect, test } from "bun:test";
import { getCharacterSpec } from "../src/features/characters/catalog";
import { evaluateUnlocks, getUnlockProgress } from "../src/features/characters/unlocks";

describe("character unlock progression", () => {
  test("starts with starter friends unlocked and tracks progress per rule", () => {
    const empty = { cookedCount: 0, paintedCount: 0, movedCount: 0, calledCount: 0, fedCount: 0 };
    const result = evaluateUnlocks(empty, []);
    expect(result.unlocked).toEqual(["momo", "pip"]);
    expect(getUnlockProgress(getCharacterSpec("bubu"), empty)).toEqual({
      current: 0,
      target: 2,
      done: false,
    });
  });

  test("unlocks each character when its activity threshold is reached", () => {
    const afterCook = evaluateUnlocks(
      { cookedCount: 2, paintedCount: 0, movedCount: 0, calledCount: 0, fedCount: 0 },
      ["momo", "pip"],
    );
    expect(afterCook.newlyUnlocked).toEqual(["bubu"]);

    const afterAll = evaluateUnlocks(
      { cookedCount: 2, paintedCount: 3, movedCount: 2, calledCount: 1, fedCount: 1 },
      afterCook.unlocked,
    );
    expect(afterAll.newlyUnlocked).toEqual(["lulu", "kiki", "nori"]);
    expect(afterAll.unlocked).toEqual(["momo", "pip", "bubu", "lulu", "kiki", "nori"]);
  });

  test("never reports already unlocked characters as newly unlocked", () => {
    const stats = { cookedCount: 5, paintedCount: 5, movedCount: 5, calledCount: 5, fedCount: 5 };
    const result = evaluateUnlocks(stats, ["momo", "pip", "bubu", "lulu", "kiki", "nori"]);
    expect(result.newlyUnlocked).toEqual([]);
  });
});
