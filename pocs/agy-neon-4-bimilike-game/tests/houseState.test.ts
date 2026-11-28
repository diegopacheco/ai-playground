import { describe, expect, test } from "bun:test";
import { HOUSE_BOUNDS } from "../src/features/house/catalog";
import {
  createInitialHouseState,
  loadHouseState,
  parseHouseState,
  reduceHouseState,
  saveHouseState,
} from "../src/features/house/state";
import type { StorageAdapter } from "../src/lib/storage";

function createMemoryStorage(): StorageAdapter {
  const store = new Map<string, string>();
  return {
    read: (k) => store.get(k) ?? null,
    write: (k, v) => {
      store.set(k, v);
    },
    remove: (k) => {
      store.delete(k);
    },
  };
}

describe("house state domain transitions", () => {
  test("clamps furniture positions within dollhouse boundaries and unlocks Kiki after two moves", () => {
    let state = createInitialHouseState();
    const move1 = reduceHouseState(state, { type: "moveFurniture", id: "sofa", x: 99, z: -99 });
    state = move1.state;
    const sofa = state.furniture.find((f) => f.id === "sofa");
    expect(sofa?.x).toBe(HOUSE_BOUNDS.maxX);
    expect(sofa?.z).toBe(HOUSE_BOUNDS.minZ);
    expect(state.stats.movedCount).toBe(1);
    expect(move1.newlyUnlocked).toEqual([]);

    const move2 = reduceHouseState(state, { type: "moveFurniture", id: "table", x: -1.5, z: 0.8 });
    expect(move2.state.stats.movedCount).toBe(2);
    expect(move2.newlyUnlocked).toEqual(["kiki"]);
    expect(move2.state.characters.some((c) => c.id === "kiki")).toBe(true);
  });

  test("paints furniture and walls, cycles styles, and unlocks Lulu on third paint", () => {
    let state = createInitialHouseState();
    state = reduceHouseState(state, { type: "paintFurniture", id: "sofa", color: "#ff8fab" }).state;
    state = reduceHouseState(state, { type: "cycleFurnitureStyle", id: "sofa" }).state;
    const third = reduceHouseState(state, { type: "paintSurface", id: "roof", color: "#74c0fc" });
    expect(third.state.surfaces.roof).toBe("#74c0fc");
    expect(third.state.furniture.find((f) => f.id === "sofa")?.styleVariant).toBe(1);
    expect(third.newlyUnlocked).toEqual(["lulu"]);
  });

  test("cooks dishes, caps table dishes, feeds a friend, and unlocks Bubu", () => {
    let state = createInitialHouseState();
    state = reduceHouseState(state, { type: "cookDish", dishId: "pancakes", uid: "d-1" }).state;
    const secondCook = reduceHouseState(state, { type: "cookDish", dishId: "pizza", uid: "d-2" });
    expect(secondCook.newlyUnlocked).toEqual(["bubu"]);

    const fed = reduceHouseState(secondCook.state, {
      type: "feedCharacter",
      dishUid: "d-2",
      characterId: "momo",
      now: 1000,
    });
    expect(fed.state.dishes.some((d) => d.uid === "d-2")).toBe(false);
    expect(fed.state.stats.fedCount).toBe(1);
    expect(fed.state.characters.find((c) => c.id === "momo")?.mood).toBe("yummy");
  });

  test("records phone call and unlocks Nori", () => {
    const initial = createInitialHouseState();
    const called = reduceHouseState(initial, {
      type: "recordCall",
      characterId: "pip",
      mood: "dancing",
      now: 2000,
    });
    expect(called.newlyUnlocked).toEqual(["nori"]);
    expect(called.state.characters.find((c) => c.id === "pip")?.mood).toBe("dancing");
  });

  test("persists and safely recovers from invalid storage payloads", () => {
    const storage = createMemoryStorage();
    const initial = createInitialHouseState();
    const updated = reduceHouseState(initial, { type: "paintSurface", id: "kitchenWall", color: "#8ce99a" }).state;
    saveHouseState(storage, updated);

    const loaded = loadHouseState(storage);
    expect(loaded.surfaces.kitchenWall).toBe("#8ce99a");
    expect(loaded.stats.paintedCount).toBe(1);

    const recovered = parseHouseState({ surfaces: "bad", furniture: 42, stats: null });
    expect(recovered.furniture.length).toBe(initial.furniture.length);
    expect(recovered.unlockedCharacters).toEqual(["momo", "pip"]);
  });
});
