import { describe, expect, test } from "bun:test";
import {
  COLLECTION_KEY,
  EMPTY_COLLECTION,
  type KeyValueStore,
  addWin,
  collectionEntries,
  loadCollection,
  saveCollection,
  totalWon,
} from "./collection";
import { PLUSH_LOOKS } from "./plush";

function memoryStore(seed?: string): KeyValueStore & { readonly data: Map<string, string> } {
  const data = new Map<string, string>(seed === undefined ? [] : [[COLLECTION_KEY, seed]]);
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => {
      data.set(key, value);
    },
  };
}

const BEAR = PLUSH_LOOKS.findIndex((look) => look.name === "bear");
const FROG = PLUSH_LOOKS.findIndex((look) => look.name === "frog");

describe("collection counting", () => {
  test("winning the same kind twice counts two of that kind, not two kinds", () => {
    const collection = addWin(addWin(addWin(EMPTY_COLLECTION, BEAR), BEAR), FROG);
    expect(collection).toEqual({ bear: 2, frog: 1 });
    expect(totalWon(collection)).toBe(3);
    expect(collectionEntries(collection).map((entry) => [entry.name, entry.count])).toEqual([
      ["bear", 2],
      ["frog", 1],
    ]);
  });

  test("a win never mutates the previous collection so the HUD can see the change", () => {
    const before = addWin(EMPTY_COLLECTION, BEAR);
    const after = addWin(before, BEAR);
    expect(before).toEqual({ bear: 1 });
    expect(after).not.toBe(before);
  });

  test("an unknown plush index is ignored instead of adding a nameless entry", () => {
    expect(addWin(EMPTY_COLLECTION, 999)).toBe(EMPTY_COLLECTION);
  });

  test("kinds never won do not show up on the shelf", () => {
    expect(collectionEntries(EMPTY_COLLECTION)).toEqual([]);
    expect(totalWon(EMPTY_COLLECTION)).toBe(0);
  });
});

describe("collection persistence", () => {
  test("wins survive a reload through storage", () => {
    const store = memoryStore();
    saveCollection(store, addWin(addWin(EMPTY_COLLECTION, FROG), FROG));
    expect(loadCollection(store)).toEqual({ frog: 2 });
  });

  test("a first visit starts with an empty collection", () => {
    expect(loadCollection(memoryStore())).toEqual(EMPTY_COLLECTION);
    expect(loadCollection(undefined)).toEqual(EMPTY_COLLECTION);
  });

  test("corrupt or tampered storage cannot break the game or invent prizes", () => {
    expect(loadCollection(memoryStore("{not json"))).toEqual(EMPTY_COLLECTION);
    expect(loadCollection(memoryStore("[1,2]"))).toEqual(EMPTY_COLLECTION);
    expect(loadCollection(memoryStore('{"bear":2,"dragon":5,"frog":-1,"cat":1.5,"pig":"3"}'))).toEqual({ bear: 2 });
  });

  test("a full or blocked storage does not stop the game", () => {
    const broken: KeyValueStore = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("quota");
      },
    };
    expect(() => saveCollection(broken, { bear: 1 })).not.toThrow();
    expect(loadCollection(broken)).toEqual(EMPTY_COLLECTION);
  });
});
