import { isGridKey, nextIndex } from "./gridKeys";

describe("gridKeys", () => {
  it("recognises the four arrows and nothing else", () => {
    expect(isGridKey("ArrowUp")).toBe(true);
    expect(isGridKey("ArrowRight")).toBe(true);
    expect(isGridKey("Enter")).toBe(false);
  });

  describe("in a 3-column grid of 7 items", () => {
    const count = 7;
    const columns = 3;

    it("moves right along the row", () => {
      expect(nextIndex("ArrowRight", 0, count, columns)).toBe(1);
    });

    it("moves down a whole row, not one item", () => {
      expect(nextIndex("ArrowDown", 0, count, columns)).toBe(3);
    });

    it("moves up a whole row", () => {
      expect(nextIndex("ArrowUp", 4, count, columns)).toBe(1);
    });

    it("wraps right past the end back to the start", () => {
      expect(nextIndex("ArrowRight", 6, count, columns)).toBe(0);
    });

    it("wraps left before the start to the end", () => {
      expect(nextIndex("ArrowLeft", 0, count, columns)).toBe(6);
    });

    it("returns to the same column when moving down past the last row", () => {
      expect(nextIndex("ArrowDown", 6, count, columns)).toBe(0);
    });

    it("jumps to the bottom of the column when moving up from the first row", () => {
      expect(nextIndex("ArrowUp", 1, count, columns)).toBe(4);
    });

    it("never lands past the end when the last row is short", () => {
      for (let index = 0; index < count; index++) {
        for (const key of ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"] as const) {
          const next = nextIndex(key, index, count, columns);
          expect(next).toBeGreaterThanOrEqual(0);
          expect(next).toBeLessThan(count);
        }
      }
    });
  });

  it("behaves like a plain list when there is one column", () => {
    expect(nextIndex("ArrowDown", 0, 5, 1)).toBe(1);
    expect(nextIndex("ArrowUp", 0, 5, 1)).toBe(4);
  });

  it("stays at zero when there is nothing to move through", () => {
    expect(nextIndex("ArrowDown", 0, 0, 3)).toBe(0);
  });
});
