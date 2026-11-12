import { describe, expect, test } from "bun:test";
import { PAGES } from "./pages";
import { clampPage, nextPage, parsePageParam, prevPage } from "./navigation";

describe("story pages", () => {
  test("the book has at least ten pages so the whole tale fits", () => {
    expect(PAGES.length).toBeGreaterThanOrEqual(10);
  });

  test("every page keeps its caption to 2 or 3 lines so it fits the bottom ribbon", () => {
    for (const page of PAGES) {
      expect(page.lines.length).toBeGreaterThanOrEqual(2);
      expect(page.lines.length).toBeLessThanOrEqual(3);
    }
  });

  test("no caption line is long enough to wrap inside the ribbon", () => {
    for (const page of PAGES) {
      for (const line of page.lines) expect(line.length).toBeLessThanOrEqual(72);
    }
  });

  test("each page has its own scene so no spread repeats", () => {
    const scenes = new Set(PAGES.map((p) => p.scene));
    expect(scenes.size).toBe(PAGES.length);
  });

  test("the tale opens on the cover and closes on the ending", () => {
    expect(PAGES[0]?.scene).toBe("cover");
    expect(PAGES[PAGES.length - 1]?.scene).toBe("theEnd");
  });

  test("the houses fall in story order: straw before sticks before the brick house holds", () => {
    const order = PAGES.map((p) => p.scene);
    expect(order.indexOf("blowStraw")).toBeLessThan(order.indexOf("blowSticks"));
    expect(order.indexOf("blowSticks")).toBeLessThan(order.indexOf("brickStands"));
  });
});

describe("navigation", () => {
  const total = 12;

  test("turning past the last page stays on the last page", () => {
    expect(nextPage(total, total)).toBe(total);
  });

  test("turning back from the first page stays on the first page", () => {
    expect(prevPage(1, total)).toBe(1);
  });

  test("next and prev move exactly one page", () => {
    expect(nextPage(4, total)).toBe(5);
    expect(prevPage(4, total)).toBe(3);
  });

  test("a bad url falls back to the first page instead of breaking the book", () => {
    expect(parsePageParam("wolf", total)).toBe(1);
    expect(parsePageParam("99", total)).toBe(total);
    expect(parsePageParam("-3", total)).toBe(1);
    expect(clampPage(Number.NaN, total)).toBe(1);
  });
});
