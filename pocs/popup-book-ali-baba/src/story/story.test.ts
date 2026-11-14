import { describe, expect, test } from "bun:test";
import { PAGES } from "./pages";
import { clampPage, nextPage, parsePageParam, prevPage } from "./navigation";
import { narrationParts } from "./narration";
import { SCENES } from "../scenes";

describe("story pages", () => {
  test("the shadow show plays twelve scenes, within the 10 to 12 allowed", () => {
    expect(PAGES.length).toBe(12);
  });

  test("every page keeps a short 2 line caption that fits the ribbon", () => {
    for (const page of PAGES) {
      expect(page.lines.length).toBe(2);
      for (const line of page.lines) expect(line.length).toBeLessThanOrEqual(80);
    }
  });

  test("every scene is unique and has a builder", () => {
    expect(new Set(PAGES.map((p) => p.scene)).size).toBe(PAGES.length);
    for (const page of PAGES) expect(typeof SCENES[page.scene]).toBe("function");
  });

  test("the plot keeps the tale's order: riders, sesame, treasure, Kassim, chalk, merchant, jars, dance, wedding", () => {
    const order = PAGES.map((p) => p.scene);
    const plot = ["riders", "sesame", "treasure", "kassim", "chalk", "merchant", "jars", "dance", "wedding"] as const;
    const positions = plot.map((s) => order.indexOf(s));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(order[0]).toBe("title");
    expect(order.at(-1)).toBe("theEnd");
  });

  test("the heroes and the magic words are all named", () => {
    const all = PAGES.flatMap((p) => [p.title, ...p.lines]).join(" ");
    for (const word of ["Ali Baba", "Morgiana", "Kassim", "Open, Sesame", "captain"]) expect(all).toContain(word);
  });
});

describe("navigation and narration", () => {
  test("pages are clamped to the show", () => {
    expect(clampPage(0, 12)).toBe(1);
    expect(clampPage(99, 12)).toBe(12);
    expect(parsePageParam("abc", 12)).toBe(1);
    expect(nextPage(12, 12)).toBe(12);
    expect(prevPage(1, 12)).toBe(1);
  });

  test("narration reads the title then each sentence on its own, without quote marks", () => {
    const parts = narrationParts(PAGES[3]!);
    expect(parts[0]).toBe("Open, Sesame!");
    expect(parts.join(" ")).not.toMatch(/[“”"]/);
  });
});
