import { describe, expect, test } from "bun:test";
import { SCENES } from "../scenes";
import { clampPage, nextPage, parsePageParam, prevPage } from "./navigation";
import { narrationParts } from "./narration";
import { PAGES } from "./pages";

describe("story pages", () => {
  test("the tunnel book has ten chambers, within the 10 to 12 allowed", () => {
    expect(PAGES.length).toBe(10);
  });

  test("every page keeps a 2 line caption that fits the ribbon", () => {
    for (const page of PAGES) {
      expect(page.lines.length).toBe(2);
      for (const line of page.lines) expect(line.length).toBeLessThanOrEqual(84);
    }
  });

  test("every scene is unique and has a builder", () => {
    expect(new Set(PAGES.map((p) => p.scene)).size).toBe(PAGES.length);
    for (const page of PAGES) expect(typeof SCENES[page.scene]).toBe("function");
  });

  test("the tale keeps Carroll's order and wakes up where it began", () => {
    const order = PAGES.map((p) => p.scene);
    expect(order).toEqual(["riverbank", "rabbit", "fall", "drinkMe", "eatMe", "caterpillar", "cheshire", "teaParty", "croquet", "waking"]);
  });

  test("the famous words are all there", () => {
    const all = PAGES.flatMap((p) => [p.title, ...p.lines]).join(" ");
    for (const word of ["White Rabbit", "DRINK ME", "EAT ME", "Curiouser and curiouser", "Who are you?", "Cheshire", "Hatter", "Off with their heads!", "pack of cards"]) expect(all).toContain(word);
  });
});

describe("navigation and narration", () => {
  test("pages are clamped to the book", () => {
    expect(clampPage(0, 10)).toBe(1);
    expect(clampPage(42, 10)).toBe(10);
    expect(parsePageParam("x", 10)).toBe(1);
    expect(nextPage(10, 10)).toBe(10);
    expect(prevPage(1, 10)).toBe(1);
  });

  test("narration speaks one sentence at a time without quote marks", () => {
    const parts = narrationParts(PAGES[1]!);
    expect(parts[0]).toBe("The White Rabbit.");
    expect(parts.join(" ")).not.toMatch(/[“”"]/);
  });
});
