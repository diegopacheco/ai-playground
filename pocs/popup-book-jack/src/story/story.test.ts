import { describe, expect, test } from "bun:test";
import { PAGES } from "./pages";
import { clampPage, nextPage, parsePageParam, prevPage } from "./navigation";
import { narrationParts, narrationText } from "./narration";

const plain = (text: string) => text.replace(/[“”"]/g, "").toLowerCase();

describe("story pages", () => {
  test("the book has ten pages so the whole tale fits", () => {
    expect(PAGES.length).toBe(10);
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

  test("the plot keeps its order: beans, the beanstalk grows, the castle, the treasure, the chase", () => {
    const order = PAGES.map((p) => p.scene);
    const plot = ["beans", "window", "beanstalk", "clouds", "castle", "treasure", "chase"] as const;
    const positions = plot.map((scene) => order.indexOf(scene));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });

  test("the text is in English and the hero is Jack", () => {
    const all = PAGES.flatMap((p) => [p.title, ...p.lines]).join(" ");
    expect(all).toContain("Jack");
    expect(all).toContain("Fee-fi-fo-fum");
    expect(all).not.toMatch(/[ãçéêóõ]/i);
  });
});

describe("navigation", () => {
  const total = 10;

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

  test("a bad url falls back to a valid page instead of breaking the book", () => {
    expect(parsePageParam("giant", total)).toBe(1);
    expect(parsePageParam("99", total)).toBe(total);
    expect(parsePageParam("-3", total)).toBe(1);
    expect(clampPage(Number.NaN, total)).toBe(1);
  });
});

describe("narration", () => {
  test("the narrator reads the page title and every caption line, so the voice never drifts from the text on screen", () => {
    for (const page of PAGES) {
      const spoken = narrationText(page).toLowerCase();
      expect(spoken.startsWith(page.title.toLowerCase())).toBe(true);
      for (const line of page.lines) expect(spoken).toContain(plain(line));
    }
  });

  test("caption lines are read in the order they are printed", () => {
    for (const page of PAGES) {
      const spoken = narrationText(page).toLowerCase();
      const positions = page.lines.map((line) => spoken.indexOf(plain(line)));
      expect(positions).toEqual([...positions].sort((a, b) => a - b));
    }
  });

  test("the title ends in a full stop so the narrator pauses before the story starts", () => {
    for (const page of PAGES) expect(narrationText(page).startsWith(`${page.title}. `)).toBe(true);
  });

  test("shouted words like WHOOSH and STOMP are not left in capitals, so the voice says them instead of spelling them out", () => {
    for (const page of PAGES) expect(narrationText(page)).not.toMatch(/(?<!\p{L})\p{Lu}{2,}(?!\p{L})/u);
  });

  test("a lone capital word like I is kept, so the giant still says I smell a boy", () => {
    expect(narrationText({ scene: "cover", title: "T", lines: ["I smell a boy!", "CRASH!"] })).toContain("I smell a boy! Crash!");
  });

  test("quote marks are not spoken, so the voice never reads the punctuation in the middle of a line", () => {
    for (const page of PAGES) expect(narrationText(page)).not.toMatch(/[“”"]/);
  });

  test("the narration is split into short sentences, so mobile and Chrome voices never cut off a long page", () => {
    for (const page of PAGES) {
      const parts = narrationParts(page);
      expect(parts.join(" ")).toBe(narrationText(page));
      expect(parts.length).toBeGreaterThanOrEqual(2);
      for (const part of parts) expect(part.length).toBeLessThanOrEqual(120);
    }
  });
});
