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

  test("the plot keeps its order: lost, then the candy house, the witch, the cage, the bone trick and the oven", () => {
    const order = PAGES.map((p) => p.scene);
    const plot = ["crumbs", "lost", "candyHouse", "witch", "cage", "bone", "oven"] as const;
    const positions = plot.map((scene) => order.indexOf(scene));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });

  test("the text is in Portuguese: the heroes are named João and Maria and accents are kept", () => {
    const all = PAGES.flatMap((p) => [p.title, ...p.lines]).join(" ");
    expect(all).toContain("João");
    expect(all).toContain("Maria");
    expect(all).not.toMatch(/Hansel|Gretel|witch|the /i);
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
    expect(parsePageParam("bruxa", total)).toBe(1);
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

  test("shouted words like PLAFT are not left in capitals, so the voice says them instead of spelling them out", () => {
    for (const page of PAGES) expect(narrationText(page)).not.toMatch(/(?<!\p{L})\p{Lu}{2,}(?!\p{L})/u);
  });

  test("shouted words with accents are lowered too, so ÁGUA is read as a word", () => {
    expect(narrationText({ scene: "cover", title: "T", lines: ["Splash de ÁGUA!", "fim"] })).toContain("Água");
  });

  test("quote marks are not spoken, so the voice never says aspas in the middle of a line", () => {
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
