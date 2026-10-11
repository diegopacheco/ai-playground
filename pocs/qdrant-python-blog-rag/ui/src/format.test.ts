import { expect, test } from "bun:test";
import { excerpt, matchPercent, publishedDay } from "./format";

test("published day ignores the time zone so a post never shows the wrong date", () => {
  expect(publishedDay("2026-09-27T23:59:51.369-03:00")).toBe("2026-09-27");
});

test("short text is shown whole", () => {
  expect(excerpt("qdrant is fast", 50)).toBe("qdrant is fast");
});

test("long text is cut at a word boundary so a card never ends mid word", () => {
  expect(excerpt("vector databases store embeddings", 20)).toBe("vector databases…");
});

test("text without spaces is still cut to the limit", () => {
  expect(excerpt("a".repeat(30), 10)).toBe(`${"a".repeat(10)}…`);
});

test("cosine score is shown as a rounded percentage", () => {
  expect(matchPercent(0.8765)).toBe("88% match");
});
