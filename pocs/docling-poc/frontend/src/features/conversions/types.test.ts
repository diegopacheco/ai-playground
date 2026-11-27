import { describe, expect, test } from "bun:test";
import { ApiError } from "../../http";
import { parseConversion, parseHistory } from "./types";

const valid = {
  id: "abc",
  filename: "a.pdf",
  format: "pdf",
  ocr: false,
  status: "success",
  createdAt: "2026-01-01T00:00:00+00:00",
  durationMs: 1200,
  document: {
    markdown: "# A",
    html: "<h1>A</h1>",
    tables: [{ index: 0, caption: "", headers: ["x"], rows: [["1"]] }],
    chunks: [{ index: 0, headings: ["A"], text: "t" }],
    stats: { pages: 1, texts: 1, tables: 1, pictures: 0, headings: 1, words: 2 },
  },
  error: null,
};

describe("parseConversion", () => {
  test("accepts the backend payload so the detail page can render it", () => {
    const c = parseConversion(valid);
    expect(c.document?.tables[0]?.rows).toEqual([["1"]]);
    expect(c.document?.stats.pages).toBe(1);
  });

  test("a failed conversion has no document but keeps the error", () => {
    const c = parseConversion({ ...valid, status: "failed", document: null, error: "bad pdf" });
    expect(c.document).toBeNull();
    expect(c.error).toBe("bad pdf");
  });

  test("an unknown status is rejected instead of rendering a wrong badge", () => {
    expect(() => parseConversion({ ...valid, status: "pending" })).toThrow(ApiError);
  });

  test("a table with non string cells is rejected", () => {
    const broken = { ...valid, document: { ...valid.document, tables: [{ index: 0, caption: "", headers: ["x"], rows: [[1]] }] } };
    expect(() => parseConversion(broken)).toThrow(ApiError);
  });
});

describe("parseHistory", () => {
  test("requires a list", () => {
    expect(() => parseHistory({})).toThrow(ApiError);
  });
});
