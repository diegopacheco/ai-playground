import { describe, expect, test } from "bun:test";
import { averageScore, latestPerSuite, percent, scoreTone, stringify, tenants } from "./format";
import type { EvalRun } from "./types";

const run = (id: number, tenant: string, suite: string, score: number, created_at: string): EvalRun => ({
  id,
  tenant,
  suite,
  score,
  passed: Math.round(score * 10),
  failed: 10 - Math.round(score * 10),
  created_at,
});

describe("score presentation drives what a reviewer flags", () => {
  test("a suite below 60% is shown as bad so regressions stand out", () => {
    expect(scoreTone(0.59)).toBe("bad");
    expect(scoreTone(0.6)).toBe("warn");
    expect(scoreTone(0.9)).toBe("good");
  });

  test("percent rounds so 2 of 3 passing reads as 67%", () => {
    expect(percent(2 / 3)).toBe("67%");
  });

  test("average of no runs is zero instead of NaN", () => {
    expect(averageScore([])).toBe(0);
  });
});

describe("tenant overview", () => {
  const runs = [
    run(1, "bookmarks-mcp", "smoke", 0.5, "2026-11-03T10:00:00Z"),
    run(2, "bookmarks-mcp", "smoke", 1, "2026-11-03T11:00:00Z"),
    run(3, "decision-log", "smoke", 0.8, "2026-11-03T09:00:00Z"),
  ];

  test("only the newest run of each tenant suite counts as its current health", () => {
    const latest = latestPerSuite(runs);
    expect(latest.map((r) => r.id)).toEqual([2, 3]);
  });

  test("the same suite name in two tenants stays separate", () => {
    const mixed = [...runs, run(4, "decision-log", "smoke", 0.1, "2026-11-03T12:00:00Z")];
    expect(latestPerSuite(mixed).map((r) => `${r.tenant}:${r.id}`)).toEqual(["bookmarks-mcp:2", "decision-log:4"]);
  });

  test("tenant list is unique and sorted for the filter", () => {
    expect(tenants(runs)).toEqual(["bookmarks-mcp", "decision-log"]);
  });
});

test("stringify keeps strings readable and encodes structured expectations", () => {
  expect(stringify("ok")).toBe("ok");
  expect(stringify({ path: "a", equals: 1 })).toBe('{"path":"a","equals":1}');
  expect(stringify(undefined)).toBe("-");
});
