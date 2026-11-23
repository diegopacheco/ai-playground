import { describe, expect, test } from "bun:test";
import { buildReport } from "./report.ts";
import type { Finding, Sample } from "./types.ts";

const flagged: Sample = { file: "slop/a.ts", title: "a", slop: "a", expect: "caught", rule: "anti-slop(no-reflect-get)" };
const blind: Sample = { file: "slop/b.ts", title: "b", slop: "b", expect: "missed", gap: "no rule" };
const finding = (file: string, rule: string): Finding => ({ file, rule, message: "m", line: 1 });
const sources = new Map([["slop/a.ts", "code a"]]);

describe("buildReport", () => {
  test("a sample flagged by its target rule is caught as expected", () => {
    const report = buildReport([flagged], [finding("slop/a.ts", "anti-slop(no-reflect-get)")], sources);

    expect(report.samples[0]).toMatchObject({ verdict: "caught", asExpected: true, source: "code a" });
    expect(report.surprises).toBe(0);
  });

  test("a sample flagged only by another rule is caught but counts as a surprise, so a wrong catalog claim shows up", () => {
    const report = buildReport([flagged], [finding("slop/a.ts", "anti-slop(no-runtime-typeof)")], sources);

    expect(report.samples[0]).toMatchObject({ verdict: "caught", asExpected: false });
    expect(report.surprises).toBe(1);
  });

  test("a known blind spot that the linter starts flagging is a surprise, so the UI never hides a new catch", () => {
    const report = buildReport([blind], [finding("slop/b.ts", "anti-slop(no-runtime-typeof)")], sources);

    expect(report.samples[0]).toMatchObject({ verdict: "caught", asExpected: false });
  });

  test("findings from other files never count for a sample", () => {
    const report = buildReport([flagged, blind], [finding("slop/a.ts", "anti-slop(no-reflect-get)")], sources);

    expect(report).toMatchObject({ caught: 1, missed: 1, surprises: 0 });
    expect(report.rules).toEqual([{ rule: "anti-slop(no-reflect-get)", hits: 1 }]);
  });
});
