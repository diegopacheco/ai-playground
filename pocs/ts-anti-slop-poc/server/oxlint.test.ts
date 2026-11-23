import { expect, test } from "bun:test";
import { LintError, parseOxlintOutput } from "./oxlint.ts";

test("parses rule, file and line from oxlint json", () => {
  const out = JSON.stringify({ diagnostics: [{ filename: "slop/a.ts", code: "anti-slop(no-reflect-get)", message: "m", labels: [{ span: { line: 4 } }] }] });

  expect(parseOxlintOutput(out)).toEqual([{ file: "slop/a.ts", rule: "anti-slop(no-reflect-get)", message: "m", line: 4 }]);
});

test("fails loud when oxlint output changes format instead of reporting zero findings", () => {
  expect(() => parseOxlintOutput(JSON.stringify({ results: [] }))).toThrow(LintError);
});
