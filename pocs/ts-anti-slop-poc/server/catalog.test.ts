import { expect, test } from "bun:test";
import { catalog } from "./catalog.ts";
import { fileReader, oxlintLinter } from "./oxlint.ts";
import { buildReport, readSources } from "./report.ts";

const root = new URL("..", import.meta.url).pathname;

test("every catalog claim about anti-slop matches what the real linter reports on slop/", async () => {
  const [findings, sources] = await Promise.all([oxlintLinter(root, "oxlint.config.ts", "slop").run(), readSources(catalog, fileReader(root))]);
  const report = buildReport(catalog, findings, sources);
  const wrong = report.samples.filter((s) => !s.asExpected).map((s) => s.file);

  expect(wrong).toEqual([]);
  expect(report.caught).toBe(catalog.filter((s) => s.expect === "caught").length);
}, 30000);
