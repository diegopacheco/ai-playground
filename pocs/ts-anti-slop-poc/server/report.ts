import type { Finding, Report, RuleCoverage, Sample, SampleResult, SourceReader } from "./types.ts";

function judge(sample: Sample, findings: readonly Finding[], source: string): SampleResult {
  const verdict = findings.length > 0 ? "caught" : "missed";
  const asExpected = sample.expect === "caught" ? findings.some((f) => f.rule === sample.rule) : verdict === "missed";

  return { ...sample, verdict, asExpected, findings, source };
}

function coverage(findings: readonly Finding[]): readonly RuleCoverage[] {
  const hits = new Map<string, number>();

  for (const finding of findings) {
    hits.set(finding.rule, (hits.get(finding.rule) ?? 0) + 1);
  }

  return [...hits.entries()].map(([rule, count]) => ({ rule, hits: count })).toSorted((a, b) => a.rule.localeCompare(b.rule));
}

export function buildReport(catalog: readonly Sample[], findings: readonly Finding[], sources: ReadonlyMap<string, string>): Report {
  const samples = catalog.map((sample) => judge(sample, findings.filter((f) => f.file === sample.file), sources.get(sample.file) ?? ""));
  const caught = samples.filter((s) => s.verdict === "caught").length;

  return {
    samples,
    caught,
    missed: samples.length - caught,
    surprises: samples.filter((s) => !s.asExpected).length,
    rules: coverage(findings),
  };
}

export async function readSources(catalog: readonly Sample[], reader: SourceReader): Promise<ReadonlyMap<string, string>> {
  const entries = await Promise.all(catalog.map(async (s) => [s.file, await reader.read(s.file)] as const));

  return new Map(entries);
}
