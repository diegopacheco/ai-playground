import type { Finding, Linter, SourceReader } from "./types.ts";

export class LintError extends Error {}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null;
}

function parseFinding(raw: unknown): Finding {
  if (!isRecord(raw) || typeof raw.filename !== "string" || typeof raw.code !== "string" || typeof raw.message !== "string" || !Array.isArray(raw.labels)) {
    throw new LintError("unexpected oxlint diagnostic");
  }
  const label: unknown = raw.labels[0];
  const line = isRecord(label) && isRecord(label.span) && typeof label.span.line === "number" ? label.span.line : 0;

  return { file: raw.filename, rule: raw.code, message: raw.message, line };
}

export function parseOxlintOutput(stdout: string): readonly Finding[] {
  const raw: unknown = JSON.parse(stdout);
  if (!isRecord(raw) || !Array.isArray(raw.diagnostics)) throw new LintError("oxlint output has no diagnostics");

  return raw.diagnostics.map(parseFinding);
}

export function oxlintLinter(rootDir: string, configFile: string, target: string): Linter {
  return {
    async run() {
      const proc = Bun.spawn(["./node_modules/.bin/oxlint", "-c", configFile, "-f", "json", target], { cwd: rootDir, stdout: "pipe", stderr: "pipe" });
      const [stdout, stderr, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
      if (code > 1) throw new LintError(`oxlint failed with ${code}: ${stderr}`);

      return parseOxlintOutput(stdout);
    },
  };
}

export function fileReader(rootDir: string): SourceReader {
  return { read: (file) => Bun.file(`${rootDir}/${file}`).text() };
}
