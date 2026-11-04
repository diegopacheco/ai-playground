import type { EvalRun } from "./types";

export function percent(score: number): string {
  return `${Math.round(score * 100)}%`;
}

export function scoreTone(score: number): "good" | "warn" | "bad" {
  if (score >= 0.9) return "good";
  if (score >= 0.6) return "warn";
  return "bad";
}

export function tenants(runs: EvalRun[]): string[] {
  return [...new Set(runs.map((run) => run.tenant))].sort();
}

export function averageScore(runs: EvalRun[]): number {
  if (runs.length === 0) return 0;
  return runs.reduce((sum, run) => sum + run.score, 0) / runs.length;
}

export function latestPerSuite(runs: EvalRun[]): EvalRun[] {
  const latest = new Map<string, EvalRun>();
  for (const run of runs) {
    const key = `${run.tenant}/${run.suite}`;
    const current = latest.get(key);
    if (!current || run.created_at > current.created_at) latest.set(key, run);
  }
  return [...latest.values()].sort((a, b) => a.tenant.localeCompare(b.tenant) || a.suite.localeCompare(b.suite));
}

export function shortTime(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function stringify(value: unknown): string {
  if (value === null || value === undefined) return "-";
  return typeof value === "string" ? value : JSON.stringify(value);
}

const KIND_ORDER: Record<string, number> = { admin: 0, skill: 1, mcp: 2, test: 3 };

export function sortCredentials<T extends { client_id: string; kind: string }>(items: T[]): T[] {
  return [...items].sort(
    (a, b) => (KIND_ORDER[a.kind] ?? 9) - (KIND_ORDER[b.kind] ?? 9) || a.client_id.localeCompare(b.client_id),
  );
}

export function hasScope(scopes: string[], scope: string): boolean {
  return scopes.includes(scope);
}

export function nsToIso(timestampNs: string): string {
  return new Date(Number(BigInt(timestampNs) / 1_000_000n)).toISOString();
}

export function decodeClaims(token: string): Record<string, unknown> {
  try {
    const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(payload.padEnd(payload.length + ((4 - (payload.length % 4)) % 4), "="))) as Record<string, unknown>;
  } catch {
    return {};
  }
}
