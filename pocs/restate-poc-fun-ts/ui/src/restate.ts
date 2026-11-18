import type { RunRow, RunState } from "./types.ts";

export function sqlText(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

export function runsQuery(runId?: string): string {
  const filter = runId ? ` AND target_service_key = ${sqlText(runId)}` : "";
  return `SELECT id, target_service_key, status, created_at FROM sys_invocation WHERE target_service_name = 'AgentWorkflow' AND target_handler_name = 'run'${filter} ORDER BY created_at DESC LIMIT 20`;
}

export function restateUiUrl(adminUrl: string, invocationId: string): string {
  return `${adminUrl}/ui/invocations/${encodeURIComponent(invocationId)}`;
}

export function newRunId(now: number = Date.now()): string {
  return `run-${now.toString(36)}`;
}

export function progress(state: RunState): number {
  if (state.phase === "done") return 100;
  const total = state.steps.length + 2;
  const finished = 1 + state.steps.filter(step => step.status === "done").length;
  return state.phase === "planning" ? 0 : Math.round((finished / total) * 100);
}

export function isFinal(state: RunState | null | undefined): boolean {
  return state?.phase === "done" || state?.phase === "failed";
}

interface SqlRow {
  id: string;
  target_service_key: string;
  status: string;
  created_at: string;
}

export function toRunRows(rows: SqlRow[]): RunRow[] {
  return rows.map(row => ({ runId: row.target_service_key, invocationId: row.id, status: row.status, createdAt: row.created_at }));
}
