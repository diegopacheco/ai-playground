import { newRunId, runsQuery, toRunRows } from "./restate.ts";
import type { RunRow, RunState } from "./types.ts";

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: body === undefined ? { accept: "application/json" } : { "content-type": "application/json", accept: "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export async function startRun(goal: string): Promise<string> {
  const runId = newRunId();
  await postJson<{ invocationId: string }>(`/ingress/AgentWorkflow/${runId}/run/send`, { goal });
  return runId;
}

export function fetchStatus(runId: string): Promise<RunState | null> {
  return postJson<RunState | null>(`/ingress/AgentWorkflow/${encodeURIComponent(runId)}/status`);
}

export async function fetchRuns(runId?: string): Promise<RunRow[]> {
  const result = await postJson<{ rows: Parameters<typeof toRunRows>[0] }>("/admin/query", { query: runsQuery(runId) });
  return toRunRows(result.rows);
}
