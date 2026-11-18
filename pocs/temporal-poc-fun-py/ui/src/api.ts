import type { RunRow, RunView } from "./types.ts";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, headers: { "content-type": "application/json", accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export async function startRun(goal: string): Promise<string> {
  const result = await request<{ runId: string }>("/api/runs", { method: "POST", body: JSON.stringify({ goal }) });
  return result.runId;
}

export function fetchRun(runId: string): Promise<RunView> {
  return request<RunView>(`/api/runs/${encodeURIComponent(runId)}`);
}

export function fetchRuns(): Promise<RunRow[]> {
  return request<RunRow[]>("/api/runs");
}
