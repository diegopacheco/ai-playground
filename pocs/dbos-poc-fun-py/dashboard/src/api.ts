import type { WorkflowDetail, WorkflowFilter, WorkflowList } from "./types.ts";

async function request<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

export function fetchWorkflows(filter: WorkflowFilter): Promise<WorkflowList> {
  const params = new URLSearchParams({ status: filter.status, search: filter.search });
  return request<WorkflowList>(`/api/workflows?${params}`);
}

export function fetchWorkflow(id: string): Promise<WorkflowDetail> {
  return request<WorkflowDetail>(`/api/workflows/${encodeURIComponent(id)}`);
}
