import type {
  StatusResponse,
  AgentAction,
  FileEntry,
  AgentLog,
  HealthResponse,
  ConversationEntry,
} from "../types";

const BASE = "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchStatus(): Promise<StatusResponse> {
  return get<StatusResponse>("/api/status");
}

export async function fetchActions(): Promise<AgentAction[]> {
  return get<AgentAction[]>("/api/actions");
}

export async function fetchFiles(): Promise<FileEntry[]> {
  return get<FileEntry[]>("/api/files");
}

export async function fetchFileContent(
  path: string
): Promise<{ path: string; content: string }> {
  return get<{ path: string; content: string }>(
    `/api/files/content?path=${encodeURIComponent(path)}`
  );
}

export async function fetchLogs(): Promise<AgentLog[]> {
  return get<AgentLog[]>("/api/logs");
}

export async function fetchHealth(): Promise<HealthResponse> {
  return get<HealthResponse>("/api/health");
}

export async function fetchConversation(): Promise<ConversationEntry[]> {
  return get<ConversationEntry[]>("/api/conversation");
}

export async function triggerRun(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/api/trigger`, { method: "POST" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
