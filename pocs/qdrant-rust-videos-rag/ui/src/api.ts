import type { Answer, Hit, SearchRequest, Stats } from "./types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error ?? `request failed with status ${response.status}`);
  }
  return payload as T;
}

export function fetchStats(): Promise<Stats> {
  return request<Stats>("/api/stats");
}

export async function searchVideos(search: SearchRequest): Promise<Hit[]> {
  return (await request<{ hits: Hit[] }>("/api/search", search)).hits;
}

export async function similarVideos(videoId: string): Promise<Hit[]> {
  return (await request<{ hits: Hit[] }>(`/api/similar/${encodeURIComponent(videoId)}`)).hits;
}

export function askQuestion(question: string): Promise<Answer> {
  return request<Answer>("/api/ask", { question });
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
