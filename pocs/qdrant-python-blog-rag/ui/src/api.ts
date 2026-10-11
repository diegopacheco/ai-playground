import type { Answer, Filters, SearchResult, Stats } from "./types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const init = body === undefined
    ? undefined
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, init);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ?? response.statusText);
  }
  return data as T;
}

export function fetchStats(): Promise<Stats> {
  return request<Stats>("/api/stats");
}

export function searchPosts(query: string, filters: Filters): Promise<SearchResult> {
  return request<SearchResult>("/api/search", { query, ...filters });
}

export function askBlog(question: string, filters: Filters): Promise<Answer> {
  return request<Answer>("/api/ask", { question, ...filters });
}
