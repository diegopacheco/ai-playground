import type { AskResponse, Category, Filters, SearchResponse, Stats } from "./types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.error ?? `request failed with status ${response.status}`);
  }
  return data as T;
}

function filterBody(filters: Filters) {
  return {
    category: filters.category || undefined,
    minPrice: filters.minPrice === "" ? undefined : Number(filters.minPrice),
    maxPrice: filters.maxPrice === "" ? undefined : Number(filters.maxPrice),
  };
}

export function fetchStats(): Promise<Stats> {
  return request<Stats>("/stats");
}

export function fetchCategories(): Promise<Category[]> {
  return request<Category[]>("/categories");
}

export function search(query: string, filters: Filters): Promise<SearchResponse> {
  return request<SearchResponse>("/search", { query, limit: 12, ...filterBody(filters) });
}

export function ask(question: string, filters: Filters): Promise<AskResponse> {
  return request<AskResponse>("/ask", { question, ...filterBody(filters) });
}
