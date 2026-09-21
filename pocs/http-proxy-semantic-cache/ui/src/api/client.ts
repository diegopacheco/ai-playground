import type { AskResult, CacheEntry, Stats } from "../types";

type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;

async function request<T>(fetcher: Fetcher, path: string, init?: RequestInit): Promise<T> {
  const response = await fetcher(path, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error ?? `request failed with ${response.status}`);
  }
  return body as T;
}

export function createClient(fetcher: Fetcher = (input, init) => fetch(input, init)) {
  return {
    ask: (question: string) =>
      request<AskResult>(fetcher, "/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }),
    stats: () => request<Stats>(fetcher, "/api/stats"),
    entries: () => request<{ entries: CacheEntry[] }>(fetcher, "/api/cache").then((body) => body.entries),
    clear: () => request<{ cleared: boolean }>(fetcher, "/api/cache", { method: "DELETE" }),
  };
}

export const api = createClient();
