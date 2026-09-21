import { describe, expect, test } from "bun:test";
import { createClient } from "./client";

function fake(status: number, body: unknown, calls: { path: string; init?: RequestInit }[] = []) {
  return createClient(async (path, init) => {
    calls.push({ path, init });
    return new Response(JSON.stringify(body), { status });
  });
}

describe("api client", () => {
  test("sends the question as JSON so the proxy can embed it", async () => {
    const calls: { path: string; init?: RequestInit }[] = [];
    await fake(200, { cached: true }, calls).ask("What is Rust?");
    expect(calls[0].path).toBe("/api/ask");
    expect(calls[0].init?.method).toBe("POST");
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ question: "What is Rust?" });
  });

  test("surfaces the proxy error message so a Claude outage is visible to the user", async () => {
    await expect(fake(502, { error: "claude is down" }).ask("x")).rejects.toThrow("claude is down");
  });

  test("falls back to the status code when the error body is not JSON", async () => {
    const client = createClient(async () => new Response("boom", { status: 500 }));
    await expect(client.stats()).rejects.toThrow("request failed with 500");
  });

  test("unwraps the cache entries list", async () => {
    const entries = await fake(200, { entries: [{ key: "qa:1", question: "q", answer: "a", hits: 2, createdAt: 1 }] }).entries();
    expect(entries.map((entry) => entry.hits)).toEqual([2]);
  });
});
