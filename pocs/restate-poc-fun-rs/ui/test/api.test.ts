import { afterEach, describe, expect, test } from "bun:test";
import { fetchStatus, startRun } from "../src/api.ts";

const realFetch = globalThis.fetch;
const sent: { url: string; headers: Record<string, string> }[] = [];

function recordFetch(): void {
  sent.length = 0;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    sent.push({ url, headers: init.headers as Record<string, string> });
    return new Response(JSON.stringify({ invocationId: "inv_1" }), { status: 200 });
  }) as typeof fetch;
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("ingress calls the Rust workflow accepts", () => {
  test("status sends no content-type, because the Rust SDK rejects a JSON header on a handler without input", async () => {
    recordFetch();
    await fetchStatus("run-1");
    expect(sent[0].url).toBe("/ingress/AgentWorkflow/run-1/status");
    expect(sent[0].headers["content-type"]).toBeUndefined();
  });

  test("starting a run sends the goal as JSON so the run handler can decode it", async () => {
    recordFetch();
    await startRun("learn rust");
    expect(sent[0].url).toMatch(/^\/ingress\/AgentWorkflow\/run-.+\/run\/send$/);
    expect(sent[0].headers["content-type"]).toBe("application/json");
  });
});
