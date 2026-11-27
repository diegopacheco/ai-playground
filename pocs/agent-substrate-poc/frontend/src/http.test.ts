import { afterEach, describe, expect, test } from "bun:test";
import { ApiError, http } from "./http";
import { CORRELATION_HEADER, currentCorrelationId } from "./logger";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function stubFetch(status: number, body: unknown, seen: Headers[]): void {
  globalThis.fetch = Object.assign(
    async (_input: RequestInfo | URL, init?: RequestInit) => {
      seen.push(new Headers(init?.headers));
      return new Response(JSON.stringify(body), { status });
    },
    { preconnect: realFetch.preconnect },
  );
}

describe("http", () => {
  test("sends the session correlation id so browser and backend logs join", async () => {
    const seen: Headers[] = [];
    stubFetch(200, { ok: true }, seen);
    await http.get("/api/health", (raw) => raw);
    expect(seen[0]?.get(CORRELATION_HEADER)).toBe(currentCorrelationId());
  });

  test("maps backend error bodies to ApiError with the server message", async () => {
    stubFetch(404, { error: "get actor ghost: not found" }, []);
    const call = http.get("/api/actors/ghost", (raw) => raw);
    await expect(call).rejects.toBeInstanceOf(ApiError);
    await expect(call).rejects.toMatchObject({ status: 404, message: "get actor ghost: not found" });
  });
});
