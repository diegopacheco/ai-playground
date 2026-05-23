import { describe, test, expect } from "bun:test";
import { formatSse, sseHeaders } from "../app/lib/sse.ts";

const decoder = new TextDecoder();

describe("formatSse", () => {
  test("emits event + data lines terminated by blank line", () => {
    const bytes = formatSse({ event: "step", data: { step: 1 } });
    const text = decoder.decode(bytes);
    expect(text).toContain("event: step");
    expect(text).toContain('data: {"step":1}');
    expect(text.endsWith("\n\n")).toBe(true);
  });

  test("includes id line when present", () => {
    const bytes = formatSse({ id: "abc", event: "status", data: { ok: true } });
    const text = decoder.decode(bytes);
    expect(text.split("\n")[0]).toBe("id: abc");
  });

  test("serializes nested data via JSON.stringify", () => {
    const bytes = formatSse({
      event: "result",
      data: { script: "test('x', async () => {})", stopReason: "done" },
    });
    const text = decoder.decode(bytes);
    expect(text).toContain('"script":"test(\'x\', async () => {})"');
  });
});

describe("sseHeaders", () => {
  test("returns text/event-stream content type and no-buffering headers", () => {
    const headers = new Headers(sseHeaders());
    expect(headers.get("Content-Type")).toContain("text/event-stream");
    expect(headers.get("Cache-Control")).toContain("no-cache");
    expect(headers.get("X-Accel-Buffering")).toBe("no");
  });
});
