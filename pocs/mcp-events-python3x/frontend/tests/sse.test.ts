import { describe, expect, test } from "bun:test";
import { parseSse } from "../src/mcp/sse";

describe("parseSse", () => {
  test("keeps a frame split across network chunks until it is complete, so no notification is lost or half parsed", () => {
    const first = parseSse('data: {"a":1}\n\ndata: {"b"');
    expect(first.messages).toEqual(['{"a":1}']);
    const second = parseSse(first.rest + ':2}\n\n');
    expect(second.messages).toEqual(['{"b":2}']);
    expect(second.rest).toBe("");
  });

  test("accepts CRLF line endings and ignores comment and event lines", () => {
    const chunk = parseSse(': keepalive\r\n\r\nevent: message\r\ndata: {"c":3}\r\n\r\n');
    expect(chunk.messages).toEqual(['{"c":3}']);
  });
});
