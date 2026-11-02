import { describe, expect, test } from "bun:test";
import { MAX_ENTRIES, applyNotification, emptyFeed, type FeedState } from "../src/mcp/feed";
import type { StreamNotification } from "../src/mcp/types";

const meta = { "io.modelcontextprotocol/subscriptionId": 1 };

const event = (id: string, cursor: string): StreamNotification => ({
  method: "notifications/events/event",
  params: { eventId: id, name: "deploy.status_changed", timestamp: "2026-11-02T12:00:00Z", data: { status: "queued" }, cursor, _meta: meta },
});

const apply = (messages: StreamNotification[]): FeedState => messages.reduce((state, m) => applyNotification(state, m, 0), emptyFeed);

describe("applyNotification", () => {
  test("client owns the cursor: it follows active, event and heartbeat so a reconnect resumes at the right place", () => {
    const state = apply([
      { method: "notifications/events/active", params: { cursor: "c_3", truncated: false, _meta: meta } },
      event("evt_4", "c_4"),
      { method: "notifications/events/heartbeat", params: { cursor: "c_9", _meta: meta } },
    ]);
    expect(state.cursor).toBe("c_9");
  });

  test("a replayed eventId is dropped so a resumed stream never shows the same event twice", () => {
    const state = apply([event("evt_1", "c_1"), event("evt_1", "c_1"), event("evt_2", "c_2")]);
    const ids = state.entries.flatMap((e) => (e.kind === "event" ? [e.event.eventId] : []));
    expect(ids).toEqual(["evt_2", "evt_1"]);
    expect(state.duplicates).toBe(1);
  });

  test("a null cursor from a server without replay does not erase the cursor the client already holds", () => {
    const state = apply([event("evt_1", "c_1"), { method: "notifications/events/heartbeat", params: { cursor: null, _meta: meta } }]);
    expect(state.cursor).toBe("c_1");
  });

  test("the feed is bounded so a long running stream does not grow memory forever", () => {
    const many = Array.from({ length: MAX_ENTRIES + 50 }, (_, i) => event(`evt_${i}`, `c_${i}`));
    expect(apply(many).entries.length).toBe(MAX_ENTRIES);
  });
});
