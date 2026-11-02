import type { EventOccurrence, StreamNotification } from "./types";

export type FeedEntry =
  | { kind: "active"; cursor: string | null; truncated: boolean; at: number }
  | { kind: "event"; event: EventOccurrence; at: number }
  | { kind: "heartbeat"; cursor: string | null; at: number }
  | { kind: "error"; message: string; at: number };

export type FeedState = {
  cursor: string | null;
  entries: FeedEntry[];
  seen: string[];
  duplicates: number;
};

export const MAX_ENTRIES = 200;

export const emptyFeed: FeedState = { cursor: null, entries: [], seen: [], duplicates: 0 };

function push(state: FeedState, entry: FeedEntry, cursor: string | null): FeedState {
  return { ...state, cursor: cursor ?? state.cursor, entries: [entry, ...state.entries].slice(0, MAX_ENTRIES) };
}

export function applyNotification(state: FeedState, message: StreamNotification, at: number): FeedState {
  switch (message.method) {
    case "notifications/events/active":
      return push(state, { kind: "active", cursor: message.params.cursor, truncated: message.params.truncated, at }, message.params.cursor);
    case "notifications/events/heartbeat":
      return push(state, { kind: "heartbeat", cursor: message.params.cursor, at }, message.params.cursor);
    case "notifications/events/event": {
      const { _meta, ...event } = message.params;
      if (state.seen.includes(event.eventId)) {
        return { ...state, cursor: event.cursor ?? state.cursor, duplicates: state.duplicates + 1 };
      }
      const next = push(state, { kind: "event", event, at }, event.cursor ?? null);
      return { ...next, seen: [event.eventId, ...state.seen].slice(0, MAX_ENTRIES * 5) };
    }
    case "notifications/events/error":
    case "notifications/events/terminated":
      return push(state, { kind: "error", message: `${message.method.split("/").pop()}: ${message.params.error.message}`, at }, null);
  }
}

export function recordError(state: FeedState, message: string, at: number): FeedState {
  return push(state, { kind: "error", message, at }, null);
}
