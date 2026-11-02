import type { FeedEntry } from "../mcp/feed";
import { EventRow } from "./EventRow";

const clock = (at: number) => new Date(at).toLocaleTimeString([], { hour12: false });

function Control({ entry }: { entry: Exclude<FeedEntry, { kind: "event" }> }) {
  switch (entry.kind) {
    case "active":
      return (
        <div className="control active">
          {clock(entry.at)} notifications/events/active cursor <b>{entry.cursor ?? "null"}</b>
          {entry.truncated && <span className="chip sev-P2">truncated</span>}
        </div>
      );
    case "heartbeat":
      return (
        <div className="control heartbeat">
          {clock(entry.at)} notifications/events/heartbeat cursor <b>{entry.cursor ?? "null"}</b>
        </div>
      );
    case "error":
      return <div className="control error">{clock(entry.at)} {entry.message}</div>;
  }
}

export function FeedList({ entries, empty }: { entries: FeedEntry[]; empty: string }) {
  if (entries.length === 0) return <div className="empty">{empty}</div>;
  return (
    <div className="feed">
      {entries.map((entry, index) =>
        entry.kind === "event" ? <EventRow key={entry.event.eventId} event={entry.event} /> : <Control key={`${entry.kind}-${entry.at}-${index}`} entry={entry} />,
      )}
    </div>
  );
}
