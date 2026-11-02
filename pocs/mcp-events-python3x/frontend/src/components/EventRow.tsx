import type { EventOccurrence } from "../mcp/types";

const shortTime = (iso: string) => new Date(iso).toLocaleTimeString([], { hour12: false });

export function EventRow({ event }: { event: EventOccurrence }) {
  const data = event.data;
  const status = typeof data.status === "string" ? data.status : null;
  const severity = typeof data.severity === "string" ? data.severity : null;
  return (
    <div className="event-row">
      <span className="mono dim">{shortTime(event.timestamp)}</span>
      <span className="mono">{event.eventId}</span>
      {status && <span className={`chip status-${status}`}>{status}</span>}
      {severity && <span className={`chip sev-${severity}`}>{severity}</span>}
      <span className="event-text">
        {String(data.service ?? data.message ?? "")} {data.env ? <b>{String(data.env)}</b> : null}{" "}
        <span className="mono dim">{String(data.deployId ?? "")}</span>
      </span>
      <span className="client-badge">{String(data.triggeredBy ?? "")}</span>
      {event.cursor !== undefined && <span className="mono cursor">{event.cursor ?? "null"}</span>}
    </div>
  );
}
