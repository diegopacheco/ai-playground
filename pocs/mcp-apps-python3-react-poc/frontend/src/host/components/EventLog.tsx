import type { HostEvent } from "./AppFrame";

type EventLogProps = {
  events: HostEvent[];
};

export function EventLog({ events }: EventLogProps) {
  return (
    <aside className="log">
      <h2>Host bridge log</h2>
      {events.length === 0 ? (
        <p className="muted">Messages between the host, the view and the MCP server show up here.</p>
      ) : (
        <ol>
          {events.map((event, index) => (
            <li key={index} className={`event event-${event.kind}`}>
              <span className="tag">{event.kind}</span>
              <code>{event.text}</code>
            </li>
          ))}
        </ol>
      )}
    </aside>
  );
}
