import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

export function LiveFeed({ name }: { name: string }) {
  const events = useQuery({ queryKey: ["events", name], queryFn: () => api.events(name), refetchInterval: 2000 });
  const items = (events.data ?? []).slice(0, 30);
  return (
    <aside className="card feed">
      <h2>Live decisions</h2>
      <p className="hint">What the policy engine decides while you type. Try <code>curl https://example.com</code>.</p>
      {items.length === 0 && <p className="empty">No network activity yet.</p>}
      <ul>
        {items.map((e) => (
          <li key={`${e.timeMs}-${e.kind}-${e.target}`} className={e.decision.toLowerCase()}>
            <div className="feed-head">
              <span className={`badge ${e.decision.toLowerCase()}`}>{e.decision}</span>
              <span className="mono">{e.kind}</span>
              <span className="muted">{new Date(e.timeMs).toLocaleTimeString()}</span>
            </div>
            <code>{e.target}</code>
            {e.reason && <div className="reason">{e.reason}</div>}
          </li>
        ))}
      </ul>
    </aside>
  );
}
