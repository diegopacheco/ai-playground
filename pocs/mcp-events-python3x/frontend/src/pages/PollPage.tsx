import { useEffect, useRef, useState } from "react";
import { EventRow } from "../components/EventRow";
import { FilterFields } from "../components/FilterFields";
import { mcp } from "../mcp/client";
import { useEventTypes } from "../mcp/hooks";
import type { EventOccurrence, JsonObject, PollResult } from "../mcp/types";

const POLL_FLOOR_MS = 1000;

export function PollPage() {
  const types = useEventTypes();
  const [name, setName] = useState("deploy.status_changed");
  const [args, setArgs] = useState<Record<string, string>>({});
  const [cursor, setCursor] = useState<string | null>(null);
  const [lastRequest, setLastRequest] = useState<JsonObject | null>(null);
  const [lastResult, setLastResult] = useState<PollResult | null>(null);
  const [events, setEvents] = useState<EventOccurrence[]>([]);
  const [auto, setAuto] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;

  const descriptor = types.data?.find((t) => t.name === name);

  const poll = async (): Promise<PollResult | null> => {
    const params: JsonObject = { name, arguments: args, cursor: cursorRef.current, maxEvents: 20 };
    setLastRequest(params);
    try {
      const result = await mcp.request<PollResult>("events/poll", params);
      setError(null);
      setLastResult(result);
      setCursor(result.cursor);
      cursorRef.current = result.cursor;
      const newestFirst = [...result.events].reverse();
      if (newestFirst.length) setEvents((prev) => [...newestFirst, ...prev].slice(0, 200));
      return result;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setAuto(false);
      return null;
    }
  };

  useEffect(() => {
    if (!auto) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      const result = await poll();
      if (stopped || !result) return;
      timer = setTimeout(loop, result.hasMore ? 0 : Math.max(result.nextPollMs, POLL_FLOOR_MS));
    };
    void loop();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [auto, name, JSON.stringify(args)]);

  const reset = (next: string) => {
    setAuto(false);
    setName(next);
    setArgs({});
    setCursor(null);
    setEvents([]);
    setLastResult(null);
  };

  return (
    <div className="page">
      <p className="lead">
        Poll delivery: <span className="mono">events/poll</span> is plain request/response. The first poll with <b>cursor null</b> starts from
        now. Every next poll sends the cursor back and waits <span className="mono">nextPollMs</span>. The server keeps no subscription state.
      </p>
      <section className="card">
        <div className="poll-controls">
          <label className="field">
            <span>event type</span>
            <select value={name} onChange={(e) => reset(e.target.value)} disabled={auto}>
              {types.data?.map((t) => (
                <option key={t.name}>{t.name}</option>
              ))}
            </select>
          </label>
          {descriptor && <FilterFields schema={descriptor.inputSchema} values={args} onChange={setArgs} disabled={auto} />}
        </div>
        <div className="actions">
          <button onClick={() => void poll()} disabled={auto}>
            Poll once
          </button>
          <button className={auto ? "secondary" : ""} onClick={() => setAuto(!auto)}>
            {auto ? "Stop auto poll" : "Auto poll (nextPollMs)"}
          </button>
          <button className="ghost" onClick={() => reset(name)}>
            Reset cursor
          </button>
          <span className="mono dim cursor-label">
            client cursor <b>{cursor ?? "null"}</b>
          </span>
        </div>
        {error && <p className="error-text">{error}</p>}
      </section>
      <div className="poll-grid">
        <section className="card">
          <h3>Last request</h3>
          <pre className="json">{lastRequest ? JSON.stringify({ method: "events/poll", params: lastRequest }, null, 2) : "nothing yet"}</pre>
          <h3>Last response</h3>
          <pre className="json">{lastResult ? JSON.stringify(lastResult, null, 2) : "nothing yet"}</pre>
        </section>
        <section className="card">
          <h3>Events received ({events.length})</h3>
          {events.length === 0 ? <div className="empty">No events yet.</div> : <div className="feed">{events.map((e) => <EventRow key={e.eventId} event={e} />)}</div>}
        </section>
      </div>
    </div>
  );
}
