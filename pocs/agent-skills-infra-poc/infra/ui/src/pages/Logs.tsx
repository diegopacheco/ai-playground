import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { fetchLogLines } from "../api";
import { useAccess, viewOf } from "../access";
import { nsToIso, shortTime } from "../format";
import { NoScope, ScopeBanner } from "./Scope";

const LEVELS = ["", "debug", "info", "warn", "error"];
const WINDOWS = [
  ["15", "15 minutes"],
  ["60", "1 hour"],
  ["1440", "24 hours"],
];

export function Logs() {
  const { session, app } = useAccess();
  const view = viewOf(session.scopes, "logs");
  const [level, setLevel] = useState("");
  const [since, setSince] = useState("60");
  const lines = useQuery({
    queryKey: ["logs", view, app, level, since],
    queryFn: () => fetchLogLines(view === "all", app, level, since),
    enabled: view !== "none",
    refetchInterval: 3000,
  });
  if (view === "none") return <NoScope capability="logs" />;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Logs</h2>
        <div className="controls">
          <select value={level} onChange={(e) => setLevel(e.target.value)} aria-label="Level">
            {LEVELS.map((l) => (
              <option key={l} value={l}>
                {l || "All levels"}
              </option>
            ))}
          </select>
          <select value={since} onChange={(e) => setSince(e.target.value)} aria-label="Window">
            {WINDOWS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <ScopeBanner capability="logs" />
      {lines.isError && <div className="alert">{lines.error.message}</div>}
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>App</th>
            <th>Level</th>
            <th>Message</th>
            <th>Fields</th>
          </tr>
        </thead>
        <tbody>
          {(lines.data ?? []).map((l) => (
            <tr key={`${l.tenant}-${l.timestamp_ns}`}>
              <td className="mono nowrap">{shortTime(nsToIso(l.timestamp_ns))}</td>
              <td>
                <span className="tenant">{l.tenant}</span>
              </td>
              <td>
                <span className={`level level-${l.level}`}>{l.level}</span>
              </td>
              <td>{l.message}</td>
              <td className="mono small">{JSON.stringify(l.fields)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {lines.data?.length === 0 && <p className="muted">No log lines in this window.</p>}
    </section>
  );
}
