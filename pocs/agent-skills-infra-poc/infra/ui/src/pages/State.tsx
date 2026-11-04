import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { fetchStateEntries } from "../api";
import { useAccess, viewOf } from "../access";
import { shortTime } from "../format";
import { NoScope, ScopeBanner } from "./Scope";

export function State() {
  const { session, app } = useAccess();
  const view = viewOf(session.scopes, "state");
  const [prefix, setPrefix] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const entries = useQuery({
    queryKey: ["state", view, app, prefix],
    queryFn: () => fetchStateEntries(view === "all", app, prefix),
    enabled: view !== "none",
    refetchInterval: 3000,
  });
  if (view === "none") return <NoScope capability="state" />;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>State</h2>
        <div className="controls">
          <input value={prefix} onChange={(e) => setPrefix(e.target.value)} placeholder="key prefix, like adr/ or flag/" aria-label="Key prefix" />
        </div>
      </div>
      <ScopeBanner capability="state" />
      {entries.isError && <div className="alert">{entries.error.message}</div>}
      <table>
        <thead>
          <tr>
            <th>App</th>
            <th>Key</th>
            <th>Value</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          {(entries.data ?? []).map((e) => {
            const id = `${e.tenant}/${e.key}`;
            const text = JSON.stringify(e.value, null, open === id ? 2 : undefined);
            return (
              <tr key={id} className="clickable" onClick={() => setOpen(open === id ? null : id)}>
                <td>
                  <span className="tenant">{e.tenant}</span>
                </td>
                <td className="mono">{e.key}</td>
                <td>{open === id ? <pre>{text}</pre> : <span className="mono small">{text.length > 110 ? `${text.slice(0, 110)}...` : text}</span>}</td>
                <td className="nowrap">{shortTime(e.updated_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {entries.data?.length === 0 && <p className="muted">No keys.</p>}
    </section>
  );
}
