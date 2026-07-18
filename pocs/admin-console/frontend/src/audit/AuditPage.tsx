import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { api } from "@lib/api";
import type { ApiError, AuditEntry } from "@lib/types";
import "./AuditPage.css";

interface Group {
  queryId: string;
  first: AuditEntry;
  pages: AuditEntry[];
}

function groupByQuery(entries: AuditEntry[]): Group[] {
  const groups = new Map<string, Group>();
  for (const entry of entries) {
    const existing = groups.get(entry.queryId);
    if (existing) {
      existing.pages.push(entry);
      if (entry.page < existing.first.page) {
        existing.first = entry;
      }
    } else {
      groups.set(entry.queryId, { queryId: entry.queryId, first: entry, pages: [entry] });
    }
  }
  return [...groups.values()];
}

export default function AuditPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [user, setUser] = useState("");
  const [allowed, setAllowed] = useState<"" | "true" | "false">("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    api
      .audit({ user: user || undefined, allowed: allowed === "" ? undefined : allowed, size: 200 })
      .then((response) => setEntries(response.entries))
      .catch((caught: ApiError) => setError(caught.message));
  }, [user, allowed]);

  useEffect(load, [load]);

  const groups = useMemo(() => groupByQuery(entries), [entries]);

  if (error) {
    return (
      <div className="page">
        <p className="audit-error" role="alert">{error}</p>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="audit-filters">
        <label>
          user
          <input value={user} onChange={(event) => setUser(event.target.value)} placeholder="all users" />
        </label>
        <label>
          outcome
          <select value={allowed} onChange={(event) => setAllowed(event.target.value as "" | "true" | "false")}>
            <option value="">all</option>
            <option value="true">allowed</option>
            <option value="false">denied</option>
          </select>
        </label>
        <Button onClick={load}>refresh</Button>
        <a className="audit-export" href="/api/audit/export.csv">export csv</a>
        <span className="audit-count">{groups.length} queries · {entries.length} rows</span>
      </div>

      <table className="audit-table">
        <thead>
          <tr>
            <th>when</th>
            <th>user</th>
            <th>engine</th>
            <th>statement</th>
            <th>outcome</th>
            <th>pages</th>
            <th>rows</th>
            <th>ms</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((group) => {
            const total = group.pages.reduce((sum, page) => sum + (page.elapsedMs ?? 0), 0);
            const rows = group.pages.reduce((sum, page) => sum + (page.rowCount ?? 0), 0);
            const open = expanded === group.queryId;
            return [
              <tr key={group.queryId} className={group.first.allowed ? "" : "audit-denied"}>
                <td>{new Date(group.first.at).toLocaleString()}</td>
                <td>{group.first.username}</td>
                <td>{group.first.kind}</td>
                <td className="audit-statement" title={group.first.statement}>
                  <button onClick={() => setExpanded(open ? null : group.queryId)}>
                    {group.first.statement.length > 80
                      ? `${group.first.statement.slice(0, 80)}…`
                      : group.first.statement}
                  </button>
                  {group.first.denialReason ? (
                    <div className="audit-reason">{group.first.denialReason}</div>
                  ) : null}
                  {group.first.error ? <div className="audit-reason">{group.first.error}</div> : null}
                </td>
                <td>
                  <Badge tone={group.first.allowed ? "ok" : "error"}>
                    {group.first.allowed ? "allowed" : "denied"}
                  </Badge>
                </td>
                <td>{group.pages.length}</td>
                <td>{rows}</td>
                <td>{total}</td>
              </tr>,
              open ? (
                <tr key={`${group.queryId}-detail`} className="audit-detail">
                  <td colSpan={8}>
                    <pre>{group.first.statement}</pre>
                    <div className="audit-pages">
                      {group.pages
                        .slice()
                        .sort((left, right) => left.page - right.page)
                        .map((page) => (
                          <span key={page.id}>
                            page {page.page}: {page.rowCount ?? 0} rows in {page.elapsedMs ?? 0}ms
                          </span>
                        ))}
                    </div>
                  </td>
                </tr>
              ) : null
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}
