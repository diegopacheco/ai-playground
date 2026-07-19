import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { EngineLogo } from "@design/EngineLogo/EngineLogo";
import { RowDetail } from "@design/RowDetail/RowDetail";
import { api } from "@lib/api";
import type { ApiError, ConnectionKind, Project, TraceResult } from "@lib/types";
import "./TracePage.css";

export default function TracePage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [term, setTerm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<TraceResult | null>(null);
  const [detail, setDetail] = useState<number | null>(null);

  useEffect(() => {
    api.projects().then((loaded) => {
      setProjects(loaded);
      setProjectId((current) => current ?? loaded[0]?.id ?? null);
    });
  }, []);

  const run = useCallback(
    async (event?: FormEvent) => {
      event?.preventDefault();
      if (!projectId || !term.trim()) {
        return;
      }
      setBusy(true);
      setError(null);
      setTrace(null);
      setDetail(null);
      try {
        setTrace(await api.trace(projectId, term.trim()));
      } catch (caught) {
        setError((caught as ApiError).message);
      } finally {
        setBusy(false);
      }
    },
    [projectId, term]
  );

  const byEngine = useMemo(() => {
    const groups = new Map<string, { kind: string; connectionName: string; count: number; sources: Set<string> }>();
    for (const hit of trace?.hits ?? []) {
      const key = `${hit.connectionName}`;
      const entry = groups.get(key) ?? { kind: hit.kind, connectionName: hit.connectionName, count: 0, sources: new Set<string>() };
      entry.count += 1;
      entry.sources.add(hit.source);
      groups.set(key, entry);
    }
    return [...groups.values()].sort((left, right) => right.count - left.count);
  }, [trace]);

  const timed = useMemo(() => (trace?.hits ?? []).filter((hit) => hit.at), [trace]);
  const untimed = useMemo(() => (trace?.hits ?? []).filter((hit) => !hit.at), [trace]);
  const detailHits = trace?.hits ?? [];

  return (
    <div className="page">
      <h1 className="trace-title">Entity trace</h1>
      <p className="trace-lead">
        Find one value everywhere it appears — the row, the messages about it, the cache entry, the document, the
        key. Every connection in the project is searched in parallel, read-only and bounded.
      </p>

      <form className="trace-form" onSubmit={run}>
        <label>
          project
          <select value={projectId ?? ""} onChange={(event) => setProjectId(Number(event.target.value))}>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>{project.name}</option>
            ))}
          </select>
        </label>
        <label className="trace-term">
          value
          <input
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            placeholder="1001, customer42@example.com, SKU-0042…"
            aria-label="value to trace"
            autoFocus
          />
        </label>
        <Button variant="primary" type="submit" disabled={busy || !term.trim()}>
          {busy ? "tracing…" : "Trace"}
        </Button>
      </form>

      {error ? <p className="trace-error" role="alert">{error}</p> : null}

      {trace ? (
        <>
          <div className="trace-summary">
            <strong>{trace.hits.length}</strong> hits for <code>{trace.term}</code> in {trace.elapsedMs}ms
            {trace.truncated ? <Badge tone="error">truncated</Badge> : null}
          </div>

          {byEngine.length > 0 ? (
            <div className="trace-engines">
              {byEngine.map((entry) => (
                <div className="trace-engine" key={entry.connectionName}>
                  <EngineLogo kind={entry.kind as ConnectionKind} size={22} />
                  <div>
                    <div className="trace-engine-name">{entry.connectionName}</div>
                    <div className="trace-engine-meta">
                      <strong>{entry.count}</strong> in {[...entry.sources].slice(0, 2).join(", ")}
                      {entry.sources.size > 2 ? ` +${entry.sources.size - 2}` : ""}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          {timed.length > 0 ? (
            <ol className="trace-timeline">
              {timed.map((hit, index) => (
                <li key={`t-${index}`} className="trace-event">
                  <span className="trace-time">{new Date(hit.at as string).toLocaleString()}</span>
                  <span className="trace-dot" />
                  <button className="trace-card" onClick={() => setDetail(detailHits.indexOf(hit))}>
                    <span className="trace-card-head">
                      <EngineLogo kind={hit.kind as ConnectionKind} size={18} />
                      <strong>{hit.connectionName}</strong>
                      <Badge tone="accent">{hit.kind}</Badge>
                      <span className="trace-source">{hit.source}</span>
                    </span>
                    <span className="trace-label">{hit.label}</span>
                  </button>
                </li>
              ))}
            </ol>
          ) : null}

          {untimed.length > 0 ? (
            <>
              <h2 className="trace-subtitle">no timestamp</h2>
              <p className="trace-note">
                These sources carry no time we could read, so they are not placed on the timeline rather than being
                given an invented order.
              </p>
              <ul className="trace-loose">
                {untimed.map((hit, index) => (
                  <li key={`u-${index}`}>
                    <button className="trace-card" onClick={() => setDetail(detailHits.indexOf(hit))}>
                      <span className="trace-card-head">
                        <EngineLogo kind={hit.kind as ConnectionKind} size={18} />
                        <strong>{hit.connectionName}</strong>
                        <Badge tone="accent">{hit.kind}</Badge>
                        <span className="trace-source">{hit.source}</span>
                      </span>
                      <span className="trace-label">{hit.label}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          ) : null}

          {trace.hits.length === 0 ? (
            <p className="trace-empty">nothing found for “{trace.term}” in this project.</p>
          ) : null}

          {trace.failures.length > 0 ? (
            <>
              <h2 className="trace-subtitle">not searched</h2>
              <ul className="trace-failures">
                {trace.failures.map((failure, index) => (
                  <li key={index}>
                    <Badge>{failure.kind}</Badge>
                    <strong>{failure.connectionName}</strong>
                    <span>{failure.reason}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </>
      ) : null}

      {detail !== null && detailHits[detail] ? (
        <RowDetail
          columns={detailHits[detail].columns}
          rows={detailHits.map((hit) => hit.row)}
          index={detail}
          onClose={() => setDetail(null)}
          onNavigate={setDetail}
        />
      ) : null}
    </div>
  );
}
