import { useCallback, useEffect, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { DataGrid } from "@design/DataGrid/DataGrid";
import { EngineLogo } from "@design/EngineLogo/EngineLogo";
import { RowDetail } from "@design/RowDetail/RowDetail";
import { api } from "@lib/api";
import type { ApiError, ConnectionKind, FederatedResult, Project } from "@lib/types";
import "./FederationPage.css";

export default function FederationPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [statement, setStatement] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FederatedResult | null>(null);
  const [detail, setDetail] = useState<number | null>(null);

  useEffect(() => {
    api.projects().then((loaded) => {
      setProjects(loaded);
      const first = loaded[0];
      setProjectId((current) => current ?? first?.id ?? null);
      if (first && first.connections.length >= 2) {
        const sql = first.connections.find((c) => c.kind === "postgres" || c.kind === "mysql");
        const other = first.connections.find((c) => c.id !== sql?.id);
        if (sql && other) {
          setStatement(
            `SELECT a.id, b.key\nFROM ${sql.name}.${sql.kind === "mysql" ? "invoices" : "orders"} a\nJOIN ${other.name}.${
              other.kind === "kafka" ? "orders.events" : other.kind === "elasticsearch" ? "products" : "shop"
            } b ON a.id = b.key\nLIMIT 50`
          );
        }
      }
    });
  }, []);

  const run = useCallback(async () => {
    if (!projectId || !statement.trim()) {
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    setDetail(null);
    try {
      setResult(await api.federatedQuery(projectId, statement));
    } catch (caught) {
      setError((caught as ApiError).message);
    } finally {
      setBusy(false);
    }
  }, [projectId, statement]);

  return (
    <div className="page">
      <h1 className="fed-title">Cross-engine join</h1>
      <p className="fed-lead">
        Join two sources that live on different engines. Each side is fetched through its own engine — so the
        read-only guards, auditing and limits all still apply — then joined in memory and returned as one grid.
      </p>

      <div className="fed-form">
        <label>
          project
          <select value={projectId ?? ""} onChange={(event) => setProjectId(Number(event.target.value))}>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>{project.name}</option>
            ))}
          </select>
        </label>
        <Button variant="primary" onClick={run} disabled={busy || !statement.trim()}>
          {busy ? "joining…" : "Run join"}
        </Button>
      </div>

      <textarea
        className="fed-editor"
        value={statement}
        onChange={(event) => setStatement(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            void run();
          }
        }}
        rows={6}
        spellCheck={false}
        aria-label="federated statement"
        placeholder={"SELECT a.email, b.name\nFROM demo-postgres.customers a\nJOIN demo-elasticsearch.products b ON a.country = b.sku\nLIMIT 50"}
      />
      <p className="fed-hint">⌘↵ runs · one equality join between two sources · INNER and LEFT</p>

      {error ? <pre className="fed-error" role="alert">{error}</pre> : null}

      {result ? (
        <>
          <div className="fed-sides">
            {result.sides.map((side) => (
              <div className="fed-side" key={side.alias}>
                <EngineLogo kind={side.kind as ConnectionKind} size={22} />
                <div>
                  <div className="fed-side-head">
                    <strong>{side.alias}</strong>
                    <span className="fed-side-source">{side.connectionName}.{side.source}</span>
                  </div>
                  <div className="fed-side-rows">
                    {side.rows} rows fetched
                    {side.truncated ? <Badge tone="error">capped — join may be incomplete</Badge> : null}
                  </div>
                </div>
              </div>
            ))}
            <span className="fed-elapsed">{result.rows.length} joined rows · {result.elapsedMs}ms</span>
          </div>

          <div className="fed-result">
            <DataGrid
              columns={result.columns}
              rows={result.rows}
              emptyLabel="the join produced no rows — check that the keys really match"
              onRowActivate={result.rows.length > 0 ? setDetail : undefined}
            />
          </div>

          {detail !== null ? (
            <RowDetail
              columns={result.columns}
              rows={result.rows}
              index={detail}
              onClose={() => setDetail(null)}
              onNavigate={setDetail}
            />
          ) : null}
        </>
      ) : null}
    </div>
  );
}
