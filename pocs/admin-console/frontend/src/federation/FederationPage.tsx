import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { DataGrid } from "@design/DataGrid/DataGrid";
import { EngineLogo } from "@design/EngineLogo/EngineLogo";
import { RowDetail } from "@design/RowDetail/RowDetail";
import { Tree, type TreeNode } from "@design/Tree/Tree";
import { FederatedEditor, type FederatedCompletion } from "./FederatedEditor";
import { insertAt, qualify } from "./qualify";
import { api } from "@lib/api";
import type { ApiError, ConnectionKind, FederatedResult, Project } from "@lib/types";
import "../ai/AskAi.css";
import "../console/RecentQueries.css";
import "./FederationPage.css";

export default function FederationPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [statement, setStatement] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FederatedResult | null>(null);
  const [detail, setDetail] = useState<number | null>(null);
  const [completions, setCompletions] = useState<FederatedCompletion[]>([]);
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [askOpen, setAskOpen] = useState(false);
  const [askPrompt, setAskPrompt] = useState("");
  const [asking, setAsking] = useState(false);
  const [suggestion, setSuggestion] = useState<{ statement: string; cli: string; model: string | null; parses: boolean; problem: string | null; declined?: boolean } | null>(null);

  useEffect(() => {
    api.projects().then(async (loaded) => {
      setProjects(loaded);
      const first = loaded[0];
      setProjectId((current) => current ?? first?.id ?? null);
      if (!first || first.connections.length < 2) {
        return;
      }
      const [a, b] = first.connections;
      const sourceOf = async (connection: (typeof first.connections)[number]) => {
        try {
          const schema = await api.schema(connection.id);
          return schema[0]?.name ?? null;
        } catch {
          return null;
        }
      };
      const [sourceA, sourceB] = await Promise.all([sourceOf(a), sourceOf(b)]);
      if (sourceA && sourceB) {
        setStatement(
          `SELECT x.*, y.*\nFROM ${a.name}.${sourceA} x\nJOIN ${b.name}.${sourceB} y ON x.id = y.id\nLIMIT 25`
        );
      }
    });
  }, []);

  const project = useMemo(
    () => projects.find((candidate) => candidate.id === projectId) ?? null,
    [projects, projectId]
  );

  useEffect(() => {
    if (!project) {
      return;
    }
    let cancelled = false;
    const build = async () => {
      const options: FederatedCompletion[] = [];
      const nodes: TreeNode[] = [];
      for (const connection of project.connections) {
        options.push({ label: connection.name, detail: connection.kind, type: "namespace" });
        try {
          const schema = await api.schema(connection.id);
          nodes.push({
            name: connection.name,
            kind: connection.kind,
            detail: `${schema.length} sources`,
            children: schema.map((node) => ({
              name: node.name,
              kind: node.kind,
              detail: node.detail,
              children: (node.children ?? []).map((child) => ({
                name: child.name,
                kind: child.kind,
                detail: child.detail
              }))
            }))
          });
          for (const node of schema) {
            options.push({
              label: `${connection.name}.${node.name}`,
              detail: `${connection.kind} ${node.kind}`,
              type: "class"
            });
            for (const child of node.children ?? []) {
              options.push({ label: child.name, detail: `${node.name} · ${child.detail ?? child.kind}`, type: "property" });
            }
          }
        } catch {
          continue;
        }
      }
      if (!cancelled) {
        const seen = new Set<string>();
        setCompletions(options.filter((option) => !seen.has(option.label) && seen.add(option.label)));
        setTree(nodes);
      }
    };
    void build();
    return () => {
      cancelled = true;
    };
  }, [project]);

  useEffect(() => {
    if (!projectId || !historyOpen) {
      return;
    }
    api.federatedHistory(projectId).then(setHistory).catch(() => setHistory([]));
  }, [projectId, historyOpen]);

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

  const ask = useCallback(async () => {
    if (!projectId || !askPrompt.trim()) {
      return;
    }
    setAsking(true);
    setError(null);
    setSuggestion(null);
    try {
      setSuggestion(await api.federatedAi(projectId, askPrompt));
    } catch (caught) {
      setError((caught as ApiError).message);
    } finally {
      setAsking(false);
    }
  }, [projectId, askPrompt]);

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
        <button className="askai-trigger" onClick={() => setAskOpen(!askOpen)}>
          ✦ ask ai
        </button>
        <div className="fed-recent">
          <button className="recent-queries-toggle" onClick={() => setHistoryOpen(!historyOpen)} aria-expanded={historyOpen}>
            recent ▾
          </button>
          {historyOpen ? (
            <ul className="fed-recent-list" role="listbox" aria-label="recent joins">
              {history.length === 0 ? (
                <li className="recent-queries-empty">no joins yet in this project</li>
              ) : (
                history.map((entry) => (
                  <li key={entry}>
                    <button
                      role="option"
                      aria-selected={false}
                      title={entry}
                      onClick={() => {
                        setStatement(entry);
                        setHistoryOpen(false);
                      }}
                    >
                      {entry.replace(/\s+/g, " ").slice(0, 78)}
                    </button>
                  </li>
                ))
              )}
            </ul>
          ) : null}
        </div>
      </div>

      {askOpen ? (
        <div className="fed-ask">
          <textarea
            className="fed-ask-prompt"
            value={askPrompt}
            onChange={(event) => setAskPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                void ask();
              }
              if (event.key === "Escape") {
                setAskOpen(false);
              }
            }}
            rows={2}
            autoFocus
            aria-label="describe the join you want"
            placeholder="describe the join — e.g. invoices with their matching product name"
          />
          <div className="fed-ask-actions">
            <span className="fed-hint">⌘↵ ask · esc close · the model only sees source and column names</span>
            <Button variant="primary" onClick={ask} disabled={asking || !askPrompt.trim()}>
              {asking ? "asking…" : "Ask"}
            </Button>
          </div>
          {suggestion ? (
            <div className="fed-suggestion">
              <div className="fed-suggestion-head">
                <Badge tone={suggestion.parses ? "ok" : "error"}>
                  {suggestion.parses ? "valid join" : suggestion.declined ? "no join possible" : "does not parse"}
                </Badge>
                <span className="fed-side-source">
                  {suggestion.cli}{suggestion.model ? ` · ${suggestion.model}` : ""}
                </span>
              </div>
              <pre className="fed-suggestion-sql">{suggestion.statement}</pre>
              {suggestion.declined ? (
                <p className="fed-suggestion-problem">
                  The model could not write this join — its answer is above. Usually that means the two sources
                  have no column in common.
                </p>
              ) : suggestion.problem ? (
                <p className="fed-suggestion-problem">{suggestion.problem.split("\n")[0]}</p>
              ) : null}
              <Button
                variant="primary"
                disabled={!suggestion.parses}
                onClick={() => {
                  setStatement(suggestion.statement);
                  setAskOpen(false);
                  setSuggestion(null);
                }}
              >
                use this join
              </Button>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="fed-layout">
        <div className="fed-editor-column">
          <div className="fed-editor-shell">
            <FederatedEditor value={statement} completions={completions} onChange={setStatement} onRun={run} />
          </div>
          <p className="fed-hint">⌘↵ runs · ⌃space for completions · one equality join between two sources · INNER and LEFT</p>
        </div>

        <aside className="fed-tree">
          <header className="fed-tree-header">
            <span>all schemas</span>
            <Badge tone="accent">{tree.length}</Badge>
          </header>
          <div className="fed-tree-body">
            <Tree
              nodes={tree}
              emptyLabel="no connections in this project"
              onSelect={(_node, path) => setStatement((current) => insertAt(current, qualify(current, path)))}
            />
          </div>
          <footer className="fed-tree-footer">click to insert · sources become connection.source · columns use the alias</footer>
        </aside>
      </div>
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
