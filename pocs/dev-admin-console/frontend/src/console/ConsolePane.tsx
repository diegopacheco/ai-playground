import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@design/Badge/Badge";
import { Button } from "@design/Button/Button";
import { DataGrid } from "@design/DataGrid/DataGrid";
import { Pager } from "@design/Pager/Pager";
import { RowDetail } from "@design/RowDetail/RowDetail";
import { Tree } from "@design/Tree/Tree";
import { engineFor } from "@engines/index";
import { api } from "@lib/api";
import { ApiError, type Connection, type QueryResult, type SchemaNode } from "@lib/types";
import { AskAi } from "../ai/AskAi";
import { SavedQueries } from "../saved/SavedQueries";
import { QueryEditor } from "./QueryEditor";
import { ReadOnlyNotice } from "./ReadOnlyNotice";
import { RecentQueries } from "./RecentQueries";
import "./ConsolePane.css";

export interface ConsolePaneProps {
  connection: Connection;
}

interface Failure {
  message: string;
  readOnlyViolation: boolean;
}

export function ConsolePane({ connection }: ConsolePaneProps) {
  const engine = useMemo(() => engineFor(connection.kind), [connection.kind]);
  const [schema, setSchema] = useState<SchemaNode[]>([]);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [statement, setStatement] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [running, setRunning] = useState(false);
  const [cursors, setCursors] = useState<(string | null)[]>([]);
  const [totalRows, setTotalRows] = useState<number | null>(null);
  const [detailIndex, setDetailIndex] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setSchema([]);
    setSchemaError(null);
    setResult(null);
    setFailure(null);
    setTotalRows(null);
    api
      .schema(connection.id)
      .then((nodes) => {
        if (cancelled) {
          return;
        }
        setSchema(nodes);
        setStatement((current) => (current === "" ? engine.sampleStatement(nodes) : current));
      })
      .catch((error: ApiError) => {
        if (!cancelled) {
          setSchemaError(error.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [connection.id, engine]);

  const completions = useMemo(() => engine.completionsFor(schema), [engine, schema]);

  const run = useCallback(
    async (cursor: string | null, pageNumber: number, queryId?: string) => {
      if (!statement.trim()) {
        return;
      }
      setRunning(true);
      setFailure(null);
      try {
        const next = await api.query(connection.id, statement, {
          cursor: cursor ?? undefined,
          pageNumber,
          queryId
        });
        setResult(next);
        setDetailIndex(null);
      } catch (error) {
        const apiError = error as ApiError;
        setResult(null);
        setFailure({
          message: apiError.message,
          readOnlyViolation: apiError.readOnlyViolation ?? false
        });
      } finally {
        setRunning(false);
      }
    },
    [connection.id, statement]
  );

  const runFirst = useCallback(() => {
    setCursors([]);
    setTotalRows(null);
    void run(null, 1);
  }, [run]);

  const next = useCallback(() => {
    if (!result?.hasMore || !result.nextCursor) {
      return;
    }
    setCursors((current) => [...current, result.nextCursor]);
    void run(result.nextCursor, result.pageNumber + 1, result.queryId);
  }, [result, run]);

  const previous = useCallback(() => {
    if (!result || result.pageNumber <= 1) {
      return;
    }
    const target = result.pageNumber - 1;
    const cursor = target === 1 ? null : cursors[target - 2] ?? null;
    setCursors((current) => current.slice(0, Math.max(target - 1, 0)));
    void run(cursor, target, result.queryId);
  }, [cursors, result, run]);

  const count = useCallback(async () => {
    try {
      const response = await api.count(connection.id, statement);
      setTotalRows(Number(response.total));
    } catch (error) {
      setFailure({ message: (error as ApiError).message, readOnlyViolation: false });
    }
  }, [connection.id, statement]);

  const insertName = useCallback((_node: SchemaNode, path: string[]) => {
    setStatement((current) => `${current}${current.endsWith(" ") || current === "" ? "" : " "}${path.at(-1)}`);
  }, []);

  return (
    <div className="console-pane">
      <aside className="console-left">
        <header className="console-left-header">
          <span>{engine.schemaLabel}</span>
          <Badge tone="accent">{engine.label}</Badge>
        </header>
        <div className="console-left-body">
          {schemaError ? (
            <p className="console-left-error">{schemaError}</p>
          ) : (
            <Tree nodes={schema} onSelect={insertName} emptyLabel={engine.emptySchemaLabel} />
          )}
        </div>
      </aside>

      <section className="console-main">
        <div className="console-toolbar">
          <Button variant="primary" onClick={runFirst} disabled={running}>
            {running ? "running…" : "Run"}
          </Button>
          <span className="console-hint">⌘↵</span>
          <AskAi connectionId={connection.id} onUse={setStatement} />
          <SavedQueries
            projectId={connection.projectId}
            connectionId={connection.id}
            connectionKind={connection.kind}
            statement={statement}
            onUse={setStatement}
          />
          <RecentQueries connectionId={connection.id} onPick={setStatement} />
          <span className="console-target">
            {connection.host}:{connection.port}
            {connection.database ? ` / ${connection.database}` : ""}
            {connection.keyspace ? ` / ${connection.keyspace}` : ""}
          </span>
        </div>

        <div className="console-editor">
          <QueryEditor
            engine={engine}
            value={statement}
            completions={completions}
            onChange={setStatement}
            onRun={runFirst}
          />
        </div>

        {failure ? (
          <ReadOnlyNotice message={failure.message} readOnlyViolation={failure.readOnlyViolation} />
        ) : null}

        <div className="console-result">
          <DataGrid
            columns={result?.columns ?? []}
            rows={result?.rows ?? []}
            emptyLabel={result ? "0 rows returned" : "run a statement to see results"}
            onRowActivate={result && result.rows.length > 0 ? setDetailIndex : undefined}
          />
        </div>

        {result && detailIndex !== null ? (
          <RowDetail
            columns={result.columns}
            rows={result.rows}
            index={detailIndex}
            onClose={() => setDetailIndex(null)}
            onNavigate={setDetailIndex}
          />
        ) : null}

        {result ? (
          <Pager
            pageNumber={result.pageNumber}
            rowCount={result.rows.length}
            elapsedMs={result.elapsedMs}
            hasMore={result.hasMore}
            totalRows={totalRows}
            onFirst={runFirst}
            onPrevious={previous}
            onNext={next}
            onCount={engine.supportsCount ? count : undefined}
          />
        ) : null}
      </section>
    </div>
  );
}
