import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { fetchRuns } from "../api";
import { useAccess, viewOf } from "../access";
import { NoScope, ScopeBanner } from "./Scope";
import { averageScore, latestPerSuite, percent, scoreTone, shortTime, tenants } from "../format";
import type { EvalRun } from "../types";

const column = createColumnHelper<EvalRun>();

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="score">
      <div className={`score-bar ${scoreTone(score)}`}>
        <span style={{ width: percent(score) }} />
      </div>
      <b>{percent(score)}</b>
    </div>
  );
}

const columns = [
  column.accessor("id", { header: "Run", cell: (c) => <span className="mono">#{c.getValue()}</span> }),
  column.accessor("tenant", {
    header: "Tenant",
    cell: (c) => <span className="tenant">{c.getValue()}</span>,
  }),
  column.accessor("suite", { header: "Suite" }),
  column.accessor("score", { header: "Score", cell: (c) => <ScoreBar score={c.getValue()} /> }),
  column.display({
    id: "cases",
    header: "Cases",
    cell: (c) => (
      <span>
        <span className="pass">{c.row.original.passed} pass</span> <span className="fail">{c.row.original.failed} fail</span>
      </span>
    ),
  }),
  column.accessor("created_at", { header: "When", cell: (c) => shortTime(c.getValue()) }),
];

export function Runs() {
  const navigate = useNavigate();
  const { session, app } = useAccess();
  const view = viewOf(session.scopes, "evals");
  const runs = useQuery({ queryKey: ["runs", app], queryFn: () => fetchRuns(app), refetchInterval: 5000, enabled: view !== "none" });
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);
  const data = useMemo(() => runs.data ?? [], [runs.data]);
  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (view === "none") return <NoScope capability="evals" />;
  if (runs.isPending) return <p className="muted">Loading runs</p>;
  if (runs.isError) return <div className="alert">{runs.error.message}</div>;

  const latest = latestPerSuite(data);
  return (
    <>
      <section className="tiles">
        <div className="tile">
          <span>Runs</span>
          <b>{data.length}</b>
        </div>
        <div className="tile">
          <span>Tenants</span>
          <b>{tenants(data).length}</b>
        </div>
        <div className="tile">
          <span>Average score</span>
          <b>{percent(averageScore(data))}</b>
        </div>
        <div className="tile">
          <span>Suites failing now</span>
          <b className={latest.some((r) => r.failed > 0) ? "fail" : "pass"}>{latest.filter((r) => r.failed > 0).length}</b>
        </div>
      </section>

      <section className="panel">
        <h2>Current health per tenant suite</h2>
        <div className="health">
          {latest.map((run) => (
            <button key={run.id} className={`health-card ${scoreTone(run.score)}`} onClick={() => navigate({ to: "/runs/$runId", params: { runId: String(run.id) } })}>
              <span className="tenant">{run.tenant}</span>
              <span className="suite">{run.suite}</span>
              <b>{percent(run.score)}</b>
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Runs</h2>
        </div>
        <ScopeBanner capability="evals" />
        <table>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th key={header.id} onClick={header.column.getToggleSortingHandler()} className={header.column.getCanSort() ? "sortable" : ""}>
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {{ asc: " ^", desc: " v" }[header.column.getIsSorted() as string] ?? ""}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="clickable" onClick={() => navigate({ to: "/runs/$runId", params: { runId: String(row.original.id) } })}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length === 0 && <p className="muted">No runs yet. Run an eval suite with the agentinfra CLI.</p>}
      </section>
    </>
  );
}
