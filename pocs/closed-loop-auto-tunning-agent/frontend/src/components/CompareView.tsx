import type { RunResult } from "../types";
import { LineChart } from "./LineChart";

type Props = { runs: RunResult[] };

const successPct = (r: RunResult) => (r.total ? Math.round((r.success / r.total) * 100) : 0);

export function CompareView({ runs }: Props) {
  if (runs.length === 0) {
    return (
      <div className="card">
        <h3>Before / after</h3>
        <p className="muted">Run traffic to capture a baseline, tune, apply, then run again to compare.</p>
      </div>
    );
  }

  const after = runs[runs.length - 1];
  const before = runs.length >= 2 ? runs[runs.length - 2] : null;

  const rows: { label: string; b: number | null; a: number; lowerBetter?: boolean }[] = [
    { label: "Success served (%)", b: before ? successPct(before) : null, a: successPct(after) },
    { label: "Short-circuited calls", b: before?.shortCircuited ?? null, a: after.shortCircuited },
    { label: "Failures hit downstream", b: before?.failure ?? null, a: after.failure, lowerBetter: true },
    { label: "Mean latency (ms)", b: before?.meanLatencyMs ?? null, a: after.meanLatencyMs, lowerBetter: true },
    { label: "p95 latency (ms)", b: before?.p95LatencyMs ?? null, a: after.p95LatencyMs, lowerBetter: true },
  ];

  const series = [];
  if (before) {
    series.push({ label: `Before · ${before.label}`, color: "#94a3b8", values: before.failureRatePoints });
  }
  series.push({ label: `After · ${after.label}`, color: "#22c55e", values: after.failureRatePoints });

  return (
    <div className="card">
      <h3>Before / after comparison</h3>
      <LineChart series={series} yMax={100} yLabel="failure %" />
      <table className="kpi">
        <thead>
          <tr>
            <th>KPI</th>
            <th>Before</th>
            <th>After</th>
            <th>Δ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const delta = r.b === null ? null : r.a - r.b;
            const good =
              delta === null || delta === 0
                ? "neutral"
                : (r.lowerBetter ? delta < 0 : delta > 0)
                  ? "good"
                  : "bad";
            return (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td>{r.b ?? "—"}</td>
                <td>{r.a}</td>
                <td className={`delta ${good}`}>
                  {delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
