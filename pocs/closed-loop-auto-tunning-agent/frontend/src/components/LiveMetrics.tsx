import type { CbMetrics } from "../types";
import { LineChart } from "./LineChart";
import { OutcomeTally } from "./OutcomeTally";

type Props = {
  metrics?: CbMetrics;
  tally: Record<string, number>;
  points: number[];
};

export function LiveMetrics({ metrics, tally, points }: Props) {
  const state = metrics?.state ?? "UNKNOWN";
  const failureRate = metrics && metrics.failureRate >= 0 ? metrics.failureRate : 0;

  return (
    <div className="card">
      <div className="card-head">
        <h3>Live circuit breaker</h3>
        <span className={`state state-${state.toLowerCase()}`}>{state}</span>
      </div>
      <div className="metric-grid">
        <Metric label="Failure rate" value={`${failureRate.toFixed(0)}%`} />
        <Metric label="Buffered" value={metrics?.bufferedCalls ?? 0} />
        <Metric label="Failed" value={metrics?.failedCalls ?? 0} />
        <Metric label="Successful" value={metrics?.successfulCalls ?? 0} />
        <Metric label="Not permitted" value={metrics?.notPermittedCalls ?? 0} />
      </div>
      <LineChart
        series={[{ label: "Measured failure rate", color: "#f0a020", values: points }]}
        yMax={100}
      />
      <OutcomeTally tally={tally} />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric">
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  );
}
