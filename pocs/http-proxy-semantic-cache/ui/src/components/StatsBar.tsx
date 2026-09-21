import { hitRate } from "../lib/format";
import type { Stats } from "../types";

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  );
}

export function StatsBar({ stats }: { stats: Stats | null }) {
  if (!stats) {
    return null;
  }
  return (
    <section className="stats">
      <Metric label="Cache hits" value={stats.hits} />
      <Metric label="Claude calls" value={stats.misses} />
      <Metric label="Hit rate" value={hitRate(stats)} />
      <Metric label="Cached answers" value={stats.entries} />
      <Metric label="Similarity threshold" value={`${Math.round(stats.threshold * 100)}%`} />
    </section>
  );
}
