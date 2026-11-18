import type { Bar } from "./dbos.ts";

export function Timeline({ bars, total }: { bars: Bar[]; total: string }) {
  if (bars.length === 0) return <p className="muted">No step recorded yet.</p>;
  return (
    <div className="timeline">
      {bars.map(bar => (
        <div className="lane" key={bar.key}>
          <span className="lane-label" title={bar.label}>{bar.label}</span>
          <div className="track">
            <div className={`bar ${bar.kind}`} style={{ left: `${bar.left}%`, width: `${bar.width}%` }} />
          </div>
          <span className="lane-time">{bar.duration}</span>
        </div>
      ))}
      <div className="axis"><span>0s</span><span>{total}</span></div>
    </div>
  );
}
