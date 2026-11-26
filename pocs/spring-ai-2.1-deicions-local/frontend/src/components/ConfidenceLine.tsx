import { decimal } from "../format";

export type Zone = { readonly from: number; readonly to: number; readonly label: string; readonly tone: "good" | "warn" | "bad" };

type ConfidenceLineProps = { readonly value: number; readonly zones: readonly Zone[]; readonly caption: string };

export function ConfidenceLine({ value, zones, caption }: ConfidenceLineProps) {
  return (
    <figure className="line">
      <div className="line-track">
        {zones.map((z) => (
          <span key={z.label} className={`line-zone zone-${z.tone}`} style={{ left: `${z.from * 100}%`, width: `${(z.to - z.from) * 100}%` }}>
            {z.label}
          </span>
        ))}
        <span className="line-marker" style={{ left: `${value * 100}%` }}>
          <span className="line-marker-value">{decimal(value)}</span>
        </span>
      </div>
      <div className="line-axis">
        <span>0</span>
        {zones.slice(1).map((z) => (
          <span key={z.label} style={{ left: `${z.from * 100}%` }}>
            {decimal(z.from)}
          </span>
        ))}
        <span>1</span>
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
