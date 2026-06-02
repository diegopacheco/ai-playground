export type Series = { label: string; color: string; values: number[] };

type Props = {
  series: Series[];
  yMax?: number;
  yLabel?: string;
};

const W = 720;
const H = 280;
const PAD_L = 44;
const PAD_B = 28;
const PAD_T = 16;
const PAD_R = 12;

export function LineChart({ series, yMax = 100, yLabel = "%" }: Props) {
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const maxLen = Math.max(2, ...series.map((s) => s.values.length));

  const x = (i: number) => PAD_L + (maxLen <= 1 ? 0 : (i / (maxLen - 1)) * plotW);
  const y = (v: number) => PAD_T + plotH - (Math.min(v, yMax) / yMax) * plotH;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img">
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid" x1={PAD_L} y1={y(t)} x2={W - PAD_R} y2={y(t)} />
            <text className="axis" x={PAD_L - 8} y={y(t) + 4} textAnchor="end">
              {Math.round(t)}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} />
        <line className="axis-line" x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} />
        <text className="axis" x={W - PAD_R} y={PAD_T - 4} textAnchor="end">
          {yLabel}
        </text>
        {series.map((s) =>
          s.values.length >= 1 ? (
            <polyline
              key={s.label}
              className="series"
              fill="none"
              stroke={s.color}
              points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
            />
          ) : null,
        )}
        {series.map((s) =>
          s.values.map((v, i) => (
            <circle key={`${s.label}-${i}`} cx={x(i)} cy={y(v)} r={2.5} fill={s.color} />
          )),
        )}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.label} className="legend-item">
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
