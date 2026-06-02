import React from 'react';
import { scoreColor, CATEGORY_LABELS } from '../lib/format.js';

function LineChart({ values, color = '#2563eb', height = 160 }) {
  const w = 520;
  const h = height;
  const pad = 24;
  if (values.length === 0) return <div className="muted">no history yet</div>;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 100);
  const span = max - min || 1;
  const step = values.length > 1 ? (w - pad * 2) / (values.length - 1) : 0;
  const pts = values.map((v, i) => {
    const x = pad + i * step;
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    return [x, y];
  });
  const line = pts.map(p => p.join(',')).join(' ');
  const area = `${pad},${h - pad} ${line} ${pad + (values.length - 1) * step},${h - pad}`;
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polyline points={area} fill={color} opacity="0.08" />
      <polyline points={line} fill="none" stroke={color} strokeWidth="2.5" />
      {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3" fill={color} />)}
    </svg>
  );
}

function BarChart({ data, height = 180 }) {
  const max = Math.max(1, ...data.map(d => d.value));
  return (
    <div className="barchart" style={{ height }}>
      {data.map(d => (
        <div className="bc-col" key={d.label}>
          <div className="bc-value">{d.value}</div>
          <div className="bc-bar" style={{ height: (d.value / max) * (height - 48) + 'px', background: d.color || '#2563eb' }} />
          <div className="bc-label">{d.label}</div>
        </div>
      ))}
    </div>
  );
}

export default function Charts({ report, history }) {
  const overalls = history.map(h => h.overall);
  const slows = history.map(h => h.tests.slow);

  const catData = Object.entries(report.scores.byCategory)
    .filter(([, v]) => v != null)
    .map(([k, v]) => ({ label: CATEGORY_LABELS[k] || k, value: v, color: scoreColor(v) }));

  const ccBuckets = [
    { label: '1-2', test: c => c <= 2 },
    { label: '3-5', test: c => c >= 3 && c <= 5 },
    { label: '6-10', test: c => c >= 6 && c <= 10 },
    { label: '11+', test: c => c >= 11 }
  ].map(b => ({
    label: b.label,
    value: report.complexity.functions.filter(f => b.test(f.cyclomatic)).length,
    color: b.label === '11+' ? '#dc2626' : '#6366f1'
  }));

  return (
    <div className="charts">
      <section className="card">
        <h3>Overall score over time</h3>
        <LineChart values={overalls} />
        <p className="muted">{overalls.length} run(s) recorded</p>
      </section>
      <section className="card">
        <h3>Slow tests over time</h3>
        <LineChart values={slows} color="#f59e0b" />
      </section>
      <section className="card">
        <h3>Category scores</h3>
        <BarChart data={catData} />
      </section>
      <section className="card">
        <h3>Cyclomatic complexity distribution</h3>
        <BarChart data={ccBuckets} />
      </section>
    </div>
  );
}
