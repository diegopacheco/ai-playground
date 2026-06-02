import React from 'react';
import Gauge from './Gauge.jsx';
import { scoreColor, CATEGORY_LABELS } from '../lib/format.js';

function Sparkline({ values }) {
  if (values.length < 2) return null;
  const w = 220;
  const h = 44;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / span) * (h - 8) - 4;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg width={w} height={h}>
      <polyline points={points} fill="none" stroke="#2563eb" strokeWidth="2" />
    </svg>
  );
}

export default function Dashboard({ report, history }) {
  const cats = report.scores.byCategory;
  const tiles = [
    { label: 'Build', value: report.build.status, color: report.build.status === 'pass' ? '#16a34a' : '#dc2626' },
    { label: 'Tests', value: report.tests.passed + '/' + report.tests.total, color: '#0f172a' },
    { label: 'Slow tests', value: report.tests.slow, color: report.tests.slow ? '#f59e0b' : '#16a34a' },
    { label: 'Max CC', value: report.complexity.maxCyclomatic, color: report.complexity.overThreshold ? '#f59e0b' : '#0f172a' },
    { label: 'Files', value: report.metrics.totalFiles, color: '#0f172a' },
    { label: 'Functions', value: report.metrics.totalFunctions, color: '#0f172a' }
  ];
  const overalls = history.map(h => h.overall);

  return (
    <div className="dashboard">
      <section className="card hero">
        <Gauge value={report.scores.overall} />
        <div className="hero-side">
          <h2>Quality score</h2>
          <p>Weighted across build, tests, complexity, and four semantic categories.</p>
          {overalls.length > 1 && (
            <div className="trend">
              <Sparkline values={overalls} />
              <span>over {overalls.length} runs</span>
            </div>
          )}
        </div>
      </section>

      <section className="cat-grid">
        {Object.entries(cats).filter(([, v]) => v != null).map(([key, value]) => (
          <div className="card cat" key={key}>
            <div className="cat-top">
              <span>{CATEGORY_LABELS[key] || key}</span>
              <strong style={{ color: scoreColor(value) }}>{value}</strong>
            </div>
            <div className="bar">
              <div className="bar-fill" style={{ width: value + '%', background: scoreColor(value) }} />
            </div>
          </div>
        ))}
      </section>

      <section className="tile-grid">
        {tiles.map(t => (
          <div className="card tile" key={t.label}>
            <div className="tile-value" style={{ color: t.color }}>{t.value}</div>
            <div className="tile-label">{t.label}</div>
          </div>
        ))}
      </section>
    </div>
  );
}
