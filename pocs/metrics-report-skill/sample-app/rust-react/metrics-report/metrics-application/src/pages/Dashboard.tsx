import React from 'react';
import { MetricsReport, TEST_TYPES, TestType } from '../types/metrics';
import ScoreGauge from '../components/ScoreGauge';
import PassFailDonut from '../components/charts/PassFailDonut';
import TypeBarChart from '../components/charts/TypeBarChart';

interface Props {
  data: MetricsReport;
}

const TYPE_LABELS: Record<TestType, string> = {
  unit: 'Unit',
  integration: 'Integration',
  contract: 'Contract',
  e2e: 'E2E',
  css: 'CSS',
  stress: 'Stress',
  chaos: 'Chaos',
  mutation: 'Mutation',
  observability: 'Observability',
  fuzzy: 'Fuzzy',
  propertybased: 'Property-Based',
};

function isFrontendFile(path: string): boolean {
  return /\.(tsx?|jsx?|css|scss|vue|svelte)$/.test(path) &&
    (/frontend|src\/__tests__|src\/components|src\/pages|\.spec\.(ts|js)x?$|\.test\.(ts|js)x?$/.test(path) ||
    !/\.java$|\.py$|\.rs$|\.go$/.test(path));
}

export default function Dashboard({ data }: Props) {
  const passRate = data.tests.total > 0
    ? ((data.tests.passing / data.tests.total) * 100).toFixed(1)
    : '0.0';

  const qualityEntries = TEST_TYPES
    .filter((t) => data.quality?.byType?.[t])
    .map((t) => ({ type: t, rating: data.quality.byType[t]!.rating }));

  const typeStats = TEST_TYPES.map((t) => {
    const typeData = data.tests.byType[t];
    let backendTotal = 0;
    let frontendTotal = 0;
    if (typeData?.files) {
      typeData.files.forEach((f) => {
        if (isFrontendFile(f.path)) {
          frontendTotal += f.testCount;
        } else {
          backendTotal += f.testCount;
        }
      });
    }
    return {
      type: t,
      label: TYPE_LABELS[t],
      total: typeData?.total || 0,
      backend: backendTotal,
      frontend: frontendTotal,
    };
  });

  return (
    <div className="dashboard-page">
      <div className="dashboard-grid">
        <div className="score-gauge-wrapper">
          <ScoreGauge score={data.score.total} breakdown={data.score.breakdown} />
        </div>
        <div className="stat-cards">
          <div className="stat-card">
            <div className="stat-card-label">Total Files</div>
            <div className="stat-card-value">{data.files.total}</div>
            <div className="stat-card-detail">
              Backend: {data.files.backend} / Frontend: {data.files.frontend}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">Test Files</div>
            <div className="stat-card-value">{data.files.testFiles}</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">Total Tests</div>
            <div className="stat-card-value">{data.tests.total}</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">Pass Rate</div>
            <div className="stat-card-value">{passRate}%</div>
          </div>
        </div>
      </div>

      <div className="type-cards-section">
        <div className="type-cards-grid">
          {typeStats.map(({ type, label, total, backend, frontend }) => (
            <div key={type} className="type-card">
              <div className="type-card-label">{label}</div>
              <div className="type-card-value">{total}</div>
              <div className="type-card-split">
                <span className="type-card-back">BE: {backend}</span>
                <span className="type-card-front">FE: {frontend}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <PassFailDonut passing={data.tests.passing} failing={data.tests.failing} />
        </div>
        <div className="chart-card">
          <TypeBarChart byType={data.tests.byType} />
        </div>
      </div>

      <div className="stacks-row">
        {data.stacks.map((stack) => (
          <span key={stack} className="stack-badge">{stack}</span>
        ))}
      </div>

      <div className="timestamp-row">
        Report generated: {new Date(data.timestamp).toLocaleString()}
      </div>

      {qualityEntries.length > 0 && (
        <div className="quality-summary">
          {qualityEntries.map(({ type, rating }) => (
            <span key={type} className={`quality-badge quality-badge-${rating}`}>
              {TYPE_LABELS[type]}: {rating}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
