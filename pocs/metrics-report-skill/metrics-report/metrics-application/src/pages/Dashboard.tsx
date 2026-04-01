import React from 'react';
import { MetricsReport, TEST_TYPES } from '../types/metrics';
import ScoreGauge from '../components/ScoreGauge';
import PassFailDonut from '../components/charts/PassFailDonut';
import TypeBarChart from '../components/charts/TypeBarChart';

interface Props {
  data: MetricsReport;
}

export default function Dashboard({ data }: Props) {
  const passRate = data.tests.total > 0
    ? ((data.tests.passing / data.tests.total) * 100).toFixed(1)
    : '0.0';

  const qualityEntries = TEST_TYPES
    .filter((t) => data.quality?.byType?.[t])
    .map((t) => ({ type: t, rating: data.quality.byType[t]!.rating }));

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
              {type}: {rating}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
