import { MetricsReport } from '../types/metrics';
import TrendLineChart from '../components/charts/TrendLineChart';
import StackedAreaChart from '../components/charts/StackedAreaChart';

interface Props {
  history: MetricsReport[];
}

export default function Trends({ history }: Props) {
  if (history.length === 0) {
    return (
      <div className="trends-page">
        <div className="empty-state">
          No history data yet. Run the metrics report multiple times to build trend data.
        </div>
      </div>
    );
  }

  return (
    <div className="trends-page">
      <div className="trends-grid">
        <div className="chart-card">
          <h3 className="chart-title">Total Tests Over Time</h3>
          <TrendLineChart history={history} metric="total" />
        </div>
        <div className="chart-card">
          <h3 className="chart-title">Pass Rate Over Time</h3>
          <TrendLineChart history={history} metric="passRate" />
        </div>
      </div>

      <div className="trends-grid">
        <div className="chart-card">
          <h3 className="chart-title">Score Over Time</h3>
          <TrendLineChart history={history} metric="score" />
        </div>
        <div className="chart-card">
          <h3 className="chart-title">Coverage Over Time</h3>
          <TrendLineChart history={history} metric="coverage" />
        </div>
      </div>

      <div className="trends-grid full-width">
        <div className="chart-card">
          <h3 className="chart-title">Tests by Type Over Time</h3>
          <StackedAreaChart history={history} />
        </div>
      </div>
    </div>
  );
}
