import { MetricsReport, TEST_TYPES, TestType } from '../types/metrics';
import ScoreRadar from '../components/charts/ScoreRadar';
import QualityCard from '../components/QualityCard';

interface Props {
  data: MetricsReport;
}

export default function Quality({ data }: Props) {
  const evaluatedTypes: TestType[] = TEST_TYPES.filter((t) => data.quality.byType[t]);

  const passRate = data.tests.total > 0
    ? (data.tests.passing / data.tests.total) * 100
    : 0;

  const poorTypes = evaluatedTypes.filter((t) => data.quality.byType[t]!.rating === 'poor');

  const recommendations: string[] = [];

  if (data.score.total < 5) {
    recommendations.push('Critical: Your test suite needs significant improvement.');
  }

  poorTypes.forEach((t) => {
    recommendations.push(`Improve ${t} test quality`);
  });

  if (evaluatedTypes.length < 4) {
    recommendations.push('Add more test types for better coverage diversity');
  }

  if (passRate < 90) {
    recommendations.push('Fix failing tests to improve reliability');
  }

  return (
    <div className="quality-page">
      <div className="chart-card">
        <ScoreRadar breakdown={data.score.breakdown} />
      </div>

      <div className="quality-grid">
        {evaluatedTypes.map((type) => (
          <QualityCard
            key={type}
            type={type}
            evaluation={data.quality.byType[type]!}
          />
        ))}
      </div>

      {recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3 className="recommendations-title">Recommendations</h3>
          <ul className="recommendations-list">
            {recommendations.map((rec, i) => (
              <li key={i} className="recommendation-item">{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
