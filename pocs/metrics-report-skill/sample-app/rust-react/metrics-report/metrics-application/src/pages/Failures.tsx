import { MetricsReport, TEST_TYPES } from '../types/metrics';
import FailureList from '../components/FailureList';

interface Props {
  data: MetricsReport;
  searchQuery: string;
  history: MetricsReport[];
}

export default function Failures({ data, searchQuery, history }: Props) {
  const totalFailures = data.failures.details.length;

  if (totalFailures === 0) {
    return (
      <div className="failures-page">
        <div className="success-message">All tests passing</div>
      </div>
    );
  }

  return (
    <div className="failures-page">
      <div className="failure-count-large">{totalFailures}</div>

      <div className="summary-row">
        {TEST_TYPES.map((type) => {
          const count = data.failures.byType[type];
          if (!count) return null;
          return (
            <span key={type} className={`badge type-${type}`}>
              {type}: {count}
            </span>
          );
        })}
      </div>

      <FailureList
        failures={data.failures.details}
        searchQuery={searchQuery}
        history={history}
      />
    </div>
  );
}
