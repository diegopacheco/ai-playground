import { MetricsReport } from '../types/metrics';
import AuthorLeaderboard from '../components/AuthorLeaderboard';

interface Props {
  data: MetricsReport;
  searchQuery: string;
}

export default function Authors({ data, searchQuery }: Props) {
  const authorCount = Object.keys(data.authors).length;

  return (
    <div className="authors-page">
      <div className="summary-row">
        <span className="badge badge-info">Authors: {authorCount}</span>
        <span className="badge badge-info">Total Tests: {data.tests.total}</span>
      </div>

      <AuthorLeaderboard authors={data.authors} searchQuery={searchQuery} />
    </div>
  );
}
