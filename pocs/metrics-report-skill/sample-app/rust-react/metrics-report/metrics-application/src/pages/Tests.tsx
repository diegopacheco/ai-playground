import { useState } from 'react';
import { MetricsReport, TEST_TYPES, TestType, TestFile } from '../types/metrics';
import TestTable from '../components/TestTable';

interface Props {
  data: MetricsReport;
  searchQuery: string;
}

export default function Tests({ data, searchQuery }: Props) {
  const [typeFilter, setTypeFilter] = useState<TestType | 'all'>('all');

  const allFiles: TestFile[] = TEST_TYPES.flatMap((type) => {
    const stats = data.tests.byType[type];
    if (!stats) return [];
    return stats.files;
  });

  const filteredFiles = typeFilter === 'all'
    ? allFiles
    : allFiles.filter((f) => f.type === typeFilter);

  return (
    <div className="tests-page">
      <div className="summary-row">
        <span className="badge badge-info">Total: {data.tests.total}</span>
        <span className="badge badge-pass">Passing: {data.tests.passing}</span>
        <span className="badge badge-fail">Failing: {data.tests.failing}</span>
      </div>

      <div className="filter-bar">
        <select
          className="filter-select"
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value as TestType | 'all')}
        >
          <option value="all">All Types</option>
          {TEST_TYPES.map((type) => {
            const stats = data.tests.byType[type];
            if (!stats || stats.total === 0) return null;
            return (
              <option key={type} value={type}>
                {type} ({stats.total})
              </option>
            );
          })}
        </select>
      </div>

      <TestTable files={filteredFiles} searchQuery={searchQuery} />
    </div>
  );
}
