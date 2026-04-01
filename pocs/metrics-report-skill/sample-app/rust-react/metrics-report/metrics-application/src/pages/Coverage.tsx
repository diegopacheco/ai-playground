import { useState } from 'react';
import { MetricsReport } from '../types/metrics';
import CoverageTable from '../components/CoverageTable';

interface Props {
  data: MetricsReport;
  searchQuery: string;
}

export default function Coverage({ data, searchQuery }: Props) {
  const [activeTab, setActiveTab] = useState<'backend' | 'frontend'>('backend');

  const files = data.coverage[activeTab];

  const totalFiles = files.length;
  const filesWithToolCoverage = files.filter((f) =>
    Object.values(f.coverage).some((c) => c && c.tool !== null)
  ).length;
  const filesWithLlmMapping = files.filter((f) =>
    Object.values(f.coverage).some((c) => c && c.llm)
  ).length;

  return (
    <div className="coverage-page">
      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === 'backend' ? 'active' : ''}`}
          onClick={() => setActiveTab('backend')}
        >
          Backend
        </button>
        <button
          className={`tab-btn ${activeTab === 'frontend' ? 'active' : ''}`}
          onClick={() => setActiveTab('frontend')}
        >
          Frontend
        </button>
      </div>

      <div className="summary-row">
        <span className="badge badge-info">Total Files: {totalFiles}</span>
        <span className="badge badge-info">Tool Coverage: {filesWithToolCoverage}</span>
        <span className="badge badge-info">LLM Mapping: {filesWithLlmMapping}</span>
      </div>

      <CoverageTable files={files} searchQuery={searchQuery} layer={activeTab} />
    </div>
  );
}
