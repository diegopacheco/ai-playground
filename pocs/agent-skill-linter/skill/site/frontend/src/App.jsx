import React, { useEffect, useState } from 'react';
import { fetchReport, fetchHistory } from './api.js';
import { scoreColor } from './lib/format.js';
import Dashboard from './components/Dashboard.jsx';
import Tests from './components/Tests.jsx';
import Charts from './components/Charts.jsx';
import Rules from './components/Rules.jsx';
import CodeViewer from './components/CodeViewer.jsx';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'tests', label: 'Tests' },
  { id: 'charts', label: 'Charts' },
  { id: 'rules', label: 'Rules' },
  { id: 'code', label: 'Code' }
];

export default function App() {
  const [report, setReport] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('dashboard');
  const [query, setQuery] = useState('');

  useEffect(() => {
    fetchReport().then(setReport).catch(e => setError(e.message));
    fetchHistory().then(setHistory).catch(() => setHistory([]));
  }, []);

  if (error) {
    return <div className="state">Could not load report: {error}. Run <code>/lint</code> first.</div>;
  }
  if (!report) {
    return <div className="state">Loading report…</div>;
  }

  const overall = report.scores.overall;
  const target = report.meta.target.split('/').pop();

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="dot" style={{ background: scoreColor(overall) }} />
          <div>
            <div className="title">Code Lint Report</div>
            <div className="target">{target} · {report.languages.join(', ')}</div>
          </div>
        </div>
        <div className="overall-pill" style={{ borderColor: scoreColor(overall) }}>
          <span style={{ color: scoreColor(overall) }}>{overall}</span>
          <small>overall</small>
        </div>
        <input
          className="search"
          placeholder="Search tests, rules, files…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
      </header>

      <nav className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={tab === t.id ? 'tab active' : 'tab'}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="content">
        {tab === 'dashboard' && <Dashboard report={report} history={history} />}
        {tab === 'tests' && <Tests report={report} query={query} />}
        {tab === 'charts' && <Charts report={report} history={history} />}
        {tab === 'rules' && <Rules report={report} query={query} />}
        {tab === 'code' && <CodeViewer report={report} query={query} />}
      </main>
    </div>
  );
}
