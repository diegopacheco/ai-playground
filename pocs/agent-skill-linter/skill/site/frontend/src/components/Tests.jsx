import React, { useState } from 'react';
import { formatMs } from '../lib/format.js';

function Histogram({ methods }) {
  const buckets = [
    { label: '<10ms', test: ms => ms < 10 },
    { label: '10-100ms', test: ms => ms >= 10 && ms < 100 },
    { label: '0.1-1s', test: ms => ms >= 100 && ms < 1000 },
    { label: '1-5s', test: ms => ms >= 1000 && ms < 5000 },
    { label: '>=5s', test: ms => ms >= 5000 }
  ].map(b => ({ label: b.label, count: methods.filter(m => b.test(m.durationMs)).length }));
  const max = Math.max(1, ...buckets.map(b => b.count));
  return (
    <div className="histogram">
      {buckets.map(b => (
        <div className="hbar" key={b.label}>
          <div className="hbar-track">
            <div
              className="hbar-fill"
              style={{ height: (b.count / max) * 100 + '%', background: b.label === '>=5s' ? '#dc2626' : '#2563eb' }}
            />
          </div>
          <div className="hbar-count">{b.count}</div>
          <div className="hbar-label">{b.label}</div>
        </div>
      ))}
    </div>
  );
}

export default function Tests({ report, query }) {
  const [slowOnly, setSlowOnly] = useState(false);
  const q = query.trim().toLowerCase();
  let methods = report.tests.methods;
  if (slowOnly) methods = methods.filter(m => m.slow);
  if (q) methods = methods.filter(m => (m.name + ' ' + m.classname + ' ' + (m.module || '')).toLowerCase().includes(q));

  return (
    <div className="tests">
      <div className="tests-head">
        <div className="stat-row">
          <span><strong>{report.tests.total}</strong> tests</span>
          <span><strong>{report.tests.passed}</strong> passed</span>
          <span className={report.tests.failed ? 'danger' : ''}><strong>{report.tests.failed}</strong> failed</span>
          <span className={report.tests.slow ? 'warn' : ''}><strong>{report.tests.slow}</strong> slow (≥{report.tests.slowThresholdSeconds}s)</span>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={slowOnly} onChange={e => setSlowOnly(e.target.checked)} />
          slow only
        </label>
      </div>

      <Histogram methods={report.tests.methods} />

      <table className="grid-table">
        <thead>
          <tr><th>Test</th><th>Location</th><th className="num">Duration</th><th>Status</th></tr>
        </thead>
        <tbody>
          {methods.map((m, i) => (
            <tr key={i} className={m.slow ? 'row-slow' : ''}>
              <td className="mono">{m.name}</td>
              <td className="muted">{m.classname || m.module}</td>
              <td className="num mono">{formatMs(m.durationMs)}</td>
              <td>
                <span className={'badge ' + m.status}>{m.status}</span>
                {m.slow && <span className="badge slow">slow</span>}
              </td>
            </tr>
          ))}
          {methods.length === 0 && <tr><td colSpan="4" className="muted">no matching tests</td></tr>}
        </tbody>
      </table>
    </div>
  );
}
