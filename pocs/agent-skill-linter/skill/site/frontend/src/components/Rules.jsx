import React from 'react';
import { statusColor, CATEGORY_LABELS } from '../lib/format.js';

export default function Rules({ report, query }) {
  const q = query.trim().toLowerCase();
  let rules = report.rules;
  if (q) {
    rules = rules.filter(r =>
      (r.id + ' ' + r.category + ' ' + (r.detail || '')).toLowerCase().includes(q));
  }

  const byCategory = {};
  for (const rule of rules) {
    (byCategory[rule.category] = byCategory[rule.category] || []).push(rule);
  }

  return (
    <div className="rules">
      {Object.entries(byCategory).map(([category, list]) => (
        <section className="card" key={category}>
          <h3>
            {CATEGORY_LABELS[category] || category}
            <span className="rule-count">
              {list.filter(r => r.status === 'pass').length}/{list.length} pass
            </span>
          </h3>
          {list.map((rule, i) => (
            <details key={i} className="rule" open={rule.status !== 'pass'}>
              <summary>
                <span className="badge" style={{ background: statusColor(rule.status) }}>{rule.status}</span>
                <span className="rule-id">{rule.id}</span>
                <span className="rule-type">{rule.type}</span>
                <span className="rule-detail">{rule.detail}</span>
              </summary>
              {rule.findings && rule.findings.length > 0 && (
                <ul className="findings">
                  {rule.findings.map((f, j) => (
                    <li key={j}>
                      <code>{f.where || f.test || ''}</code>
                      {f.message ? ' — ' + f.message : ''}
                      {f.cyclomatic != null ? ' (CC ' + f.cyclomatic + ')' : ''}
                      {f.durationMs != null ? ' (' + f.durationMs + 'ms)' : ''}
                      {f.loc != null ? ' (' + f.loc + ' lines)' : ''}
                    </li>
                  ))}
                </ul>
              )}
              {rule.samples && (
                <div className="samples">
                  <div className="sample bad">
                    <span className="sample-tag">avoid</span>
                    <pre>{rule.samples.bad}</pre>
                  </div>
                  <div className="sample good">
                    <span className="sample-tag">prefer</span>
                    <pre>{rule.samples.good}</pre>
                  </div>
                </div>
              )}
            </details>
          ))}
        </section>
      ))}
      {rules.length === 0 && <div className="muted">no matching rules</div>}
    </div>
  );
}
