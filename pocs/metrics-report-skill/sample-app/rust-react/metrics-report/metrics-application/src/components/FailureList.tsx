import React, { useState, useMemo } from 'react';
import { FailureDetail, TestType, MetricsReport } from '../types/metrics';

interface FailureListProps {
  failures: FailureDetail[];
  searchQuery: string;
  history?: MetricsReport[];
}

const TYPE_COLORS: Record<string, string> = {
  unit: '#3b82f6',
  integration: '#8b5cf6',
  contract: '#f59e0b',
  e2e: '#10b981',
  css: '#ec4899',
  stress: '#ef4444',
  chaos: '#f97316',
  mutation: '#6366f1',
  observability: '#14b8a6',
};

export default function FailureList({ failures, searchQuery, history }: FailureListProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const frequencyMap = useMemo(() => {
    if (!history || history.length === 0) return new Map<string, number>();
    const map = new Map<string, number>();
    history.forEach((report) => {
      report.failures.details.forEach((f) => {
        const key = `${f.test}::${f.file}`;
        map.set(key, (map.get(key) || 0) + 1);
      });
    });
    return map;
  }, [history]);

  const filtered = useMemo(() => {
    if (!searchQuery) return failures;
    const q = searchQuery.toLowerCase();
    return failures.filter(
      (f) =>
        f.test.toLowerCase().includes(q) ||
        f.file.toLowerCase().includes(q) ||
        f.error.toLowerCase().includes(q)
    );
  }, [failures, searchQuery]);

  const grouped = useMemo(() => {
    const groups: Record<string, FailureDetail[]> = {};
    filtered.forEach((f) => {
      if (!groups[f.type]) groups[f.type] = [];
      groups[f.type].push(f);
    });
    return groups;
  }, [filtered]);

  const toggleExpand = (key: string) => {
    const next = new Set(expandedItems);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    setExpandedItems(next);
  };

  return (
    <div>
      {Object.entries(grouped).map(([type, items]) => (
        <div key={type} style={{ marginBottom: 24 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 12,
              padding: '8px 12px',
              background: '#f9fafb',
              borderRadius: 8,
            }}
          >
            <span
              style={{
                background: TYPE_COLORS[type] || '#6b7280',
                color: '#fff',
                padding: '2px 10px',
                borderRadius: 12,
                fontSize: 13,
                fontWeight: 600,
              }}
            >
              {type}
            </span>
            <span style={{ fontSize: 14, color: '#6b7280' }}>{items.length} failure{items.length !== 1 ? 's' : ''}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {items.map((failure, idx) => {
              const itemKey = `${type}-${failure.test}-${failure.file}-${idx}`;
              const isExpanded = expandedItems.has(itemKey);
              const freq = frequencyMap.get(`${failure.test}::${failure.file}`);
              return (
                <div
                  key={itemKey}
                  style={{
                    border: '1px solid #fecaca',
                    borderRadius: 8,
                    padding: 12,
                    background: '#fff',
                  }}
                >
                  <div
                    style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', cursor: failure.stackTrace ? 'pointer' : 'default' }}
                    onClick={() => failure.stackTrace && toggleExpand(itemKey)}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{failure.test}</div>
                      <div style={{ fontSize: 13, marginBottom: 4 }}>
                        <a
                          href={failure.githubUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: '#3b82f6', textDecoration: 'none' }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          {failure.file}:{failure.line}
                        </a>
                      </div>
                      <div style={{ fontSize: 13, color: '#ef4444' }}>{failure.error}</div>
                    </div>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
                      {(failure.failureCount || (freq && freq > 1)) && (
                        <span
                          style={{
                            background: '#fef2f2',
                            color: '#ef4444',
                            padding: '2px 8px',
                            borderRadius: 12,
                            fontSize: 11,
                            fontWeight: 600,
                          }}
                        >
                          Failed {failure.failureCount || freq} times
                        </span>
                      )}
                      {failure.stackTrace && (
                        <span style={{ fontSize: 12, color: '#9ca3af' }}>{isExpanded ? '[-]' : '[+]'}</span>
                      )}
                    </div>
                  </div>
                  {isExpanded && failure.stackTrace && (
                    <pre
                      style={{
                        marginTop: 8,
                        padding: 12,
                        background: '#1f2937',
                        color: '#f9fafb',
                        borderRadius: 6,
                        fontSize: 12,
                        overflow: 'auto',
                        maxHeight: 300,
                      }}
                    >
                      {failure.stackTrace}
                    </pre>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
      {Object.keys(grouped).length === 0 && (
        <div style={{ textAlign: 'center', padding: 32, color: '#9ca3af' }}>No failures found</div>
      )}
    </div>
  );
}
