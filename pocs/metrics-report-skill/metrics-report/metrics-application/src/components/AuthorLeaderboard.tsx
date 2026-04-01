import React, { useMemo } from 'react';
import { AuthorStats, TEST_TYPES, TestType } from '../types/metrics';

interface AuthorLeaderboardProps {
  authors: Record<string, AuthorStats>;
  searchQuery: string;
}

const TYPE_COLORS: Record<TestType, string> = {
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

export default function AuthorLeaderboard({ authors, searchQuery }: AuthorLeaderboardProps) {
  const ranked = useMemo(() => {
    let entries = Object.entries(authors);
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      entries = entries.filter(([name]) => name.toLowerCase().includes(q));
    }
    return entries.sort((a, b) => b[1].total - a[1].total);
  }, [authors, searchQuery]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {ranked.map(([name, stats], idx) => {
        const maxTotal = ranked.length > 0 ? ranked[0][1].total : 1;
        return (
          <div
            key={name}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 16,
              padding: '12px 16px',
              background: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: 8,
            }}
          >
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                background: idx < 3 ? ['#f59e0b', '#9ca3af', '#cd7f32'][idx] : '#e5e7eb',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 700,
                fontSize: 14,
                color: idx < 3 ? '#fff' : '#6b7280',
                flexShrink: 0,
              }}
            >
              {idx + 1}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 14 }}>{name}</span>
                <span style={{ fontSize: 13, color: '#6b7280' }}>{stats.total} tests</span>
              </div>
              <div
                style={{
                  display: 'flex',
                  height: 8,
                  borderRadius: 4,
                  overflow: 'hidden',
                  background: '#f3f4f6',
                  width: `${(stats.total / maxTotal) * 100}%`,
                }}
              >
                {TEST_TYPES.map((type) => {
                  const count = stats[type as keyof AuthorStats] as number;
                  if (count <= 0) return null;
                  return (
                    <div
                      key={type}
                      style={{
                        width: `${(count / stats.total) * 100}%`,
                        background: TYPE_COLORS[type],
                        height: '100%',
                      }}
                      title={`${type}: ${count}`}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}
      {ranked.length === 0 && (
        <div style={{ textAlign: 'center', padding: 32, color: '#9ca3af' }}>No authors found</div>
      )}
    </div>
  );
}
