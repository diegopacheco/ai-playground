import React, { useState, useMemo } from 'react';
import { TestFile } from '../types/metrics';

interface TestTableProps {
  files: TestFile[];
  searchQuery: string;
}

type SortKey = 'path' | 'type' | 'testCount' | 'passing' | 'failing' | 'author' | 'status';
type SortDir = 'asc' | 'desc';

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

export default function TestTable({ files, searchQuery }: TestTableProps) {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [sortKey, setSortKey] = useState<SortKey>('path');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const filtered = useMemo(() => {
    if (!searchQuery) return files;
    const q = searchQuery.toLowerCase();
    return files.filter(
      (f) =>
        f.path.toLowerCase().includes(q) ||
        f.type.toLowerCase().includes(q) ||
        f.author.toLowerCase().includes(q) ||
        f.tests.some((t) => t.name.toLowerCase().includes(q))
    );
  }, [files, searchQuery]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'path': cmp = a.path.localeCompare(b.path); break;
        case 'type': cmp = a.type.localeCompare(b.type); break;
        case 'testCount': cmp = a.testCount - b.testCount; break;
        case 'passing': cmp = a.passing - b.passing; break;
        case 'failing': cmp = a.failing - b.failing; break;
        case 'author': cmp = a.author.localeCompare(b.author); break;
        case 'status': cmp = (a.failing > 0 ? 1 : 0) - (b.failing > 0 ? 1 : 0); break;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [filtered, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const toggleRow = (path: string) => {
    const next = new Set(expandedRows);
    if (next.has(path)) {
      next.delete(path);
    } else {
      next.add(path);
    }
    setExpandedRows(next);
  };

  const headerStyle: React.CSSProperties = {
    cursor: 'pointer',
    userSelect: 'none',
    padding: '8px 12px',
    textAlign: 'left',
    fontWeight: 600,
    borderBottom: '2px solid #e5e7eb',
  };

  const cellStyle: React.CSSProperties = {
    padding: '8px 12px',
    borderBottom: '1px solid #f3f4f6',
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
        <thead>
          <tr>
            <th style={headerStyle} onClick={() => handleSort('path')}>File {sortKey === 'path' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
            <th style={headerStyle} onClick={() => handleSort('type')}>Type {sortKey === 'type' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
            <th style={headerStyle} onClick={() => handleSort('testCount')}>Tests {sortKey === 'testCount' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
            <th style={headerStyle} onClick={() => handleSort('passing')}>Passing {sortKey === 'passing' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
            <th style={headerStyle} onClick={() => handleSort('failing')}>Failing {sortKey === 'failing' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
            <th style={headerStyle} onClick={() => handleSort('author')}>Author {sortKey === 'author' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
            <th style={headerStyle} onClick={() => handleSort('status')}>Status {sortKey === 'status' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((file) => (
            <React.Fragment key={file.path}>
              <tr
                onClick={() => toggleRow(file.path)}
                style={{ cursor: 'pointer', background: expandedRows.has(file.path) ? '#f9fafb' : 'transparent' }}
              >
                <td style={cellStyle}>
                  <a
                    href={file.githubUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    style={{ color: '#3b82f6', textDecoration: 'none' }}
                  >
                    {file.path}
                  </a>
                </td>
                <td style={cellStyle}>
                  <span
                    style={{
                      background: TYPE_COLORS[file.type] || '#6b7280',
                      color: '#fff',
                      padding: '2px 8px',
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                  >
                    {file.type}
                  </span>
                </td>
                <td style={cellStyle}>{file.testCount}</td>
                <td style={cellStyle}>{file.passing}</td>
                <td style={cellStyle}>{file.failing}</td>
                <td style={cellStyle}>{file.author}</td>
                <td style={cellStyle}>
                  <span
                    style={{
                      background: file.failing > 0 ? '#fef2f2' : '#f0fdf4',
                      color: file.failing > 0 ? '#ef4444' : '#22c55e',
                      padding: '2px 8px',
                      borderRadius: 12,
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    {file.failing > 0 ? 'FAIL' : 'PASS'}
                  </span>
                </td>
              </tr>
              {expandedRows.has(file.path) && (
                <tr>
                  <td colSpan={7} style={{ padding: '0 12px 12px 32px', background: '#f9fafb' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>Test Name</th>
                          <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>Status</th>
                          <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>Duration</th>
                          <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>Error</th>
                        </tr>
                      </thead>
                      <tbody>
                        {file.tests.map((test) => (
                          <tr key={test.name + test.line}>
                            <td style={{ padding: '4px 8px' }}>
                              <a
                                href={test.githubUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{ color: '#3b82f6', textDecoration: 'none' }}
                              >
                                {test.name}
                              </a>
                            </td>
                            <td style={{ padding: '4px 8px' }}>
                              <span style={{ color: test.status === 'pass' ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
                                {test.status.toUpperCase()}
                              </span>
                            </td>
                            <td style={{ padding: '4px 8px' }}>{test.duration}ms</td>
                            <td style={{ padding: '4px 8px', color: '#ef4444', fontSize: 12 }}>{test.error || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
