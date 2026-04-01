import React, { useState, useMemo } from 'react';
import { FileCoverage, TEST_TYPES, TestType } from '../types/metrics';

interface CoverageTableProps {
  files: FileCoverage[];
  searchQuery: string;
  layer: 'backend' | 'frontend';
}

type SortDir = 'asc' | 'desc';

function getCoverageColor(pct: number): string {
  if (pct <= 30) return '#ef4444';
  if (pct <= 70) return '#f59e0b';
  return '#22c55e';
}

export default function CoverageTable({ files, searchQuery, layer }: CoverageTableProps) {
  const [sortKey, setSortKey] = useState<string>('file');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const filtered = useMemo(() => {
    const layerFiles = files.filter((f) => f.layer === layer);
    if (!searchQuery) return layerFiles;
    const q = searchQuery.toLowerCase();
    return layerFiles.filter((f) => f.file.toLowerCase().includes(q));
  }, [files, searchQuery, layer]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'file') {
        cmp = a.file.localeCompare(b.file);
      } else {
        const typeKey = sortKey as TestType;
        const aVal = a.coverage[typeKey]?.tool ?? -1;
        const bVal = b.coverage[typeKey]?.tool ?? -1;
        cmp = aVal - bVal;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [filtered, sortKey, sortDir]);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const headerStyle: React.CSSProperties = {
    cursor: 'pointer',
    userSelect: 'none',
    padding: '8px 6px',
    textAlign: 'left',
    fontWeight: 600,
    fontSize: 12,
    borderBottom: '2px solid #e5e7eb',
    whiteSpace: 'nowrap',
  };

  const cellStyle: React.CSSProperties = {
    padding: '6px',
    borderBottom: '1px solid #f3f4f6',
    fontSize: 13,
    textAlign: 'center',
  };

  const renderCell = (cov: { tool: number | null; llm: boolean } | undefined) => {
    if (!cov) {
      return <span style={{ color: '#d1d5db' }}>-</span>;
    }
    if (cov.tool !== null) {
      const pct = Math.round(cov.tool);
      return (
        <span style={{ color: getCoverageColor(pct), fontWeight: 600 }}>
          {pct}%
        </span>
      );
    }
    if (cov.llm) {
      return (
        <span
          style={{
            background: '#dbeafe',
            color: '#3b82f6',
            padding: '2px 6px',
            borderRadius: 8,
            fontSize: 11,
          }}
        >
          mapped
        </span>
      );
    }
    return <span style={{ color: '#d1d5db' }}>-</span>;
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ ...headerStyle, textAlign: 'left' }} onClick={() => handleSort('file')}>
              File {sortKey === 'file' ? (sortDir === 'asc' ? ' ^' : ' v') : ''}
            </th>
            {TEST_TYPES.map((type) => (
              <th key={type} style={headerStyle} onClick={() => handleSort(type)}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
                {sortKey === type ? (sortDir === 'asc' ? ' ^' : ' v') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((file) => (
            <tr key={file.file}>
              <td style={{ ...cellStyle, textAlign: 'left' }}>
                <a
                  href={file.githubUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: '#3b82f6', textDecoration: 'none', fontSize: 13 }}
                >
                  {file.file}
                </a>
              </td>
              {TEST_TYPES.map((type) => (
                <td key={type} style={cellStyle}>
                  {renderCell(file.coverage[type])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
