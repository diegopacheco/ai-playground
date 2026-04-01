import React from 'react';

interface SearchBarProps {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}

export default function SearchBar({ value, onChange, placeholder }: SearchBarProps) {
  return (
    <div className="search-bar">
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ marginRight: 8, flexShrink: 0 }}
      >
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || 'Search...'}
        style={{
          border: 'none',
          outline: 'none',
          background: 'transparent',
          flex: 1,
          fontSize: '14px',
          color: 'inherit',
        }}
      />
      {value && (
        <button
          onClick={() => onChange('')}
          style={{
            border: 'none',
            background: 'transparent',
            cursor: 'pointer',
            fontSize: '16px',
            color: '#999',
            padding: '0 4px',
          }}
        >
          X
        </button>
      )}
    </div>
  );
}
