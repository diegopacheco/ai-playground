import React from 'react';
import { scoreColor } from '../lib/format.js';

export default function Gauge({ value, size = 180, label = 'overall' }) {
  const stroke = 16;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, value));
  const offset = circumference * (1 - pct / 100);
  const color = scoreColor(value);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#eef1f5" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="49%" textAnchor="middle" fontSize={size * 0.28} fontWeight="700" fill="#0f172a">
        {Math.round(value)}
      </text>
      <text x="50%" y="65%" textAnchor="middle" fontSize={size * 0.085} fill="#64748b">
        {label}
      </text>
    </svg>
  );
}
