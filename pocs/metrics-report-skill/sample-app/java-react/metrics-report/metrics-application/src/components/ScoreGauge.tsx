import React from 'react';
import { ScoreBreakdown } from '../types/metrics';

interface ScoreGaugeProps {
  score: number;
  breakdown: ScoreBreakdown;
}

function getScoreColor(score: number): string {
  if (score <= 3) return '#ef4444';
  if (score <= 6) return '#f59e0b';
  if (score <= 8) return '#22c55e';
  return '#3b82f6';
}

export default function ScoreGauge({ score, breakdown }: ScoreGaugeProps) {
  const radius = 80;
  const strokeWidth = 12;
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 10) * circumference;
  const color = getScoreColor(score);

  const breakdownItems = [
    { label: 'Coverage Breadth', value: breakdown.coverageBreadth, max: 3 },
    { label: 'Type Diversity', value: breakdown.typeDiversity, max: 2 },
    { label: 'Pass Rate', value: breakdown.passRate, max: 2 },
    { label: 'Test Quality', value: breakdown.testQuality, max: 2 },
    { label: 'Code:Test Ratio', value: breakdown.codeToTestRatio, max: 1 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 24 }}>
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle
          cx="100"
          cy="100"
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
        />
        <circle
          cx="100"
          cy="100"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${progress} ${circumference - progress}`}
          strokeDashoffset={circumference * 0.25}
          strokeLinecap="round"
          transform="rotate(-90 100 100)"
        />
        <text
          x="100"
          y="100"
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="40"
          fontWeight="bold"
          fill={color}
        >
          {score.toFixed(1)}
        </text>
      </svg>
      <div style={{ width: '100%', maxWidth: 300 }}>
        {breakdownItems.map((item) => (
          <div key={item.label} style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
              <span>{item.label}</span>
              <span>{item.value}/{item.max}</span>
            </div>
            <div style={{ background: '#e5e7eb', borderRadius: 4, height: 6, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${(item.value / item.max) * 100}%`,
                  height: '100%',
                  background: color,
                  borderRadius: 4,
                  transition: 'width 0.3s',
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
