import React from 'react';
import { TestType, QualityEvaluation } from '../types/metrics';

interface QualityCardProps {
  type: TestType;
  evaluation: QualityEvaluation;
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
  fuzzy: '#a855f7',
  propertybased: '#0ea5e9',
};

const RATING_COLORS: Record<string, { bg: string; text: string }> = {
  poor: { bg: '#fef2f2', text: '#ef4444' },
  fair: { bg: '#fffbeb', text: '#f59e0b' },
  good: { bg: '#f0fdf4', text: '#22c55e' },
  excellent: { bg: '#eff6ff', text: '#3b82f6' },
};

export default function QualityCard({ type, evaluation }: QualityCardProps) {
  const ratingStyle = RATING_COLORS[evaluation.rating] || RATING_COLORS.fair;

  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: 12,
        padding: 16,
        background: '#fff',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span
          style={{
            background: TYPE_COLORS[type],
            color: '#fff',
            padding: '4px 12px',
            borderRadius: 12,
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          {type}
        </span>
        <span
          style={{
            background: ratingStyle.bg,
            color: ratingStyle.text,
            padding: '4px 12px',
            borderRadius: 12,
            fontSize: 13,
            fontWeight: 600,
            textTransform: 'uppercase',
          }}
        >
          {evaluation.rating}
        </span>
      </div>
      <p style={{ margin: 0, fontSize: 14, color: '#4b5563', lineHeight: 1.5 }}>
        {evaluation.justification}
      </p>
    </div>
  );
}
