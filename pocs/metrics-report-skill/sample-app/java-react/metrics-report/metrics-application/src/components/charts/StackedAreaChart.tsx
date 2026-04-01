import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { MetricsReport, TEST_TYPES } from '../../types/metrics';

interface StackedAreaChartProps {
  history: MetricsReport[];
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

export default function StackedAreaChart({ history }: StackedAreaChartProps) {
  const data = useMemo(() => {
    return history.map((report) => {
      const date = new Date(report.timestamp);
      const label = `${date.getMonth() + 1}/${date.getDate()}`;
      const entry: Record<string, string | number> = { date: label };
      TEST_TYPES.forEach((type) => {
        entry[type] = report.tests.byType[type]?.total || 0;
      });
      return entry;
    });
  }, [history]);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" fontSize={12} />
        <YAxis fontSize={12} />
        <Tooltip />
        <Legend />
        {TEST_TYPES.map((type) => (
          <Area
            key={type}
            type="monotone"
            dataKey={type}
            stackId="1"
            stroke={TYPE_COLORS[type]}
            fill={TYPE_COLORS[type]}
            fillOpacity={0.6}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
