import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';
import { TestType, TestTypeStats, TEST_TYPES } from '../../types/metrics';

interface TypeBarChartProps {
  byType: Record<TestType, TestTypeStats>;
}

export default function TypeBarChart({ byType }: TypeBarChartProps) {
  const data = TEST_TYPES.map((type) => ({
    name: type,
    passing: byType[type]?.passing || 0,
    failing: byType[type]?.failing || 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" fontSize={12} />
        <YAxis fontSize={12} />
        <Tooltip />
        <Legend />
        <Bar dataKey="passing" stackId="a" fill="#22c55e" name="Passing" />
        <Bar dataKey="failing" stackId="a" fill="#ef4444" name="Failing" />
      </BarChart>
    </ResponsiveContainer>
  );
}
