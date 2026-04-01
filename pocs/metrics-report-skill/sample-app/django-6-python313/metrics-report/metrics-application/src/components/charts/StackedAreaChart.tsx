import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  LabelList,
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
  fuzzy: '#a855f7',
  propertybased: '#0ea5e9',
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

  const activeTypes = useMemo(() => {
    return TEST_TYPES.filter((type) =>
      data.some((d) => (d[type] as number) > 0)
    );
  }, [data]);

  if (data.length <= 1) {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} barSize={80}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" fontSize={12} />
          <YAxis fontSize={12} />
          <Tooltip />
          <Legend />
          {activeTypes.map((type) => (
            <Bar key={type} dataKey={type} stackId="1" fill={TYPE_COLORS[type]} radius={[6, 6, 0, 0]}>
              {activeTypes.indexOf(type) === activeTypes.length - 1 && (
                <LabelList dataKey={type} position="top" fontSize={14} fontWeight={600} fill="#3b82f6" />
              )}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

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
