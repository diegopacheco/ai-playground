import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';
import { MetricsReport } from '../../types/metrics';

interface TrendLineChartProps {
  history: MetricsReport[];
  metric: 'total' | 'passRate' | 'score' | 'coverage';
}

export default function TrendLineChart({ history, metric }: TrendLineChartProps) {
  const data = useMemo(() => {
    return history.map((report) => {
      const date = new Date(report.timestamp);
      const label = `${date.getMonth() + 1}/${date.getDate()}`;
      let value = 0;

      switch (metric) {
        case 'total':
          value = report.tests.total;
          break;
        case 'passRate':
          value = report.tests.total > 0 ? Math.round((report.tests.passing / report.tests.total) * 100) : 0;
          break;
        case 'score':
          value = report.score.total;
          break;
        case 'coverage': {
          const backend = report.coverage?.backend || [];
          const frontend = report.coverage?.frontend || [];
          const allFiles = [...backend, ...frontend];
          if (allFiles.length === 0) {
            value = 0;
            break;
          }
          let sum = 0;
          let count = 0;
          allFiles.forEach((f) => {
            if (f.coverage) {
              Object.values(f.coverage).forEach((c: any) => {
                if (c && c.tool !== null && c.tool !== undefined) {
                  sum += c.tool;
                  count++;
                }
              });
            }
          });
          value = count > 0 ? Math.round(sum / count) : 0;
          break;
        }
      }

      return { date: label, value };
    });
  }, [history, metric]);

  const metricLabels: Record<string, string> = {
    total: 'Total Tests',
    passRate: 'Pass Rate (%)',
    score: 'Score',
    coverage: 'Avg Coverage (%)',
  };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" fontSize={12} />
        <YAxis fontSize={12} />
        <Tooltip
          formatter={(val: number) => [val, metricLabels[metric]]}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={{ r: 4 }}
          name={metricLabels[metric]}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
