import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PassFailDonutProps {
  passing: number;
  failing: number;
}

export default function PassFailDonut({ passing, failing }: PassFailDonutProps) {
  const total = passing + failing;
  const passPercent = total > 0 ? Math.round((passing / total) * 100) : 0;

  const data = [
    { name: 'Passing', value: passing },
    { name: 'Failing', value: failing },
  ];

  const COLORS = ['#22c55e', '#ef4444'];

  return (
    <div style={{ width: '100%', height: 300, position: 'relative' }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={90}
            dataKey="value"
            stroke="none"
          >
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index]} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
      <div
        style={{
          position: 'absolute',
          top: '45%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          pointerEvents: 'none',
        }}
      >
        <div style={{ fontSize: 28, fontWeight: 700 }}>{passPercent}%</div>
        <div style={{ fontSize: 12, color: '#6b7280' }}>pass rate</div>
      </div>
    </div>
  );
}
