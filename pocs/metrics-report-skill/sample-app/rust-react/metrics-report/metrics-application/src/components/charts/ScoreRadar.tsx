import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';
import { ScoreBreakdown } from '../../types/metrics';

interface ScoreRadarProps {
  breakdown: ScoreBreakdown;
}

export default function ScoreRadar({ breakdown }: ScoreRadarProps) {
  const data = [
    { subject: 'Coverage Breadth', value: (breakdown.coverageBreadth / 3) * 100 },
    { subject: 'Type Diversity', value: (breakdown.typeDiversity / 2) * 100 },
    { subject: 'Pass Rate', value: (breakdown.passRate / 2) * 100 },
    { subject: 'Test Quality', value: (breakdown.testQuality / 2) * 100 },
    { subject: 'Code:Test Ratio', value: (breakdown.codeToTestRatio / 1) * 100 },
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
        <PolarGrid />
        <PolarAngleAxis dataKey="subject" fontSize={11} />
        <Radar
          name="Score"
          dataKey="value"
          stroke="#3b82f6"
          fill="#3b82f6"
          fillOpacity={0.3}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
