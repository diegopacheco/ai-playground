import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../lib/api";
import { Card, Stat } from "../components/Card";
import { Loading, ErrorView, Empty } from "../components/Status";

const bucketColors = ["#3fb950", "#d29922", "#db6d28", "#f85149"];

function shortWeek(week: string): string {
  return week.slice(5);
}

export function InsightsTab() {
  const insights = useQuery({ queryKey: ["insights"], queryFn: api.insights });

  if (insights.isLoading) {
    return <Loading what="insights" />;
  }
  if (insights.isError) {
    return <ErrorView error={insights.error} />;
  }
  const data = insights.data;
  if (!data) {
    return <Empty message="No data yet. Add repos, then press Sync." />;
  }

  return (
    <div className="stack">
      <div className="stat-grid">
        <Stat label="Median time to merge (h)" value={data.medianTimeToMergeHours} tone="purple" />
        <Stat label="Average time to merge (h)" value={data.avgTimeToMergeHours} tone="muted" />
      </div>

      <Card title="PR throughput — merged per week">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.prThroughput}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e1e4e8" />
            <XAxis dataKey="week" tickFormatter={shortWeek} stroke="#656d76" fontSize={12} />
            <YAxis allowDecimals={false} stroke="#656d76" fontSize={12} />
            <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d0d7de", borderRadius: 8 }} />
            <Bar dataKey="count" fill="#7c3aed" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Issue flow — opened vs closed per week">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.issueFlow}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e1e4e8" />
            <XAxis dataKey="week" tickFormatter={shortWeek} stroke="#656d76" fontSize={12} />
            <YAxis allowDecimals={false} stroke="#656d76" fontSize={12} />
            <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d0d7de", borderRadius: 8 }} />
            <Line type="monotone" dataKey="opened" stroke="#1a7f37" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="closed" stroke="#cf222e" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <div className="chart-grid">
        <Card title="Open PR age">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.stalePrBuckets}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e1e4e8" />
              <XAxis dataKey="label" stroke="#656d76" fontSize={12} />
              <YAxis allowDecimals={false} stroke="#656d76" fontSize={12} />
              <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d0d7de", borderRadius: 8 }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {data.stalePrBuckets.map((_, index) => (
                  <Cell key={index} fill={bucketColors[index % bucketColors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Top contributors">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.contributors} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#e1e4e8" />
              <XAxis type="number" allowDecimals={false} stroke="#656d76" fontSize={12} />
              <YAxis type="category" dataKey="login" width={110} stroke="#656d76" fontSize={12} />
              <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #d0d7de", borderRadius: 8 }} />
              <Bar dataKey="total" fill="#0969da" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
