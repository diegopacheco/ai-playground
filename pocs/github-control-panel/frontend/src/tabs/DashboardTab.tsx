import type { ColumnDef } from "@tanstack/react-table";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card, Stat } from "../components/Card";
import { DataTable } from "../components/DataTable";
import { Loading, ErrorView, Empty } from "../components/Status";
import type { Contributor, RepoSummary } from "../lib/types";

const repoColumns: ColumnDef<RepoSummary, any>[] = [
  { accessorKey: "fullName", header: "Repository" },
  { accessorKey: "openIssues", header: "Open issues" },
  { accessorKey: "openPrs", header: "Open PRs" },
  {
    id: "synced",
    header: "Last sync",
    accessorFn: (row) => row.lastSyncedAt,
    cell: ({ getValue }) => {
      const value = getValue() as string | null;
      return value ? new Date(value).toLocaleString() : "never";
    },
  },
];

const contributorColumns: ColumnDef<Contributor, any>[] = [
  { accessorKey: "login", header: "Contributor" },
  { accessorKey: "commits", header: "Commits" },
  { accessorKey: "prsOpened", header: "PRs" },
  { accessorKey: "issuesOpened", header: "Issues" },
  { accessorKey: "reviews", header: "Reviews" },
  { accessorKey: "total", header: "Total" },
];

export function DashboardTab() {
  const dashboard = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard });

  if (dashboard.isLoading) {
    return <Loading what="dashboard" />;
  }
  if (dashboard.isError) {
    return <ErrorView error={dashboard.error} />;
  }
  const data = dashboard.data;
  if (!data || data.repoCount === 0) {
    return <Empty message="No data yet. Add repos on the Repos tab, then press Sync." />;
  }

  return (
    <div className="stack">
      <div className="stat-grid">
        <Stat label="Repositories" value={data.repoCount} />
        <Stat label="Open issues" value={data.issues.open} tone="green" />
        <Stat label="Closed issues" value={data.issues.closed} tone="muted" />
        <Stat label="Open PRs" value={data.prs.open} tone="green" />
        <Stat label="Pending review" value={data.prs.pendingReview} tone="amber" />
        <Stat label="Draft PRs" value={data.prs.draft} tone="muted" />
        <Stat label="Merged PRs" value={data.prs.merged} tone="purple" />
        <Stat label="Closed PRs" value={data.prs.closed} tone="muted" />
      </div>

      <Card title="Per-repository breakdown">
        <DataTable data={data.repos} columns={repoColumns} empty="No repositories synced." />
      </Card>

      <Card title="Contributions across all repos">
        <DataTable data={data.contributors} columns={contributorColumns} empty="No contributions yet." />
      </Card>
    </div>
  );
}
