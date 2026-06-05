import type { ColumnDef } from "@tanstack/react-table";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Card } from "../components/Card";
import { DataTable } from "../components/DataTable";
import { Loading, ErrorView, Empty } from "../components/Status";
import type { ActionItem } from "../lib/types";

const columns: ColumnDef<ActionItem, any>[] = [
  { accessorKey: "repo", header: "Repository" },
  {
    id: "ref",
    header: "#",
    accessorFn: (row) => row.number,
    cell: ({ row }) => (
      <a href={row.original.url} target="_blank" rel="noreferrer">
        #{row.original.number}
      </a>
    ),
  },
  { accessorKey: "title", header: "Title" },
  { accessorKey: "author", header: "Author", cell: ({ getValue }) => (getValue() as string) ?? "ghost" },
  {
    accessorKey: "ageDays",
    header: "Age",
    cell: ({ getValue }) => `${getValue() as number}d`,
  },
  { accessorKey: "detail", header: "Detail" },
];

export function ActionCenterTab() {
  const action = useQuery({ queryKey: ["action-center"], queryFn: api.actionCenter });

  if (action.isLoading) {
    return <Loading what="action center" />;
  }
  if (action.isError) {
    return <ErrorView error={action.error} />;
  }
  const data = action.data;
  if (!data) {
    return <Empty message="No data yet. Add repos, then press Sync." />;
  }

  const total =
    data.prsAwaitingReview.length +
    data.prsFailingCi.length +
    data.prsOpenTooLong.length +
    data.staleIssues.length +
    data.unassignedIssues.length;

  if (total === 0) {
    return <Empty message="Nothing needs attention. Everything is fresh." />;
  }

  return (
    <div className="stack">
      <p className="hint">
        Thresholds: stale / open-too-long after <strong>{data.staleDays} days</strong>.
      </p>
      <Card title={`PRs awaiting review (${data.prsAwaitingReview.length})`}>
        <DataTable data={data.prsAwaitingReview} columns={columns} empty="No PRs awaiting review." />
      </Card>
      <Card title={`PRs with failing CI (${data.prsFailingCi.length})`}>
        <DataTable data={data.prsFailingCi} columns={columns} empty="No PRs with failing CI." />
      </Card>
      <Card title={`PRs open too long (${data.prsOpenTooLong.length})`}>
        <DataTable data={data.prsOpenTooLong} columns={columns} empty="No long-running PRs." />
      </Card>
      <Card title={`Stale issues (${data.staleIssues.length})`}>
        <DataTable data={data.staleIssues} columns={columns} empty="No stale issues." />
      </Card>
      <Card title={`Unassigned issues (${data.unassignedIssues.length})`}>
        <DataTable data={data.unassignedIssues} columns={columns} empty="No unassigned issues." />
      </Card>
    </div>
  );
}
