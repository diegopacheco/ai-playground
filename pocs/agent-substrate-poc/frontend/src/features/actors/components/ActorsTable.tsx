import { Link } from "@tanstack/react-router";
import { createColumnHelper, tableFeatures, useTable } from "@tanstack/react-table";
import { age } from "../format";
import type { Actor } from "../types";
import { ActionButtons } from "./ActionButtons";
import { StateBadge } from "./StateBadge";

const features = tableFeatures({});
const helper = createColumnHelper<typeof features, Actor>();
const columns = helper.columns([
  helper.accessor("name", {
    header: "Actor",
    cell: (info) => (
      <Link to="/actors/$name" params={{ name: info.getValue() }}>
        {info.getValue()}
      </Link>
    ),
  }),
  helper.accessor("state", { header: "State", cell: (info) => <StateBadge state={info.getValue()} /> }),
  helper.accessor("workerPod", { header: "Worker pod", cell: (info) => info.getValue() || "none (hibernated)" }),
  helper.accessor("version", { header: "Version" }),
  helper.accessor("snapshots", { header: "Snapshots" }),
  helper.accessor("createdAt", { header: "Age", cell: (info) => age(info.getValue(), Date.now()) }),
  helper.display({
    id: "actions",
    header: "Actions",
    cell: (info) => <ActionButtons name={info.row.original.name} state={info.row.original.state} />,
  }),
]);

type ActorsTableProps = { readonly rows: Actor[] };

export function ActorsTable({ rows }: ActorsTableProps) {
  const table = useTable({ features, columns, data: rows });
  return (
    <table className="grid">
      <thead>
        {table.getHeaderGroups().map((group) => (
          <tr key={group.id}>
            {group.headers.map((header) => (
              <th key={header.id}>{header.isPlaceholder ? null : <table.FlexRender header={header} />}</th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getAllCells().map((cell) => (
              <td key={cell.id}>
                <table.FlexRender cell={cell} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
