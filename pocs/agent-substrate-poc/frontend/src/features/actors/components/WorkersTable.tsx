import { createColumnHelper, tableFeatures, useTable } from "@tanstack/react-table";
import type { Worker } from "../types";

const features = tableFeatures({});
const helper = createColumnHelper<typeof features, Worker>();
const columns = helper.columns([
  helper.accessor("pod", { header: "Worker pod" }),
  helper.accessor("pool", { header: "Pool" }),
  helper.accessor("node", { header: "Node" }),
  helper.accessor("state", { header: "State" }),
  helper.display({
    id: "occupancy",
    header: "Actors",
    cell: (info) => {
      const w = info.row.original;
      return <span className={w.actors > 0 ? "busy" : "free"}>{w.actors > 0 ? `${w.actors} actor running` : "free, waiting"}</span>;
    },
  }),
]);

type WorkersTableProps = { readonly rows: Worker[] };

export function WorkersTable({ rows }: WorkersTableProps) {
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
