import { Link } from "@tanstack/react-router";
import { createColumnHelper, tableFeatures, useTable } from "@tanstack/react-table";
import { formatDuration } from "../format";
import type { ConversionSummary } from "../types";
import { StatusBadge } from "./StatusBadge";

const features = tableFeatures({});
const helper = createColumnHelper<typeof features, ConversionSummary>();
const columns = helper.columns([
  helper.accessor("filename", {
    header: "File",
    cell: (info) => (
      <Link to="/conversions/$id" params={{ id: info.row.original.id }}>
        {info.getValue()}
      </Link>
    ),
  }),
  helper.accessor("format", { header: "Format" }),
  helper.accessor("status", { header: "Status", cell: (info) => <StatusBadge status={info.getValue()} /> }),
  helper.accessor("pages", { header: "Pages" }),
  helper.accessor("tables", { header: "Tables" }),
  helper.accessor("ocr", { header: "OCR", cell: (info) => (info.getValue() ? "yes" : "no") }),
  helper.accessor("durationMs", { header: "Time", cell: (info) => formatDuration(info.getValue()) }),
  helper.accessor("createdAt", { header: "When", cell: (info) => new Date(info.getValue()).toLocaleTimeString() }),
]);

type HistoryTableProps = { readonly rows: ConversionSummary[] };

export function HistoryTable({ rows }: HistoryTableProps) {
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
