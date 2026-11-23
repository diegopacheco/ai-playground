import { createColumnHelper, tableFeatures, useTable } from "@tanstack/react-table";
import { Fragment, useState } from "react";
import type { SampleResult } from "../types.ts";
import { SampleDetail } from "./SampleDetail.tsx";

const features = tableFeatures({});
const helper = createColumnHelper<typeof features, SampleResult>();

const columns = helper.columns([
  helper.accessor("verdict", { header: "Verdict", cell: (c) => <span className={`badge badge-${c.getValue()}`}>{c.getValue()}</span> }),
  helper.accessor("title", { header: "Slop" }),
  helper.accessor("slop", { header: "Why it is slop" }),
  helper.accessor((s) => (s.expect === "caught" ? s.rule : s.gap), { id: "why", header: "Target rule or gap" }),
  helper.accessor("asExpected", { header: "Matches catalog", cell: (c) => (c.getValue() ? "yes" : "SURPRISE") }),
]);

type SampleTableProps = { readonly samples: readonly SampleResult[] };

export function SampleTable({ samples }: SampleTableProps) {
  const [open, setOpen] = useState<string | undefined>(undefined);
  const table = useTable({ features, columns, data: samples, getRowId: (s) => s.file });

  return (
    <table className="samples">
      <thead>
        {table.getHeaderGroups().map((group) => (
          <tr key={group.id}>
            {group.headers.map((header) => <th key={header.id}><table.FlexRender header={header} /></th>)}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <Fragment key={row.id}>
            <tr className="row" onClick={() => setOpen(open === row.id ? undefined : row.id)}>
              {row.getAllCells().map((cell) => <td key={cell.id}><table.FlexRender cell={cell} /></td>)}
            </tr>
            {open === row.id && (
              <tr><td colSpan={columns.length}><SampleDetail sample={row.original} /></td></tr>
            )}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}
