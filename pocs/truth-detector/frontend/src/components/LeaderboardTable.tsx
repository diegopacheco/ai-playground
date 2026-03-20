import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";
import type { LeaderboardEntry } from "../types";
import RankBadge from "./RankBadge";
import ClassificationBadge from "./ClassificationBadge";

interface LeaderboardTableProps {
  entries: LeaderboardEntry[];
}

const columnHelper = createColumnHelper<LeaderboardEntry>();

const columns = [
  columnHelper.accessor("rank", {
    header: "#",
    cell: (info) => <RankBadge rank={info.getValue()} />,
  }),
  columnHelper.accessor("github_user", {
    header: "User",
    cell: (info) => (
      <span className="font-medium text-gray-200">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor("avg_score", {
    header: "Avg Score",
    cell: (info) => (
      <span className="text-white font-bold">{info.getValue().toFixed(1)}</span>
    ),
  }),
  columnHelper.accessor("total_score", {
    header: "Total",
  }),
  columnHelper.accessor("deep_count", {
    header: "Deep",
    cell: (info) => (
      <span className="flex items-center gap-1">
        <ClassificationBadge classification="DEEP" />
        {info.getValue()}
      </span>
    ),
  }),
  columnHelper.accessor("decent_count", {
    header: "Decent",
    cell: (info) => (
      <span className="flex items-center gap-1">
        <ClassificationBadge classification="DECENT" />
        {info.getValue()}
      </span>
    ),
  }),
  columnHelper.accessor("shallow_count", {
    header: "Shallow",
    cell: (info) => (
      <span className="flex items-center gap-1">
        <ClassificationBadge classification="SHALLOW" />
        {info.getValue()}
      </span>
    ),
  }),
];

function LeaderboardTable({ entries }: LeaderboardTableProps) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "avg_score", desc: true },
  ]);

  const table = useReactTable({
    data: entries,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id} className="border-b border-gray-800">
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  className="text-left px-3 py-2 text-gray-400 font-medium cursor-pointer hover:text-gray-200 transition-colors select-none"
                >
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext()
                  )}
                  {header.column.getIsSorted() === "asc" && " \u2191"}
                  {header.column.getIsSorted() === "desc" && " \u2193"}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr
              key={row.id}
              className="border-b border-gray-800/50 hover:bg-gray-900/50 transition-colors"
            >
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-2.5 text-gray-300">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default LeaderboardTable;
