import { useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getExpandedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import type { Auction } from "../types/index.ts";
import { AGENT_COLORS } from "../types/index.ts";
import { BidCard } from "./BidCard.tsx";

interface HistoryTableProps {
  auctions: Auction[];
}

export function HistoryTable({ auctions }: HistoryTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [expanded, setExpanded] = useState({});

  const columns: ColumnDef<Auction, unknown>[] = [
    {
      id: "expander",
      header: () => null,
      cell: ({ row }) => (
        <button
          onClick={row.getToggleExpandedHandler()}
          className="text-gray-400 hover:text-white cursor-pointer"
        >
          {row.getIsExpanded() ? "v" : ">"}
        </button>
      ),
      size: 40,
    },
    {
      accessorKey: "created_at",
      header: "Date",
      cell: ({ getValue }) => {
        const val = getValue() as string;
        if (!val) return "-";
        return new Date(val).toLocaleDateString();
      },
    },
    {
      id: "agents",
      header: "Agents",
      cell: ({ row }) => (
        <div className="flex gap-2">
          {(row.original.agents || []).map((a) => (
            <span
              key={a.agent_name}
              className="capitalize text-sm font-medium px-2 py-0.5 rounded"
              style={{
                color: AGENT_COLORS[a.agent_name] || "#6b7280",
                backgroundColor: `${AGENT_COLORS[a.agent_name] || "#6b7280"}20`,
              }}
            >
              {a.agent_name}
            </span>
          ))}
        </div>
      ),
    },
    {
      accessorKey: "winner",
      header: "Winner",
      cell: ({ getValue }) => {
        const winner = getValue() as string;
        if (!winner) return "-";
        return (
          <span
            className="capitalize font-bold"
            style={{ color: AGENT_COLORS[winner] || "#6b7280" }}
          >
            {winner}
          </span>
        );
      },
    },
    {
      id: "items_won",
      header: "Items Won",
      cell: ({ row }) => {
        const winner = row.original.winner;
        const agent = (row.original.agents || []).find(
          (a) => a.agent_name === winner
        );
        return agent ? agent.items_won : "-";
      },
    },
    {
      id: "total_spent",
      header: "Total Spent",
      cell: ({ row }) => {
        const winner = row.original.winner;
        const agent = (row.original.agents || []).find(
          (a) => a.agent_name === winner
        );
        if (!agent) return "-";
        return `$${agent.initial_budget - agent.remaining_budget}`;
      },
    },
  ];

  const table = useReactTable({
    data: auctions,
    columns,
    state: { sorting, expanded },
    onSortingChange: setSorting,
    onExpandedChange: setExpanded,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    getRowCanExpand: () => true,
  });

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <table className="w-full">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-gray-700">
              {hg.headers.map((header) => (
                <th
                  key={header.id}
                  className="text-left px-4 py-3 text-gray-400 text-sm font-medium cursor-pointer hover:text-white"
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                  {header.column.getIsSorted() === "asc"
                    ? " ^"
                    : header.column.getIsSorted() === "desc"
                    ? " v"
                    : ""}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <>
              <tr
                key={row.id}
                className="border-b border-gray-700/50 hover:bg-gray-700/30"
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3 text-white">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
              {row.getIsExpanded() && (
                <tr key={`${row.id}-expanded`}>
                  <td colSpan={columns.length} className="px-4 py-4 bg-gray-900">
                    <div className="space-y-4">
                      {(row.original.rounds || []).map((round) => (
                        <div key={round.round_number} className="space-y-2">
                          <div className="flex items-center gap-2 text-gray-300">
                            <span className="text-sm font-medium">
                              Round {round.round_number}:
                            </span>
                            <span>
                              {round.item_emoji} {round.item_name}
                            </span>
                            <span className="text-gray-500">-</span>
                            <span
                              className="capitalize font-bold"
                              style={{
                                color:
                                  AGENT_COLORS[round.winner_agent] || "#6b7280",
                              }}
                            >
                              {round.winner_agent} won (${round.winning_bid})
                            </span>
                          </div>
                          <div className="pl-4 space-y-2">
                            {(round.bids || []).map((bid) => (
                              <BidCard
                                key={`${round.round_number}-${bid.agent_name}`}
                                bid={bid}
                              />
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
      {auctions.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No auctions yet. Start one from the home page!
        </div>
      )}
    </div>
  );
}
