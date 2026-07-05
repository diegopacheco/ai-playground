import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table"
import { useMemo } from "react"
import type { Episode } from "../../shared/types"
import { Icon } from "./Icon"

export function EpisodeList({ episodes, onToggle, busy }: { episodes: Episode[]; onToggle: (id: string) => void; busy: boolean }) {
  const columns = useMemo<ColumnDef<Episode>[]>(() => [
    { accessorKey: "number", header: "Episode", cell: info => <span className="episode-number">E{String(info.getValue<number>()).padStart(2, "0")}</span> },
    { accessorKey: "title", header: "Title", cell: info => <strong>{info.getValue<string>()}</strong> },
    { accessorKey: "runtime", header: "Runtime", cell: info => <span>{info.getValue<number>()} min</span> },
    { accessorKey: "watched", header: "Status", cell: info => {
      const episode = info.row.original
      return <button className={episode.watched ? "episode-toggle watched" : "episode-toggle"} onClick={() => onToggle(episode.id)} disabled={busy} aria-label={`${episode.watched ? "Mark unwatched" : "Mark watched"}: ${episode.title}`}>
        <Icon name="check" size={15}/>{episode.watched ? "Watched" : "Mark watched"}
      </button>
    }}
  ], [onToggle, busy])
  const table = useReactTable({ data: episodes, columns, getCoreRowModel: getCoreRowModel() })
  return <div className="episode-table-wrap"><table className="episode-table">
    <thead>{table.getHeaderGroups().map(group => <tr key={group.id}>{group.headers.map(header => <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>)}</tr>)}</thead>
    <tbody>{table.getRowModel().rows.map(row => <tr key={row.id} className={row.original.watched ? "is-watched" : ""}>{row.getVisibleCells().map(cell => <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}</tbody>
  </table></div>
}
