import { useState } from 'react'
import { useReactTable, getCoreRowModel, flexRender, createColumnHelper } from '@tanstack/react-table'
import { useStore } from './useStore.js'
import './History.css'

const columnHelper = createColumnHelper()

const columns = [
  columnHelper.accessor('date', { header: 'Date' }),
  columnHelper.accessor((row) => row.player1.name, { id: 'player1', header: 'Player 1' }),
  columnHelper.accessor((row) => row.player2.name, { id: 'player2', header: 'Player 2' }),
  columnHelper.accessor((row) => `${row.player1.score} - ${row.player2.score}`, { id: 'score', header: 'Score' }),
  columnHelper.accessor('champion', { header: 'Champion' }),
]

function History() {
  const state = useStore()
  const [selectedBattle, setSelectedBattle] = useState(null)

  const table = useReactTable({
    data: state.battleHistory,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  if (state.battleHistory.length === 0) {
    return (
      <div className="history-page">
        <h2>Battle History</h2>
        <p className="no-history">No battles yet! Go fight some Pokemon!</p>
      </div>
    )
  }

  return (
    <div className="history-page">
      <h2>Battle History</h2>
      <div className="history-table-wrapper">
        <table className="history-table">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => (
                  <th key={header.id}>
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={() => setSelectedBattle(row.original)}
                className={selectedBattle?.id === row.original.id ? 'selected' : ''}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedBattle && (
        <div className="battle-detail">
          <h3>Battle Details</h3>
          <div className="detail-champion">
            Champion: <span className="champ-name">{selectedBattle.champion}</span>
          </div>
          <div className="detail-score">
            {selectedBattle.player1.name} {selectedBattle.player1.score} - {selectedBattle.player2.score} {selectedBattle.player2.name}
          </div>
          <div className="detail-rounds">
            {selectedBattle.rounds.map((r, i) => (
              <div key={i} className="detail-round">
                <span className="round-label">Round {r.round}</span>
                <div className="round-matchup">
                  <div className="round-pokemon">
                    <img src={r.p1Card.sprite} alt={r.p1Card.name} />
                    <span>{r.p1Card.name}</span>
                    <span className="round-power">Power: {r.p1Card.power}</span>
                  </div>
                  <span className="round-vs">VS</span>
                  <div className="round-pokemon">
                    <img src={r.p2Card.sprite} alt={r.p2Card.name} />
                    <span>{r.p2Card.name}</span>
                    <span className="round-power">Power: {r.p2Card.power}</span>
                  </div>
                </div>
                <span className={`round-winner ${r.winner === 0 ? 'tie' : ''}`}>
                  {r.winner === 0 ? 'Tie' : r.winner === 1 ? `${selectedBattle.player1.name} wins` : `${selectedBattle.player2.name} wins`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default History
