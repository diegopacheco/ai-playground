import { GameCell } from './GameCell'
import { getAgentInfo } from '../types'

interface GameBoardProps {
  board: string[][]
  currentPlayer: string
  status: 'playing' | 'finished'
  winner: string | null
  isDraw: boolean
  thinkingAgent: string | null
  lastMove: number | null
  durationMs: number | null
  error: string | null
  agentA: string
  agentB: string
}

export function GameBoard({
  board,
  currentPlayer,
  status,
  winner,
  isDraw,
  thinkingAgent,
  lastMove,
  durationMs,
  error,
  agentA,
  agentB,
}: GameBoardProps) {
  const findLastMoveRow = () => {
    if (lastMove === null) return -1
    for (let row = 0; row < 6; row++) {
      if (board[row][lastMove] !== '.') {
        return row
      }
    }
    return -1
  }
  const lastMoveRow = findLastMoveRow()

  const agentAInfo = getAgentInfo(agentA)
  const agentBInfo = getAgentInfo(agentB)
  const thinkingAgentInfo = thinkingAgent ? getAgentInfo(thinkingAgent) : null
  const winnerInfo = winner ? getAgentInfo(winner) : null

  return (
    <div className="flex flex-col items-center gap-6 p-8">
      <div className="flex justify-between w-full max-w-2xl">
        <div className={`flex items-center gap-3 p-4 rounded-lg ${currentPlayer === agentA ? 'bg-gray-700 ring-2 ring-red-500' : 'bg-gray-800'}`}>
          <div className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center font-bold text-sm">X</div>
          <div>
            <div className="font-bold text-lg" style={{ color: agentAInfo.color }}>{agentAInfo.name}</div>
            <div className="text-xs text-gray-400 font-mono">{agentAInfo.model}</div>
          </div>
        </div>
        <div className="flex items-center text-2xl font-bold text-gray-500">VS</div>
        <div className={`flex items-center gap-3 p-4 rounded-lg ${currentPlayer === agentB ? 'bg-gray-700 ring-2 ring-yellow-400' : 'bg-gray-800'}`}>
          <div>
            <div className="font-bold text-lg text-right" style={{ color: agentBInfo.color }}>{agentBInfo.name}</div>
            <div className="text-xs text-gray-400 font-mono text-right">{agentBInfo.model}</div>
          </div>
          <div className="w-8 h-8 rounded-full bg-yellow-400 flex items-center justify-center font-bold text-sm text-gray-900">O</div>
        </div>
      </div>
      {status === 'playing' && thinkingAgentInfo && (
        <div className="flex items-center gap-3 text-xl">
          <div className="w-3 h-3 rounded-full animate-pulse" style={{ backgroundColor: thinkingAgentInfo.color }} />
          <span style={{ color: thinkingAgentInfo.color }}>{thinkingAgentInfo.name}</span>
          <span className="text-gray-400">is thinking...</span>
          <span className="text-gray-500 text-sm font-mono">({thinkingAgentInfo.model})</span>
        </div>
      )}
      {status === 'finished' && (
        <div className="flex flex-col items-center gap-2">
          <div className="text-3xl font-bold">
            {isDraw ? (
              <span className="text-gray-300">Draw!</span>
            ) : winnerInfo && (
              <span style={{ color: winnerInfo.color }}>
                {winnerInfo.name} wins!
              </span>
            )}
          </div>
          {winner && winnerInfo && (
            <div className="text-sm text-gray-400">
              Model: <span className="font-mono text-white">{winnerInfo.model}</span>
            </div>
          )}
          {durationMs && (
            <div className="text-sm text-gray-500">
              Game duration: {(durationMs / 1000).toFixed(1)}s
            </div>
          )}
        </div>
      )}
      {error && (
        <div className="text-red-400 text-sm bg-red-900/20 px-4 py-2 rounded">{error}</div>
      )}
      <div className="bg-blue-800 p-4 rounded-xl shadow-2xl">
        <div className="flex gap-1 mb-2 px-1">
          {[0, 1, 2, 3, 4, 5, 6].map((col) => (
            <div key={col} className="w-14 text-center text-gray-400 text-sm font-mono">
              {col}
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-1">
          {board.map((row, rowIndex) => (
            <div key={rowIndex} className="flex gap-1">
              {row.map((cell, colIndex) => (
                <GameCell
                  key={`${rowIndex}-${colIndex}`}
                  value={cell}
                  isLastMove={rowIndex === lastMoveRow && colIndex === lastMove}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
