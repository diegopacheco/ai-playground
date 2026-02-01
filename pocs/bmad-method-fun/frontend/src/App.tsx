import { useEffect, useState } from 'react'

type GameStatus = 'idle' | 'running' | 'paused' | 'ended'

type Piece = {
  row: number
  col: number
  placed: boolean
}

const rows = 20
const cols = 10
const spawnRow = 0
const spawnCol = 4
const forcedDropMs = 40000

export default function App() {
  const [status, setStatus] = useState<GameStatus>('idle')
  const [piece, setPiece] = useState<Piece | null>(null)

  const startGame = () => {
    setStatus('running')
    setPiece({ row: spawnRow, col: spawnCol, placed: false })
  }

  const isStartScreen = status === 'idle'
  const isRunning = status === 'running'

  useEffect(() => {
    if (!isRunning || !piece || piece.placed) return
    const id = setInterval(() => {
      setPiece((current) => {
        if (!current || current.placed) return current
        const nextRow = Math.min(current.row + 1, rows - 1)
        const placed = nextRow === rows - 1
        return { ...current, row: nextRow, placed }
      })
    }, forcedDropMs)
    return () => clearInterval(id)
  }, [isRunning, piece])

  const cells = Array.from({ length: rows * cols }, (_, index) => {
    const row = Math.floor(index / cols)
    const col = index % cols
    const isPiece = isRunning && piece && row === piece.row && col === piece.col
    return { key: `${row}-${col}`, isPiece }
  })

  return (
    <div className="app">
      <h1>Retris</h1>
      {isStartScreen ? (
        <button type="button" onClick={startGame} aria-label="Start game">
          Start
        </button>
      ) : null}
      <div className="status" aria-live="polite">
        Status: {status}
      </div>
      {isRunning ? (
        <div className="board" data-testid="board">
          {cells.map((cell) => (
            <div
              key={cell.key}
              className={cell.isPiece ? 'cell piece' : 'cell'}
              data-testid={cell.isPiece ? 'piece' : undefined}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}
