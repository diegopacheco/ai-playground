import { useEffect, useState } from 'react'

type GameStatus = 'idle' | 'running' | 'paused' | 'ended'

type Piece = {
  row: number
  col: number
}

const rows = 20
const cols = 10
const spawnRow = 0
const spawnCol = 4
const forcedDropMs = 40000
const pointsPerGoodMove = 10
const pointsPerLevel = 100

export default function App() {
  const [status, setStatus] = useState<GameStatus>('idle')
  const [piece, setPiece] = useState<Piece | null>(null)
  const [score, setScore] = useState(0)
  const [level, setLevel] = useState(1)

  const startGame = () => {
    setStatus('running')
    setPiece({ row: spawnRow, col: spawnCol })
    setScore(0)
    setLevel(1)
  }

  const isStartScreen = status === 'idle'
  const isRunning = status === 'running'

  useEffect(() => {
    if (!isRunning || !piece) return
    const id = setInterval(() => {
      setPiece((current) => {
        if (!current) return current
        const nextRow = current.row + 1
        if (nextRow >= rows - 1) {
          setScore((prev) => {
            const nextScore = prev + pointsPerGoodMove
            setLevel(Math.floor(nextScore / pointsPerLevel) + 1)
            return nextScore
          })
          return { row: spawnRow, col: spawnCol }
        }
        return { ...current, row: nextRow }
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
        <div className="hud">
          <div>Score: {score}</div>
          <div>Level: {level}</div>
        </div>
      ) : null}
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
