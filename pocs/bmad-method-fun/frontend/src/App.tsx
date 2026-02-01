import { useState } from 'react'

type GameStatus = 'idle' | 'running' | 'paused' | 'ended'

const rows = 20
const cols = 10
const spawnRow = 0
const spawnCol = 4

export default function App() {
  const [status, setStatus] = useState<GameStatus>('idle')

  const startGame = () => {
    setStatus('running')
  }

  const isStartScreen = status === 'idle'
  const isRunning = status === 'running'

  const cells = Array.from({ length: rows * cols }, (_, index) => {
    const row = Math.floor(index / cols)
    const col = index % cols
    const isPiece = isRunning && row === spawnRow && col === spawnCol
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
