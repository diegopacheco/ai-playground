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
const boardExpandMs = 30000
const pointsPerGoodMove = 10
const pointsPerLevel = 100
const maxPlacements = 10

export default function App() {
  const [status, setStatus] = useState<GameStatus>('idle')
  const [piece, setPiece] = useState<Piece | null>(null)
  const [score, setScore] = useState(0)
  const [level, setLevel] = useState(1)
  const [forcedDropSeconds, setForcedDropSeconds] = useState(0)
  const [boardExpandSeconds, setBoardExpandSeconds] = useState(0)
  const [placements, setPlacements] = useState(0)
  const [adminOpen, setAdminOpen] = useState(false)

  const startGame = () => {
    setStatus('running')
    setPiece({ row: spawnRow, col: spawnCol })
    setScore(0)
    setLevel(1)
    setPlacements(0)
    setForcedDropSeconds(Math.ceil(forcedDropMs / 1000))
    setBoardExpandSeconds(Math.ceil(boardExpandMs / 1000))
  }

  const isStartScreen = status === 'idle'
  const isRunning = status === 'running'
  const isEnded = status === 'ended'

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
          setPlacements((prev) => {
            const nextPlacements = prev + 1
            if (nextPlacements >= maxPlacements) {
              setStatus('ended')
              return nextPlacements
            }
            return nextPlacements
          })
          return { row: spawnRow, col: spawnCol }
        }
        return { ...current, row: nextRow }
      })
      setForcedDropSeconds(Math.ceil(forcedDropMs / 1000))
    }, forcedDropMs)
    return () => clearInterval(id)
  }, [isRunning, piece])

  useEffect(() => {
    if (!isRunning) return
    const id = setInterval(() => {
      setBoardExpandSeconds(Math.ceil(boardExpandMs / 1000))
    }, boardExpandMs)
    return () => clearInterval(id)
  }, [isRunning])

  useEffect(() => {
    if (!isRunning) return
    const id = setInterval(() => {
      setForcedDropSeconds((current) => (current > 0 ? current - 1 : current))
      setBoardExpandSeconds((current) => (current > 0 ? current - 1 : current))
    }, 1000)
    return () => clearInterval(id)
  }, [isRunning])

  const cells = Array.from({ length: rows * cols }, (_, index) => {
    const row = Math.floor(index / cols)
    const col = index % cols
    const isPiece = isRunning && piece && row === piece.row && col === piece.col
    return { key: `${row}-${col}`, isPiece }
  })

  return (
    <div className="app">
      <h1>Retris</h1>
      <div className="controls">
        {isStartScreen ? (
          <button type="button" onClick={startGame} aria-label="Start game">
            Start
          </button>
        ) : null}
        <button type="button" onClick={() => setAdminOpen((v) => !v)} aria-label="Admin settings">
          Admin
        </button>
      </div>
      <div className="status" aria-live="polite">
        Status: {status}
      </div>
      {adminOpen ? (
        <div className="admin" role="region" aria-label="Admin configuration">
          <h2>Admin Settings</h2>
          <div className="admin-list">
            <div>Level backgrounds</div>
            <div>Timers</div>
            <div>Difficulty</div>
            <div>Number of levels</div>
          </div>
        </div>
      ) : null}
      {isRunning ? (
        <div className="hud">
          <div>Score: {score}</div>
          <div>Level: {level}</div>
          <div>
            Forced drop in: {forcedDropSeconds}s
          </div>
          <div>
            Board expands in: {boardExpandSeconds}s
          </div>
          <div>Placements: {placements}/{maxPlacements}</div>
        </div>
      ) : null}
      {isEnded ? (
        <div className="end-state" role="status">
          Game over. Final score: {score}
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
