import { useEffect, useState } from 'react'

type GameStatus = 'idle' | 'running' | 'paused' | 'ended'

type Piece = {
  row: number
  col: number
}

type Background = 'nebula' | 'sunset' | 'matrix'

type Difficulty = 'easy' | 'normal' | 'hard'

const rows = 20
const cols = 10
const spawnRow = 0
const spawnCol = 4
const pointsPerGoodMove = 10
const pointsPerLevel = 100

const difficultyMultiplier: Record<Difficulty, number> = {
  easy: 1.2,
  normal: 1,
  hard: 0.8
}

export default function App() {
  const [status, setStatus] = useState<GameStatus>('idle')
  const [piece, setPiece] = useState<Piece | null>(null)
  const [score, setScore] = useState(0)
  const [level, setLevel] = useState(1)
  const [forcedDropSeconds, setForcedDropSeconds] = useState(0)
  const [boardExpandSeconds, setBoardExpandSeconds] = useState(0)
  const [placements, setPlacements] = useState(0)
  const [adminOpen, setAdminOpen] = useState(false)
  const [background, setBackground] = useState<Background>('nebula')
  const [forcedDropIntervalSec, setForcedDropIntervalSec] = useState(40)
  const [boardExpandIntervalSec, setBoardExpandIntervalSec] = useState(30)
  const [difficulty, setDifficulty] = useState<Difficulty>('normal')
  const [maxLevels, setMaxLevels] = useState(10)

  const maxPlacements = maxLevels * 10

  const startGame = () => {
    setStatus('running')
    setPiece({ row: spawnRow, col: spawnCol })
    setScore(0)
    setLevel(1)
    setPlacements(0)
    setForcedDropSeconds(Math.ceil(forcedDropIntervalSec * difficultyMultiplier[difficulty]))
    setBoardExpandSeconds(boardExpandIntervalSec)
  }

  const isStartScreen = status === 'idle'
  const isRunning = status === 'running'
  const isEnded = status === 'ended'

  useEffect(() => {
    if (!isRunning || !piece) return
    const intervalMs = Math.ceil(forcedDropIntervalSec * difficultyMultiplier[difficulty] * 1000)
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
      setForcedDropSeconds(Math.ceil(forcedDropIntervalSec * difficultyMultiplier[difficulty]))
    }, intervalMs)
    return () => clearInterval(id)
  }, [isRunning, piece, forcedDropIntervalSec, difficulty, maxPlacements])

  useEffect(() => {
    if (!isRunning) return
    const id = setInterval(() => {
      setBoardExpandSeconds(boardExpandIntervalSec)
    }, boardExpandIntervalSec * 1000)
    return () => clearInterval(id)
  }, [isRunning, boardExpandIntervalSec])

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
          <label className="admin-field">
            Background
            <select
              value={background}
              onChange={(event) => setBackground(event.target.value as Background)}
            >
              <option value="nebula">Nebula</option>
              <option value="sunset">Sunset</option>
              <option value="matrix">Matrix</option>
            </select>
          </label>
          <label className="admin-field">
            Difficulty
            <select
              value={difficulty}
              onChange={(event) => {
                const value = event.target.value as Difficulty
                setDifficulty(value)
                setForcedDropSeconds(Math.ceil(forcedDropIntervalSec * difficultyMultiplier[value]))
              }}
            >
              <option value="easy">Easy</option>
              <option value="normal">Normal</option>
              <option value="hard">Hard</option>
            </select>
          </label>
          <label className="admin-field">
            Forced drop seconds
            <input
              type="number"
              min={1}
              value={forcedDropIntervalSec}
              onChange={(event) => {
                const value = Math.max(1, Number(event.target.value))
                setForcedDropIntervalSec(value)
                setForcedDropSeconds(Math.ceil(value * difficultyMultiplier[difficulty]))
              }}
            />
          </label>
          <label className="admin-field">
            Board expand seconds
            <input
              type="number"
              min={1}
              value={boardExpandIntervalSec}
              onChange={(event) => {
                const value = Math.max(1, Number(event.target.value))
                setBoardExpandIntervalSec(value)
                setBoardExpandSeconds(value)
              }}
            />
          </label>
          <label className="admin-field">
            Number of levels
            <input
              type="number"
              min={1}
              value={maxLevels}
              onChange={(event) => {
                const value = Math.max(1, Number(event.target.value))
                setMaxLevels(value)
              }}
            />
          </label>
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
        <div className={`board ${background}`} data-testid="board">
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
