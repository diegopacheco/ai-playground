import { useState } from 'react'

type GameStatus = 'idle' | 'running' | 'paused' | 'ended'

export default function App() {
  const [status, setStatus] = useState<GameStatus>('idle')

  const startGame = () => {
    setStatus('running')
  }

  const isStartScreen = status === 'idle'

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
    </div>
  )
}
