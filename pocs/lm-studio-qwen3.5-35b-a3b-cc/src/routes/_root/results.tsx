import { useState, useEffect } from 'react'

type Choice = 'rock' | 'paper' | 'scissors'
type Result = 'win' | 'lose' | 'draw'

interface GameResult {
  id: number
  playerChoice: Choice
  computerChoice: Choice
  result: Result
  timestamp: string
}

function formatChoice(choice: Choice): string {
  const icons = { rock: '🪨', paper: '📄', scissors: '✂️' }
  return `${icons[choice]} ${choice.charAt(0).toUpperCase() + choice.slice(1)}`
}

export default function Results() {
  const [results, setResults] = useState<GameResult[]>([])
  const [gameCount, setGameCount] = useState(0)

  useEffect(() => {
    const stored = localStorage.getItem('rockPaperScissorsResults')
    if (stored) {
      const parsed: GameResult[] = JSON.parse(stored)
      setResults(parsed)
      setGameCount(parsed.length)
    }
  }, [])

  function clearResults() {
    localStorage.removeItem('rockPaperScissorsResults')
    setResults([])
    setGameCount(0)
  }

  const stats = results.reduce(
    (acc, r) => {
      acc.total++
      if (r.result === 'win') acc.wins++
      else if (r.result === 'lose') acc.losses++
      else acc.draws++
      return acc
    },
    { total: 0, wins: 0, losses: 0, draws: 0 }
  )

  return (
    <div>
      <h2>All Results</h2>
      <p style={{ color: '#666' }}>Total games: {gameCount}</p>

      {results.length > 0 && (
        <div style={{ marginBottom: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
          <h3 style={{ margin: '0 0 10px 0' }}>Statistics</h3>
          <p style={{ margin: '5px 0' }}>Wins: {stats.wins}</p>
          <p style={{ margin: '5px 0' }}>Losses: {stats.losses}</p>
          <p style={{ margin: '5px 0' }}>Draws: {stats.draws}</p>
        </div>
      )}

      {results.length === 0 ? (
        <p style={{ color: '#666' }}>No games played yet. <a href="/" style={{ color: '#007bff' }}>Go play!</a></p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {results.map((r) => (
            <div
              key={r.id}
              style={{
                padding: '15px',
                background: r.result === 'win' ? '#d4edda' : r.result === 'lose' ? '#f8d7da' : '#e2e3e5',
                borderRadius: '8px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div>
                <span>{formatChoice(r.playerChoice)}</span>
                {' vs '}
                <span>{formatChoice(r.computerChoice)}</span>
              </div>
              <span style={{ fontWeight: 'bold' }}>
                {r.result === 'win' ? 'WIN' : r.result === 'lose' ? 'LOSE' : 'DRAW'}
              </span>
            </div>
          ))}
        </div>
      )}

      {results.length > 0 && (
        <button
          onClick={clearResults}
          style={{
            marginTop: '20px',
            padding: '10px 20px',
            cursor: 'pointer',
            border: 'none',
            borderRadius: '5px',
            background: '#dc3545',
            color: 'white',
          }}
        >
          Clear All Results
        </button>
      )}
    </div>
  )
}
