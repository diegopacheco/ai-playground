import { useState, useEffect } from 'react'
import { Link } from '@tanstack/react-router'

type Choice = 'rock' | 'paper' | 'scissors'
type Result = 'win' | 'lose' | 'draw'

interface GameResult {
  id: number
  playerChoice: Choice
  computerChoice: Choice
  result: Result
  timestamp: string
}

const choices: Choice[] = ['rock', 'paper', 'scissors']

function determineResult(player: Choice, computer: Choice): Result {
  if (player === computer) return 'draw'
  if (
    (player === 'rock' && computer === 'scissors') ||
    (player === 'paper' && computer === 'rock') ||
    (player === 'scissors' && computer === 'paper')
  ) {
    return 'win'
  }
  return 'lose'
}

function formatChoice(choice: Choice): string {
  const icons = { rock: '🪨', paper: '📄', scissors: '✂️' }
  return `${icons[choice]} ${choice.charAt(0).toUpperCase() + choice.slice(1)}`
}

export default function Home() {
  const [results, setResults] = useState<GameResult[]>([])
  const [playerChoice, setPlayerChoice] = useState<Choice | null>(null)
  const [computerChoice, setComputerChoice] = useState<Choice | null>(null)
  const [currentResult, setCurrentResult] = useState<Result | null>(null)
  const [gameCount, setGameCount] = useState(0)

  useEffect(() => {
    const stored = localStorage.getItem('rockPaperScissorsResults')
    if (stored) {
      const parsed: GameResult[] = JSON.parse(stored)
      setResults(parsed)
      setGameCount(parsed.length)
    }
  }, [])

  function play(player: Choice) {
    const computer = choices[Math.floor(Math.random() * choices.length)]
    const result = determineResult(player, computer)

    const newResult: GameResult = {
      id: Date.now(),
      playerChoice: player,
      computerChoice: computer,
      result,
      timestamp: new Date().toISOString(),
    }

    const updated = [newResult, ...results]
    setResults(updated)
    localStorage.setItem('rockPaperScissorsResults', JSON.stringify(updated))
    setGameCount(updated.length)
    setCurrentResult(result)
    setPlayerChoice(player)
    setComputerChoice(computer)
  }

  function resetGame() {
    setPlayerChoice(null)
    setComputerChoice(null)
    setCurrentResult(null)
  }

  const resultText = currentResult === 'win' ? 'You Win!' : currentResult === 'lose' ? 'You Lose!' : "It's a Draw!"

  return (
    <div>
      <p style={{ color: '#666', marginBottom: '20px' }}>Games played: {gameCount}</p>

      <h2>Play</h2>

      {!playerChoice ? (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {choices.map((choice) => (
            <button
              key={choice}
              onClick={() => play(choice)}
              style={{
                padding: '15px 25px',
                fontSize: '24px',
                cursor: 'pointer',
                border: 'none',
                borderRadius: '8px',
                background: '#007bff',
                color: 'white',
              }}
            >
              {formatChoice(choice)}
            </button>
          ))}
        </div>
      ) : (
        <div style={{ marginTop: '20px' }}>
          <p>
            You chose: {formatChoice(playerChoice)}
          </p>
          <p>
            Computer chose: {formatChoice(computerChoice)}
          </p>
          <h3 style={{ color: currentResult === 'win' ? '#28a745' : currentResult === 'lose' ? '#dc3545' : '#6c757d' }}>
            {resultText}
          </h3>
          <button
            onClick={resetGame}
            style={{
              padding: '10px 20px',
              marginTop: '10px',
              cursor: 'pointer',
              border: 'none',
              borderRadius: '5px',
              background: '#6c757d',
              color: 'white',
            }}
          >
            Play Again
          </button>
        </div>
      )}

      <h2 style={{ marginTop: '30px' }}>Recent Results</h2>
      {results.slice(0, 5).map((r) => (
        <div key={r.id} style={{ padding: '10px', marginBottom: '10px', background: '#f8f9fa', borderRadius: '5px' }}>
          {formatChoice(r.playerChoice)} vs {formatChoice(r.computerChoice)} - {r.result.toUpperCase()}
        </div>
      ))}

      <p style={{ marginTop: '20px' }}>
        View all results: <Link to="/results" style={{ color: '#007bff' }}>Results Page</Link>
      </p>
    </div>
  )
}
