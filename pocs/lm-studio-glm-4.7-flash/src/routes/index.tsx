import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useState, useEffect } from 'react'
import React from 'react'

type Choice = 'rock' | 'paper' | 'scissors'
type Result = 'win' | 'lose' | 'tie'

export const Route = createFileRoute('/')({
  component: Game,
})

function Game() {
  const [playerChoice, setPlayerChoice] = useState<Choice | null>(null)
  const [computerChoice, setComputerChoice] = useState<Choice | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [history, setHistory] = useState<Array<{ player: Choice; computer: Choice; result: Result }>>([])
  const navigate = useNavigate()

  const choices: Choice[] = ['rock', 'paper', 'scissors']

  const loadHistory = () => {
    const savedHistory = localStorage.getItem('gameHistory')
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory))
    }
  }

  const saveHistory = (newHistory: Array<{ player: Choice; computer: Choice; result: Result }>) => {
    localStorage.setItem('gameHistory', JSON.stringify(newHistory))
  }

  const play = (choice: Choice) => {
    const computerChoice = choices[Math.floor(Math.random() * choices.length)]
    const player = choice
    const computer = computerChoice

    let result: Result
    if (player === computer) {
      result = 'tie'
    } else if (
      (player === 'rock' && computer === 'scissors') ||
      (player === 'paper' && computer === 'rock') ||
      (player === 'scissors' && computer === 'paper')
    ) {
      result = 'win'
    } else {
      result = 'lose'
    }

    setPlayerChoice(player)
    setComputerChoice(computer)
    setResult(result)

    const newHistory = [...history, { player, computer, result }]
    setHistory(newHistory)
    saveHistory(newHistory)
  }

  const reset = () => {
    setPlayerChoice(null)
    setComputerChoice(null)
    setResult(null)
  }

  React.useEffect(() => {
    loadHistory()
  }, [])

  const getResultColor = (result: Result) => {
    if (result === 'win') return 'green'
    if (result === 'lose') return 'red'
    return 'gray'
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Paper, Rock, Scissors</h1>
      <p>Choose your weapon</p>

      {playerChoice && computerChoice && result && (
        <div style={{ marginTop: '2rem', padding: '1rem', background: '#f0f0f0', borderRadius: '8px' }}>
          <p>You chose: <strong>{playerChoice}</strong></p>
          <p>Computer chose: <strong>{computerChoice}</strong></p>
          <p style={{ color: getResultColor(result), fontSize: '1.2rem', fontWeight: 'bold' }}>
            {result === 'win' && 'You win!'}
            {result === 'lose' && 'You lose!'}
            {result === 'tie' && "It's a tie!"}
          </p>
          <button
            onClick={reset}
            style={{ marginTop: '1rem', padding: '0.5rem 1rem', cursor: 'pointer' }}
          >
            Play again
          </button>
        </div>
      )}

      <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
        {choices.map((choice) => (
          <button
            key={choice}
            onClick={() => play(choice)}
            disabled={!!playerChoice}
            style={{
              padding: '1rem 2rem',
              fontSize: '1.2rem',
              cursor: playerChoice ? 'not-allowed' : 'pointer',
              background: playerChoice ? '#ccc' : '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px'
            }}
          >
            {choice}
          </button>
        ))}
      </div>

      <div style={{ marginTop: '2rem' }}>
        <button
          onClick={() => navigate({ to: '/history' })}
          style={{ padding: '0.5rem 1rem', cursor: 'pointer' }}
        >
          View History
        </button>
      </div>
    </div>
  )
}