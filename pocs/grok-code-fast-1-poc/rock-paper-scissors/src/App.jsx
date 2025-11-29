import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [score, setScore] = useState({ wins: 0, losses: 0, ties: 0 })
  const [playerChoice, setPlayerChoice] = useState('')
  const [computerChoice, setComputerChoice] = useState('')
  const [result, setResult] = useState('')

  const choices = ['rock', 'paper', 'scissors']

  useEffect(() => {
    const storedScore = localStorage.getItem('rps-score')
    if (storedScore) {
      setScore(JSON.parse(storedScore))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('rps-score', JSON.stringify(score))
  }, [score])

  const playGame = (choice) => {
    const computer = choices[Math.floor(Math.random() * choices.length)]
    setPlayerChoice(choice)
    setComputerChoice(computer)

    if (choice === computer) {
      setResult('It\'s a tie!')
      setScore(prev => ({ ...prev, ties: prev.ties + 1 }))
    } else if (
      (choice === 'rock' && computer === 'scissors') ||
      (choice === 'paper' && computer === 'rock') ||
      (choice === 'scissors' && computer === 'paper')
    ) {
      setResult('You win!')
      setScore(prev => ({ ...prev, wins: prev.wins + 1 }))
    } else {
      setResult('You lose!')
      setScore(prev => ({ ...prev, losses: prev.losses + 1 }))
    }
  }

  const resetScore = () => {
    setScore({ wins: 0, losses: 0, ties: 0 })
    setPlayerChoice('')
    setComputerChoice('')
    setResult('')
  }

  return (
    <div className="app">
      <h1>Rock Paper Scissors</h1>
      <div className="score">
        <p>Wins: {score.wins}</p>
        <p>Losses: {score.losses}</p>
        <p>Ties: {score.ties}</p>
      </div>
      <div className="choices">
        <button onClick={() => playGame('rock')}>Rock</button>
        <button onClick={() => playGame('paper')}>Paper</button>
        <button onClick={() => playGame('scissors')}>Scissors</button>
      </div>
      {playerChoice && (
        <div className="result">
          <p>You chose: {playerChoice}</p>
          <p>Computer chose: {computerChoice}</p>
          <p>{result}</p>
        </div>
      )}
      <button onClick={resetScore} className="reset">Reset Score</button>
    </div>
  )
}

export default App
