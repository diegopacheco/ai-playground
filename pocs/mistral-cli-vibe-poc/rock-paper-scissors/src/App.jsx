import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [playerChoice, setPlayerChoice] = useState(null)
  const [computerChoice, setComputerChoice] = useState(null)
  const [result, setResult] = useState(null)
  const [score, setScore] = useState({ player: 0, computer: 0 })
  const [isAnimating, setIsAnimating] = useState(false)
  const [gameStarted, setGameStarted] = useState(false)
  
  const choices = ['rock', 'paper', 'scissors']
  
  const determineWinner = (player, computer) => {
    if (player === computer) return 'draw'
    
    if (
      (player === 'rock' && computer === 'scissors') ||
      (player === 'paper' && computer === 'rock') ||
      (player === 'scissors' && computer === 'paper')
    ) {
      return 'player'
    }
    
    return 'computer'
  }
  
  const handleChoice = (choice) => {
    setPlayerChoice(choice)
    setGameStarted(true)
    setIsAnimating(true)
    
    // Computer makes random choice after a delay
    setTimeout(() => {
      const randomChoice = choices[Math.floor(Math.random() * choices.length)]
      setComputerChoice(randomChoice)
      
      const winner = determineWinner(choice, randomChoice)
      setResult(winner)
      
      if (winner === 'player') {
        setScore(prev => ({ ...prev, player: prev.player + 1 }))
      } else if (winner === 'computer') {
        setScore(prev => ({ ...prev, computer: prev.computer + 1 }))
      }
      
      setIsAnimating(false)
    }, 1500)
  }
  
  const resetGame = () => {
    setPlayerChoice(null)
    setComputerChoice(null)
    setResult(null)
    setGameStarted(false)
  }
  
  return (
    <div className="game-container">
      <h1>Rock Paper Scissors</h1>
      
      <div className="score-board">
        <div className="score">
          <span>Player: {score.player}</span>
          <span>Computer: {score.computer}</span>
        </div>
      </div>
      
      {!gameStarted ? (
        <div className="choices">
          {choices.map((choice) => (
            <button
              key={choice}
              className={`choice-button ${choice}`}
              onClick={() => handleChoice(choice)}
            >
              {choice.charAt(0).toUpperCase() + choice.slice(1)}
            </button>
          ))}
        </div>
      ) : (
        <div className="game-area">
          <div className="player-section">
            <h3>Your Choice</h3>
            <div className={`choice ${playerChoice} ${isAnimating ? 'animate' : ''}`}>
              {playerChoice && (
                <img
                  src={`/${playerChoice}.svg`}
                  alt={playerChoice}
                  className={isAnimating ? 'shake' : ''}
                />
              )}
            </div>
          </div>
          
          <div className="result">
            {result === 'draw' && <h2>It's a Draw!</h2>}
            {result === 'player' && <h2>You Win! 🎉</h2>}
            {result === 'computer' && <h2>Computer Wins! 🤖</h2>}
            
            <button className="play-again" onClick={resetGame}>Play Again</button>
          </div>
          
          <div className="computer-section">
            <h3>Computer's Choice</h3>
            <div className={`choice ${computerChoice} ${isAnimating ? 'animate' : ''}`}>
              {computerChoice ? (
                <img
                  src={`/${computerChoice}.svg`}
                  alt={computerChoice}
                  className={isAnimating ? 'shake' : ''}
                />
              ) : (
                <div className="question-mark">?</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App