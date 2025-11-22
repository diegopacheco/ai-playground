import { useState, useEffect } from 'react'
import confetti from 'canvas-confetti'
import { questions } from './questions'
import './App.css'

function App() {
  const [gameState, setGameState] = useState(() => {
    const saved = localStorage.getItem('vibeCoderGame')
    return saved ? JSON.parse(saved) : {
      currentQuestion: 0,
      score: 0,
      skipsLeft: 2,
      vibeCodeUsed: false,
      timeBonus: 0,
      usedQuestions: [],
      gameStatus: 'playing'
    }
  })

  const [timeLeft, setTimeLeft] = useState(21)
  const [selectedAnswer, setSelectedAnswer] = useState(null)
  const [showResult, setShowResult] = useState(false)
  const [currentQ, setCurrentQ] = useState(null)

  useEffect(() => {
    localStorage.setItem('vibeCoderGame', JSON.stringify(gameState))
  }, [gameState])

  useEffect(() => {
    if (gameState.gameStatus === 'playing' && !showResult) {
      const getNewQuestion = () => {
        const available = questions.filter((_, idx) => !gameState.usedQuestions.includes(idx))
        if (available.length === 0) return null
        const randomIdx = Math.floor(Math.random() * available.length)
        const questionIdx = questions.findIndex((q, idx) =>
          q === available[randomIdx] && !gameState.usedQuestions.includes(idx)
        )
        return { question: questions[questionIdx], index: questionIdx }
      }

      if (!currentQ) {
        const newQ = getNewQuestion()
        if (newQ) {
          setCurrentQ(newQ)
          setTimeLeft(21 + gameState.timeBonus)
        }
      }
    }
  }, [gameState, currentQ, showResult])

  useEffect(() => {
    if (gameState.gameStatus === 'playing' && !showResult && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0 && !showResult) {
      handleGameOver()
    }
  }, [timeLeft, gameState.gameStatus, showResult])

  const handleAnswer = (answerIdx) => {
    if (selectedAnswer !== null || showResult) return
    setSelectedAnswer(answerIdx)
    setShowResult(true)

    if (answerIdx === currentQ.question.correct) {
      confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
      })

      setTimeout(() => {
        const newUsedQuestions = [...gameState.usedQuestions, currentQ.index]
        const newScore = gameState.score + 1

        if (newScore === 10) {
          setGameState({
            ...gameState,
            score: newScore,
            usedQuestions: newUsedQuestions,
            gameStatus: 'won'
          })
        } else {
          setGameState({
            ...gameState,
            score: newScore,
            usedQuestions: newUsedQuestions,
            timeBonus: 0
          })
          setCurrentQ(null)
          setSelectedAnswer(null)
          setShowResult(false)
        }
      }, 2000)
    } else {
      setTimeout(() => {
        handleGameOver()
      }, 2000)
    }
  }

  const handleSkip = () => {
    if (gameState.skipsLeft > 0) {
      setGameState({
        ...gameState,
        skipsLeft: gameState.skipsLeft - 1
      })
      setCurrentQ(null)
      setSelectedAnswer(null)
      setShowResult(false)
    }
  }

  const handleVibeCode = () => {
    if (!gameState.vibeCodeUsed) {
      const roll = Math.random()
      if (roll < 0.3) {
        setGameState({
          ...gameState,
          vibeCodeUsed: true,
          timeBonus: 10
        })
        setTimeLeft(timeLeft + 10)
        alert('VIBE CODE SUCCESS! +10 seconds!')
      } else {
        setGameState({
          ...gameState,
          vibeCodeUsed: true
        })
        alert('VIBE CODE FAILED! No bonus this time.')
      }
    }
  }

  const handleGameOver = () => {
    setGameState({
      ...gameState,
      gameStatus: 'lost'
    })
  }

  const resetGame = () => {
    setGameState({
      currentQuestion: 0,
      score: 0,
      skipsLeft: 2,
      vibeCodeUsed: false,
      timeBonus: 0,
      usedQuestions: [],
      gameStatus: 'playing'
    })
    setCurrentQ(null)
    setSelectedAnswer(null)
    setShowResult(false)
    setTimeLeft(21)
  }

  if (gameState.gameStatus === 'won') {
    return (
      <div className="game-container">
        <div className="victory-screen">
          <h1>🎉 CONGRATULATIONS! 🎉</h1>
          <h2>You are officially a VIBE CODER!</h2>
          <p>You answered all 10 questions correctly!</p>
          <button onClick={resetGame} className="reset-btn">Play Again</button>
        </div>
      </div>
    )
  }

  if (gameState.gameStatus === 'lost') {
    return (
      <div className="game-container">
        <div className="game-over-screen">
          <h1>Game Over</h1>
          <h2>Score: {gameState.score}/10</h2>
          <p>Better luck next time, future Vibe Coder!</p>
          <button onClick={resetGame} className="reset-btn">Try Again</button>
        </div>
      </div>
    )
  }

  if (!currentQ) {
    return <div className="game-container"><h2>Loading...</h2></div>
  }

  return (
    <div className="game-container">
      <header className="game-header">
        <h1>Who Wants to be a Vibe Coder?</h1>
        <div className="stats">
          <div>Score: {gameState.score}/10</div>
          <div className={`timer ${timeLeft <= 5 ? 'warning' : ''}`}>Time: {timeLeft}s</div>
        </div>
      </header>

      <div className="question-section">
        <h2>{currentQ.question.question}</h2>
        <div className="answers">
          {currentQ.question.options.map((option, idx) => (
            <button
              key={idx}
              onClick={() => handleAnswer(idx)}
              disabled={selectedAnswer !== null}
              className={`answer-btn ${
                showResult && idx === currentQ.question.correct ? 'correct' :
                showResult && idx === selectedAnswer ? 'wrong' : ''
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      <div className="powerups">
        <button
          onClick={handleSkip}
          disabled={gameState.skipsLeft === 0 || showResult}
          className="powerup-btn"
        >
          Skip ({gameState.skipsLeft} left)
        </button>
        <button
          onClick={handleVibeCode}
          disabled={gameState.vibeCodeUsed || showResult}
          className="powerup-btn"
        >
          Vibe Code {gameState.vibeCodeUsed ? '(Used)' : ''}
        </button>
      </div>
    </div>
  )
}

export default App
