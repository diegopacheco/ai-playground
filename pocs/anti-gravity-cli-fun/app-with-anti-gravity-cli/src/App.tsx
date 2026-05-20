import { useState, useEffect } from 'react'

type Choice = 'rock' | 'paper' | 'scissors'
type Outcome = 'win' | 'lose' | 'draw'

interface Round {
  id: string
  player: Choice
  cpu: Choice
  outcome: Outcome
}

const choices: Choice[] = ['rock', 'paper', 'scissors']

const emojiMap: Record<Choice, string> = {
  rock: '✊',
  paper: '✋',
  scissors: '✌️'
}

function App() {
  const [playerScore, setPlayerScore] = useState(0)
  const [cpuScore, setCpuScore] = useState(0)
  const [draws, setDraws] = useState(0)
  const [streak, setStreak] = useState(0)
  const [playerChoice, setPlayerChoice] = useState<Choice | null>(null)
  const [cpuChoice, setCpuChoice] = useState<Choice | null>(null)
  const [shuffleChoice, setShuffleChoice] = useState<Choice>('rock')
  const [result, setResult] = useState<Outcome | null>(null)
  const [isShuffling, setIsShuffling] = useState(false)
  const [history, setHistory] = useState<Round[]>([])

  useEffect(() => {
    let intervalId: any
    if (isShuffling) {
      let index = 0
      intervalId = setInterval(() => {
        setShuffleChoice(choices[index % choices.length])
        index++
      }, 100)
    }
    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [isShuffling])

  const determineWinner = (player: Choice, cpu: Choice): Outcome => {
    if (player === cpu) return 'draw'
    if (
      (player === 'rock' && cpu === 'scissors') ||
      (player === 'paper' && cpu === 'rock') ||
      (player === 'scissors' && cpu === 'paper')
    ) {
      return 'win'
    }
    return 'lose'
  }

  const handlePlay = (choice: Choice) => {
    if (isShuffling) return

    setPlayerChoice(choice)
    setCpuChoice(null)
    setResult(null)
    setIsShuffling(true)

    setTimeout(() => {
      const randomCpuChoice = choices[Math.floor(Math.random() * choices.length)]
      const finalOutcome = determineWinner(choice, randomCpuChoice)

      setCpuChoice(randomCpuChoice)
      setResult(finalOutcome)
      setIsShuffling(false)

      if (finalOutcome === 'win') {
        setPlayerScore((prev) => prev + 1)
        setStreak((prev) => prev + 1)
      } else if (finalOutcome === 'lose') {
        setCpuScore((prev) => prev + 1)
        setStreak(0)
      } else {
        setDraws((prev) => prev + 1)
      }

      setHistory((prev) => [
        {
          id: Math.random().toString(36).substring(2, 9),
          player: choice,
          cpu: randomCpuChoice,
          outcome: finalOutcome
        },
        ...prev.slice(0, 4)
      ])
    }, 800)
  }

  const handleReset = () => {
    setPlayerScore(0)
    setCpuScore(0)
    setDraws(0)
    setStreak(0)
    setPlayerChoice(null)
    setCpuChoice(null)
    setResult(null)
    setHistory([])
  }

  const totalGames = playerScore + cpuScore + draws
  const winRate = totalGames > 0 ? Math.round((playerScore / totalGames) * 100) : 0

  return (
    <div className="game-container">
      <div className="header">
        <div className="title-area">
          <h1>RPS Arena</h1>
          <p>Defeat the CPU in a classic battle</p>
        </div>
        <div className="scoreboard">
          <div className="score-item">
            <span className="score-label">Player</span>
            <span className="score-val highlight-player">{playerScore}</span>
          </div>
          <div className="score-item">
            <span className="score-label">VS</span>
            <span className="score-val">&mdash;</span>
          </div>
          <div className="score-item">
            <span className="score-label">CPU</span>
            <span className="score-val highlight-cpu">{cpuScore}</span>
          </div>
        </div>
      </div>

      <div className="stats-bar">
        <span>Streak: <strong>{streak}</strong></span>
        <span>Draws: <strong>{draws}</strong></span>
        <span>Win Rate: <strong>{winRate}%</strong></span>
      </div>

      <div className="arena">
        <div className="fighter-card">
          <span className="card-title">Your Choice</span>
          <div className={`card-display ${playerChoice ? `active-${playerChoice}` : ''}`}>
            {playerChoice ? emojiMap[playerChoice] : '❓'}
          </div>
        </div>
        <div className="fighter-card">
          <span className="card-title">CPU Choice</span>
          <div className={`card-display ${isShuffling ? 'cpu-shuffle' : cpuChoice ? `active-${cpuChoice}` : ''}`}>
            {isShuffling ? emojiMap[shuffleChoice] : cpuChoice ? emojiMap[cpuChoice] : '❓'}
          </div>
        </div>
      </div>

      <div className="result-overlay">
        {result && (
          <>
            <span className={`result-text ${result}`}>
              {result === 'win' ? 'Victory!' : result === 'lose' ? 'Defeat!' : 'Draw!'}
            </span>
            <span className="result-subtitle">
              {result === 'win'
                ? `${playerChoice ? playerChoice.toUpperCase() : ''} beats ${cpuChoice ? cpuChoice.toUpperCase() : ''}`
                : result === 'lose'
                ? `${cpuChoice ? cpuChoice.toUpperCase() : ''} beats ${playerChoice ? playerChoice.toUpperCase() : ''}`
                : 'Great minds think alike'}
            </span>
          </>
        )}
      </div>

      <div className="choices-container">
        {choices.map((choice) => (
          <button
            key={choice}
            onClick={() => handlePlay(choice)}
            disabled={isShuffling}
            className={`choice-btn ${choice}`}
          >
            <span className="emoji">{emojiMap[choice]}</span>
            <span className="label">{choice}</span>
          </button>
        ))}
      </div>

      <div className="actions">
        <button onClick={handleReset} className="action-btn">
          Reset Game
        </button>
      </div>

      {history.length > 0 && (
        <div className="history-panel">
          <div className="history-header">
            <span>Recent Matches</span>
            <span>Last {history.length} Rounds</span>
          </div>
          <div className="history-list">
            {history.map((round) => (
              <div key={round.id} className="history-item">
                <div className="history-details">
                  <span>Player: {emojiMap[round.player]}</span>
                  <span>vs</span>
                  <span>CPU: {emojiMap[round.cpu]}</span>
                </div>
                <span className={`history-outcome ${round.outcome}`}>
                  {round.outcome}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
