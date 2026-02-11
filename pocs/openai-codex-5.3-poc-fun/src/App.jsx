import { useMemo, useState } from 'react'

const choices = ['Rock', 'Paper', 'Scissors']

const beats = {
  Rock: 'Scissors',
  Paper: 'Rock',
  Scissors: 'Paper'
}

function getRoundResult(player, cpu) {
  if (player === cpu) return 'Draw'
  if (beats[player] === cpu) return 'Win'
  return 'Lose'
}

function cpuPick() {
  return choices[Math.floor(Math.random() * choices.length)]
}

export default function App() {
  const [playerChoice, setPlayerChoice] = useState('')
  const [cpuChoice, setCpuChoice] = useState('')
  const [result, setResult] = useState('Pick your move')
  const [score, setScore] = useState({ wins: 0, losses: 0, draws: 0 })

  const totalGames = useMemo(
    () => score.wins + score.losses + score.draws,
    [score.draws, score.losses, score.wins]
  )

  const winRate = useMemo(() => {
    if (totalGames === 0) return 0
    return Math.round((score.wins / totalGames) * 100)
  }, [score.wins, totalGames])

  const play = (move) => {
    const nextCpu = cpuPick()
    const round = getRoundResult(move, nextCpu)

    setPlayerChoice(move)
    setCpuChoice(nextCpu)
    setResult(round)

    setScore((current) => {
      if (round === 'Win') return { ...current, wins: current.wins + 1 }
      if (round === 'Lose') return { ...current, losses: current.losses + 1 }
      return { ...current, draws: current.draws + 1 }
    })
  }

  const reset = () => {
    setPlayerChoice('')
    setCpuChoice('')
    setResult('Pick your move')
    setScore({ wins: 0, losses: 0, draws: 0 })
  }

  return (
    <main className="page">
      <section className="card">
        <h1>Rock Paper Sizer</h1>
        <p className="subtitle">Choose your move and beat the machine.</p>

        <div className="actions">
          {choices.map((item) => (
            <button key={item} onClick={() => play(item)} className="move-btn">
              {item}
            </button>
          ))}
        </div>

        <div className="battle">
          <div className="pick-box">
            <span>You</span>
            <strong>{playerChoice || '-'}</strong>
          </div>
          <div className="result-box" data-result={result.toLowerCase()}>{result}</div>
          <div className="pick-box">
            <span>CPU</span>
            <strong>{cpuChoice || '-'}</strong>
          </div>
        </div>

        <div className="stats">
          <div>
            <span>Wins</span>
            <strong>{score.wins}</strong>
          </div>
          <div>
            <span>Losses</span>
            <strong>{score.losses}</strong>
          </div>
          <div>
            <span>Draws</span>
            <strong>{score.draws}</strong>
          </div>
          <div>
            <span>Win Rate</span>
            <strong>{winRate}%</strong>
          </div>
        </div>

        <button className="reset-btn" onClick={reset}>Reset</button>
      </section>
    </main>
  )
}
