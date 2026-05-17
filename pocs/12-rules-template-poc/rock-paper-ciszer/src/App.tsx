import { useState } from 'react'

type Move = 'rock' | 'paper' | 'scissors'
type Outcome = 'win' | 'lose' | 'draw'

const MOVES: Move[] = ['rock', 'paper', 'scissors']
const ICON: Record<Move, string> = { rock: '✊', paper: '✋', scissors: '✌️' }

function judge(player: Move, cpu: Move): Outcome {
  if (player === cpu) return 'draw'
  if (
    (player === 'rock' && cpu === 'scissors') ||
    (player === 'paper' && cpu === 'rock') ||
    (player === 'scissors' && cpu === 'paper')
  ) return 'win'
  return 'lose'
}

export default function App() {
  const [player, setPlayer] = useState<Move | null>(null)
  const [cpu, setCpu] = useState<Move | null>(null)
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [score, setScore] = useState({ win: 0, lose: 0, draw: 0 })

  function play(move: Move) {
    const cpuMove = MOVES[Math.floor(Math.random() * 3)]
    const result = judge(move, cpuMove)
    setPlayer(move)
    setCpu(cpuMove)
    setOutcome(result)
    setScore(s => ({ ...s, [result]: s[result] + 1 }))
  }

  function reset() {
    setPlayer(null)
    setCpu(null)
    setOutcome(null)
    setScore({ win: 0, lose: 0, draw: 0 })
  }

  return (
    <div className="app">
      <h1>Rock · Paper · Scissors</h1>

      <div className="score">
        <span className="pill win">Wins {score.win}</span>
        <span className="pill draw">Draws {score.draw}</span>
        <span className="pill lose">Losses {score.lose}</span>
      </div>

      <div className="arena">
        <div className="side">
          <div className="label">You</div>
          <div className="hand">{player ? ICON[player] : '❔'}</div>
        </div>
        <div className="vs">vs</div>
        <div className="side">
          <div className="label">CPU</div>
          <div className="hand">{cpu ? ICON[cpu] : '❔'}</div>
        </div>
      </div>

      {outcome && (
        <div className={`outcome ${outcome}`}>
          {outcome === 'win' && 'You win!'}
          {outcome === 'lose' && 'You lose.'}
          {outcome === 'draw' && 'Draw.'}
        </div>
      )}

      <div className="controls">
        {MOVES.map(m => (
          <button key={m} className="move" onClick={() => play(m)}>
            <span className="moveIcon">{ICON[m]}</span>
            <span className="moveName">{m}</span>
          </button>
        ))}
      </div>

      <button className="reset" onClick={reset}>Reset</button>
    </div>
  )
}
