import { useState, useEffect } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { useStore } from './useStore.js'
import { startBattle, recordRound } from './store.js'
import './Battle.css'

function Battle() {
  const navigate = useNavigate()
  const state = useStore()
  const [animating, setAnimating] = useState(false)
  const [roundResult, setRoundResult] = useState(null)
  const [showChampion, setShowChampion] = useState(false)

  useEffect(() => {
    if (state.phase === 'setup' && state.player1.cards.length > 0) {
      startBattle()
    }
  }, [])

  const p1Card = state.player1.cards[state.currentRound]
  const p2Card = state.player2.cards[state.currentRound]

  const handleFight = () => {
    if (animating || !p1Card || !p2Card) return
    setAnimating(true)
    setRoundResult(null)

    setTimeout(() => {
      const winner = p1Card.power > p2Card.power ? 1 : p2Card.power > p1Card.power ? 2 : 0
      recordRound(p1Card, p2Card, winner)
      setRoundResult({ winner, p1Card, p2Card })
      setAnimating(false)

      if (state.currentRound >= 2) {
        setTimeout(() => setShowChampion(true), 1500)
      }
    }, 2000)
  }

  if (state.player1.cards.length === 0) {
    return (
      <div className="battle-page">
        <h2>No cards selected!</h2>
        <button className="action-btn battle-btn" onClick={() => navigate({ to: '/cards' })}>
          Go pick cards first
        </button>
      </div>
    )
  }

  const champion = state.player1.score > state.player2.score ? state.player1.name :
    state.player2.score > state.player1.score ? state.player2.name : 'Draw'

  return (
    <div className="battle-page">
      <div className="scoreboard">
        <div className="score p1-score">
          <span>{state.player1.name}</span>
          <span className="score-num">{state.player1.score}</span>
        </div>
        <div className="round-info">Round {Math.min(state.currentRound + 1, 3)} / 3</div>
        <div className="score p2-score">
          <span>{state.player2.name}</span>
          <span className="score-num">{state.player2.score}</span>
        </div>
      </div>

      {showChampion ? (
        <div className="champion-screen">
          <h2 className="champion-title">
            {champion === 'Draw' ? "It's a Draw!" : `${champion} is the Champion!`}
          </h2>
          <div className="trophy">🏆</div>
          <button className="action-btn battle-btn" onClick={() => navigate({ to: '/history' })}>
            View History
          </button>
        </div>
      ) : (
        <>
          <div className="arena">
            <div className={`fighter left ${animating ? 'attack-left' : ''}`}>
              {p1Card && (
                <div className="fighter-card">
                  <img src={p1Card.sprite} alt={p1Card.name} />
                  <h3>{p1Card.name}</h3>
                  <span className="power-badge">Power: {p1Card.power}</span>
                </div>
              )}
            </div>

            <div className="vs-battle">
              {animating ? (
                <div className="clash-effect">CLASH!</div>
              ) : roundResult ? (
                <div className="round-result">
                  {roundResult.winner === 0 ? 'TIE!' :
                    roundResult.winner === 1 ? `${state.player1.name} wins!` :
                      `${state.player2.name} wins!`}
                </div>
              ) : (
                <span>VS</span>
              )}
            </div>

            <div className={`fighter right ${animating ? 'attack-right' : ''}`}>
              {p2Card && (
                <div className="fighter-card">
                  <img src={p2Card.sprite} alt={p2Card.name} />
                  <h3>{p2Card.name}</h3>
                  <span className="power-badge">Power: {p2Card.power}</span>
                </div>
              )}
            </div>
          </div>

          {state.phase !== 'finished' && (
            <button
              className={`action-btn fight-btn ${animating ? 'disabled' : ''}`}
              onClick={handleFight}
              disabled={animating}
            >
              {animating ? 'Fighting...' : 'FIGHT!'}
            </button>
          )}
        </>
      )}

      {state.rounds.length > 0 && !showChampion && (
        <div className="round-history">
          {state.rounds.map((r, i) => (
            <div key={i} className="round-entry">
              <span>Round {r.round}: {r.p1Card.name} vs {r.p2Card.name} -
                {r.winner === 0 ? ' Tie' : r.winner === 1 ? ` ${state.player1.name}` : ` ${state.player2.name}`}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default Battle
