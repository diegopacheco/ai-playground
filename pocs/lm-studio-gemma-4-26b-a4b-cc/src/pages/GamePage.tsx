import React, { useState } from 'react'
import { useGameHistory } from '../hooks/useGameHistory'
import { getGameResult } from '../logic/gameLogic'
import type { Move } from '../logic/gameLogic'
import { Link } from '@tanstack/react-router'

const moves: Move[] = ['Rock', 'Paper', 'Scissors']

export function GamePage() {
  const [lastResult, setLastResult] = useState<{
    playerMove: Move;
    computerMove: Move;
    result: string;
  } | null>(null)

  const { addResult } = useGameHistory()

  const play = (playerMove: Move) => {
    const computerMove = moves[Math.floor(Math.random() * moves.length)] as Move
    const result = getGameResult(playerMove, computerMove)

    const newResult = {
      playerMove,
      computerMove,
      result,
      timestamp: Date.now(),
    }

    setLastResult({
      playerMove,
      computerMove,
      result,
    })

    addResult(newResult)
  }

  return (
    <div style={{ padding: '20px' }}>
      <h1>Rock Paper Scissors</h1>
      <nav>
        <Link to="/history">View History</Link>
      </nav>
      <div style={{ marginTop: '20px' }}>
        {moves.map((move) => (
          <button key={move} onClick={() => play(move)} style={{ margin: '5px', padding: '10px 20px' }}>
            {move}
          </button>
        ))}
      </div>
      {lastResult && (
        <div style={{ marginTop: '20px', fontWeight: 'bold' }}>
          You chose {lastResult.playerMove}, Computer chose {lastResult.computerMove}. Result: {lastResult.result}
        </div>
      )}
    </div>
  )
}
