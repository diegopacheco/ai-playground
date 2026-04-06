import React, { useState } from 'react';
import { Move, Result, getResult, getRandomMove } from '../logic/gameLogic';
import { useGameHistory } from '../hooks/useGameHistory';

export const GamePage: React.FC = () => {
  const [lastResult, setLastResult] = useState<{ player: Move; computer: Move; result: Result } | null>(null);
  const { addResult } = useGameHistory();

  const play = (playerMove: Move) => {
    const computerMove = getRandomMove();
    const result = getResult(playerMove, computerMove);
    setLastResult({ player: playerMove, computer: computerMove, result });
    
    addResult({
      id: crypto.randomUUID(),
      playerMove,
      computerMove,
      result,
      timestamp: Date.now(),
    });
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h1>Rock Paper Scissors</h1>
      <div style={{ marginBottom: '20px' }}>
        {(['Rock', 'Paper', 'Scissors'] as const).map((move) => (
          <button key={move} onClick={() => play(move)} style={{ margin: '5px', padding: '10px 20px' }}>
            {move}
          </button>
        ))}
      </div>
      {lastResult && (
        <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px' }}>
          <p>You: {lastResult.player}</p>
          <p>Computer: {lastResult.computer}</p>
          <p><strong>Result: {lastResult.result}</strong></p>
        </div>
      )}
    </div>
  );
};
