import React from 'react';
import { useGameHistory } from '../hooks/useGameHistory';

export const HistoryPage: React.FC = () => {
  const { history, clearHistory } = useGameHistory();

  return (
    <div style={{ padding: '20px' }}>
      <h1>Game History</h1>
      <button onClick={clearHistory}>Clear History</button>
      {history.length === 0 ? (
        <p>No games played yet.</p>
      ) : (
        <ul>
          {history.map((game) => (
            <li key={game.id}>
              {new Date(game.timestamp).toLocaleString()} - You: {game.playerMove}, Computer: {game.computerMove} -> <strong>{game.result}</strong>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
