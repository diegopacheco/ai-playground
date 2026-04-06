
import React from 'react'

export function HistoryPage() {
  const { history } = useGameHistory()

  return (
    <div style={{ padding: '20px' }}>
      <h1>Game History</h1>
      <nav>
        <Link to="/">Play Again</Link>
      </nav>
      {history.length === 0 ? (
        <p>No games played yet.</p>
      ) : (
        <ul style={{ marginTop: '20px' }}>
          {history.map((game) => (
            <li key={game.timestamp} style={{ marginBottom: '10px' }}>
              {new Date(game.timestamp).toLocaleTimeString()} - You: {game.playerMove}, Computer: {game.computerMove} ({game.result})
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
