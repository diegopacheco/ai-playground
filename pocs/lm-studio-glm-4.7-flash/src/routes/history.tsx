import { createFileRoute, Link } from '@tanstack/react-router'

type Result = 'win' | 'lose' | 'tie'

export const Route = createFileRoute('/history')({
  component: History,
})

function History() {
  const history = JSON.parse(localStorage.getItem('gameHistory') || '[]')

  const getResultColor = (result: Result) => {
    if (result === 'win') return 'green'
    if (result === 'lose') return 'red'
    return 'gray'
  }

  const getResultText = (result: Result) => {
    if (result === 'win') return 'You win!'
    if (result === 'lose') return 'You lose!'
    return "It's a tie!"
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Game History</h1>
      <p>View all your past games</p>

      {history.length === 0 ? (
        <p>No games played yet. Go to the Play page to start!</p>
      ) : (
        <div style={{ marginTop: '2rem' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #ddd' }}>
                <th style={{ padding: '1rem', textAlign: 'left' }}>Player</th>
                <th style={{ padding: '1rem', textAlign: 'left' }}>Computer</th>
                <th style={{ padding: '1rem', textAlign: 'left' }}>Result</th>
              </tr>
            </thead>
            <tbody>
              {history.map((game: any, index: number) => (
                <tr key={index} style={{ borderBottom: '1px solid #ddd' }}>
                  <td style={{ padding: '1rem' }}>{game.player}</td>
                  <td style={{ padding: '1rem' }}>{game.computer}</td>
                  <td style={{ padding: '1rem', color: getResultColor(game.result) }}>
                    {getResultText(game.result)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={{ marginTop: '2rem' }}>
        <Link to="/" style={{ padding: '0.5rem 1rem', cursor: 'pointer', textDecoration: 'none' }}>
          Back to Game
        </Link>
      </div>
    </div>
  )
}