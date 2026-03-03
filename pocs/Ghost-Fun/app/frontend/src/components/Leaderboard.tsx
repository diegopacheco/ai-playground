import { useQuery } from '@tanstack/react-query'
import { fetchLeaderboard } from '../api'

export default function Leaderboard() {
  const { data, isLoading } = useQuery({ queryKey: ['leaderboard'], queryFn: fetchLeaderboard, refetchInterval: 5000 })

  if (isLoading) return <div className="leaderboard"><p>Loading...</p></div>

  return (
    <div className="leaderboard">
      <h2>Leaderboard</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Player</th>
            <th>Moves</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {(data ?? []).map((score, i) => (
            <tr key={score.id}>
              <td>{i + 1}</td>
              <td>{score.player_name}</td>
              <td>{score.moves}</td>
              <td>{score.time_taken}s</td>
            </tr>
          ))}
          {(data ?? []).length === 0 && (
            <tr><td colSpan={4}>No scores yet</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
