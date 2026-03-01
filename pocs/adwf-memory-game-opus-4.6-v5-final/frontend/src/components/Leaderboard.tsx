import { useQuery } from "@tanstack/react-query"
import { getLeaderboard } from "../api"

interface LeaderboardProps {
  onBack: () => void
}

export default function Leaderboard({ onBack }: LeaderboardProps) {
  const { data: entries, isLoading } = useQuery({
    queryKey: ["leaderboard"],
    queryFn: getLeaderboard,
  })

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4">
      <div className="bg-gray-800 rounded-2xl p-6 max-w-lg w-full shadow-2xl border border-gray-700" data-testid="leaderboard">
        <h2 className="text-2xl font-bold text-white text-center mb-6">Leaderboard</h2>
        {isLoading ? (
          <p className="text-gray-400 text-center">Loading...</p>
        ) : !entries || entries.length === 0 ? (
          <p className="text-gray-400 text-center">No scores yet</p>
        ) : (
          <table className="w-full text-left">
            <thead>
              <tr className="text-gray-400 text-sm uppercase tracking-wide border-b border-gray-700">
                <th className="pb-3 pr-2">#</th>
                <th className="pb-3">Player</th>
                <th className="pb-3 text-right">Score</th>
                <th className="pb-3 text-right">Moves</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry, i) => (
                <tr key={i} className="text-white border-b border-gray-700/50">
                  <td className="py-3 pr-2 text-gray-400">{i + 1}</td>
                  <td className="py-3 font-medium">{entry.player_name}</td>
                  <td className="py-3 text-right font-bold text-indigo-400">{entry.score}</td>
                  <td className="py-3 text-right">{entry.moves}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <button
          onClick={onBack}
          className="mt-6 w-full py-3 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-xl transition-colors"
        >
          Back
        </button>
      </div>
    </div>
  )
}
