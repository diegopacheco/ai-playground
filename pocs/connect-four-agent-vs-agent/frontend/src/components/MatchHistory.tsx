import { useMatchHistory } from '../hooks/useMatchHistory'
import { getAgentInfo } from '../types'

export function MatchHistory() {
  const { data: matches, isLoading } = useMatchHistory()

  if (isLoading) {
    return <div className="text-center p-4">Loading history...</div>
  }

  if (!matches || matches.length === 0) {
    return (
      <div className="p-8 bg-gray-800 rounded-lg">
        <h2 className="text-xl font-bold mb-4">Match History</h2>
        <p className="text-gray-400">No matches played yet.</p>
      </div>
    )
  }

  return (
    <div className="p-8 bg-gray-800 rounded-lg overflow-x-auto">
      <h2 className="text-xl font-bold mb-4">Match History</h2>
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="py-3 px-4">Player X</th>
            <th className="py-3 px-4">Player O</th>
            <th className="py-3 px-4">Winner</th>
            <th className="py-3 px-4">Duration</th>
            <th className="py-3 px-4">Date</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((match) => {
            const agentAInfo = getAgentInfo(match.agent_a)
            const agentBInfo = getAgentInfo(match.agent_b)
            const winnerInfo = match.winner ? getAgentInfo(match.winner) : null
            const isWinnerA = match.winner === match.agent_a
            const isWinnerB = match.winner === match.agent_b
            return (
              <tr key={match.id} className="border-b border-gray-700 hover:bg-gray-700/50">
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div className={`w-4 h-4 rounded-full ${isWinnerA ? 'bg-green-500' : 'bg-red-500'}`} />
                    <div>
                      <div className={`font-semibold ${isWinnerA ? 'text-green-400' : ''}`} style={{ color: isWinnerA ? undefined : agentAInfo.color }}>
                        {agentAInfo.name}
                      </div>
                      <div className="text-xs text-gray-500 font-mono">{agentAInfo.model}</div>
                    </div>
                  </div>
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div className={`w-4 h-4 rounded-full ${isWinnerB ? 'bg-green-500' : 'bg-yellow-400'}`} />
                    <div>
                      <div className={`font-semibold ${isWinnerB ? 'text-green-400' : ''}`} style={{ color: isWinnerB ? undefined : agentBInfo.color }}>
                        {agentBInfo.name}
                      </div>
                      <div className="text-xs text-gray-500 font-mono">{agentBInfo.model}</div>
                    </div>
                  </div>
                </td>
                <td className="py-3 px-4">
                  {match.is_draw ? (
                    <span className="text-gray-400 font-semibold">Draw</span>
                  ) : winnerInfo && (
                    <div>
                      <div className="text-green-400 font-semibold">{winnerInfo.name}</div>
                      <div className="text-xs text-gray-500 font-mono">{winnerInfo.model}</div>
                    </div>
                  )}
                </td>
                <td className="py-3 px-4 text-gray-400 font-mono">
                  {match.duration_ms ? `${(match.duration_ms / 1000).toFixed(1)}s` : '-'}
                </td>
                <td className="py-3 px-4 text-gray-400 text-sm">
                  {new Date(match.started_at).toLocaleString()}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
