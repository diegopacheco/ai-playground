import { useState, useEffect } from 'react'
import { listGames } from '../api/games'

interface Game {
  id: string
  player_name: string
  setting: string
  status: string
  created_at: string
}

interface Props {
  onResume: (id: string) => void
}

export default function GameHistory({ onResume }: Props) {
  const [games, setGames] = useState<Game[]>([])

  useEffect(() => {
    listGames().then(setGames)
  }, [])

  return (
    <div>
      <h2 className="text-3xl font-bold text-amber-300 mb-6">Past Adventures</h2>

      {games.length === 0 ? (
        <p className="text-amber-700">No adventures yet. Start a new quest!</p>
      ) : (
        <div className="space-y-3">
          {games.map((g) => (
            <div
              key={g.id}
              onClick={() => onResume(g.id)}
              className="bg-[#3a2815] border border-amber-800/40 rounded-lg p-4 cursor-pointer hover:border-amber-500/50 transition-colors"
            >
              <div className="flex justify-between items-center">
                <div>
                  <span className="text-amber-200 font-bold">{g.player_name}</span>
                  <span className="text-amber-700 mx-2">-</span>
                  <span className="text-amber-500">{g.setting}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-xs px-2 py-1 rounded ${g.status === 'active' ? 'bg-emerald-900/40 text-emerald-400' : 'bg-amber-900/40 text-amber-500'}`}>
                    {g.status}
                  </span>
                  <span className="text-amber-700 text-sm">
                    {new Date(g.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
