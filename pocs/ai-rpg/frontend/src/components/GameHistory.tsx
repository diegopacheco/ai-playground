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
    <div className="fade-in">
      <h2
        className="text-3xl font-bold text-amber-200/90 mb-8"
        style={{ fontFamily: 'Cinzel, serif' }}
      >
        Past Adventures
      </h2>

      {games.length === 0 ? (
        <div className="text-center py-20">
          <div className="text-4xl mb-4 opacity-30">&#9876;</div>
          <p className="text-white/30 text-sm">No adventures yet. Start a new quest!</p>
        </div>
      ) : (
        <div className="space-y-3">
          {games.map((g) => (
            <div
              key={g.id}
              onClick={() => onResume(g.id)}
              className="bg-[#1e1528]/80 backdrop-blur border border-white/8 rounded-xl p-4 cursor-pointer hover:border-amber-500/30 hover:bg-white/5 transition-all duration-200 group"
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
                    <span className="text-amber-400/60 text-sm font-bold">{g.player_name.charAt(0).toUpperCase()}</span>
                  </div>
                  <div>
                    <span className="text-amber-200/90 font-medium text-sm">{g.player_name}</span>
                    <div className="text-white/30 text-xs mt-0.5">{g.setting}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-[10px] font-semibold uppercase tracking-wider px-2.5 py-1 rounded-lg ${
                    g.status === 'active'
                      ? 'bg-emerald-500/15 text-emerald-400/80'
                      : 'bg-white/5 text-white/30'
                  }`}>
                    {g.status}
                  </span>
                  <span className="text-white/20 text-xs">
                    {new Date(g.created_at).toLocaleDateString()}
                  </span>
                  <svg className="w-4 h-4 text-white/20 group-hover:text-amber-400/60 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
