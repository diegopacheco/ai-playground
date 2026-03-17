import { useState } from 'react'
import { createGame } from '../api/games'

const SETTINGS = [
  { name: 'Medieval Fantasy', icon: '&#9876;' },
  { name: 'Dark Dungeon Crawl', icon: '&#128420;' },
  { name: 'Sci-Fi Space Station', icon: '&#128640;' },
  { name: 'Post-Apocalyptic Wasteland', icon: '&#9762;' },
  { name: 'Pirate Adventure', icon: '&#9875;' },
  { name: 'Haunted Castle', icon: '&#127984;' },
  { name: 'Ancient Egypt', icon: '&#9883;' },
  { name: 'Cyberpunk City', icon: '&#9889;' },
]

interface Props {
  onGameCreated: (id: string) => void
}

export default function GameSetup({ onGameCreated }: Props) {
  const [playerName, setPlayerName] = useState('')
  const [setting, setSetting] = useState(SETTINGS[0].name)
  const [loading, setLoading] = useState(false)

  const handleStart = async () => {
    if (!playerName.trim()) return
    setLoading(true)
    const id = await createGame(playerName.trim(), setting)
    onGameCreated(id)
  }

  return (
    <div className="flex flex-col items-center mt-12 fade-in">
      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-amber-500/20 to-amber-900/20 border border-amber-500/20 flex items-center justify-center mb-8">
        <span className="text-4xl" dangerouslySetInnerHTML={{ __html: '&#9876;' }} />
      </div>

      <h2
        className="text-4xl font-bold text-amber-200/90 mb-2"
        style={{ fontFamily: 'Cinzel, serif' }}
      >
        Begin Your Quest
      </h2>
      <p className="text-amber-500/50 mb-12 text-sm">Choose your path, adventurer. The Dungeon Master awaits.</p>

      <div className="w-full max-w-lg space-y-8">
        <div>
          <label className="block text-amber-300/70 mb-2 text-xs font-semibold uppercase tracking-wider">Character Name</label>
          <input
            type="text"
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            placeholder="Enter your name..."
            className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-amber-100 placeholder-white/20 focus:outline-none focus:border-amber-500/50 focus:bg-white/8 transition-all duration-200 text-sm"
          />
        </div>

        <div>
          <label className="block text-amber-300/70 mb-3 text-xs font-semibold uppercase tracking-wider">Setting</label>
          <div className="grid grid-cols-2 gap-2">
            {SETTINGS.map((s) => (
              <button
                key={s.name}
                onClick={() => setSetting(s.name)}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl text-left text-sm transition-all duration-200 ${
                  setting === s.name
                    ? 'bg-amber-500/15 border border-amber-500/40 text-amber-200'
                    : 'bg-white/3 border border-white/8 text-white/50 hover:bg-white/6 hover:text-white/70'
                }`}
              >
                <span className="text-lg" dangerouslySetInnerHTML={{ __html: s.icon }} />
                <span>{s.name}</span>
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleStart}
          disabled={loading || !playerName.trim()}
          className="w-full py-3.5 rounded-xl bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-white font-semibold text-sm transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-amber-900/30"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Summoning the Dungeon Master...
            </span>
          ) : 'Start Adventure'}
        </button>
      </div>
    </div>
  )
}
