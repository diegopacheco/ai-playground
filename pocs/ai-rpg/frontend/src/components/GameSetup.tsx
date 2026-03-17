import { useState } from 'react'
import { createGame } from '../api/games'

const SETTINGS = [
  'Medieval Fantasy',
  'Dark Dungeon Crawl',
  'Sci-Fi Space Station',
  'Post-Apocalyptic Wasteland',
  'Pirate Adventure',
  'Haunted Castle',
  'Ancient Egypt',
  'Cyberpunk City',
]

interface Props {
  onGameCreated: (id: string) => void
}

export default function GameSetup({ onGameCreated }: Props) {
  const [playerName, setPlayerName] = useState('')
  const [setting, setSetting] = useState(SETTINGS[0])
  const [loading, setLoading] = useState(false)

  const handleStart = async () => {
    if (!playerName.trim()) return
    setLoading(true)
    const id = await createGame(playerName.trim(), setting)
    onGameCreated(id)
  }

  return (
    <div className="flex flex-col items-center mt-16">
      <div className="text-6xl mb-6">&#9876;</div>
      <h2 className="text-4xl font-bold text-amber-300 mb-2">Begin Your Quest</h2>
      <p className="text-amber-700 mb-10">The Dungeon Master awaits your arrival...</p>

      <div className="w-full max-w-md space-y-6">
        <div>
          <label className="block text-amber-200 mb-2 text-sm">Character Name</label>
          <input
            type="text"
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            placeholder="Enter your name, adventurer..."
            className="w-full px-4 py-3 rounded bg-[#3a2815] border border-amber-800/50 text-amber-100 placeholder-amber-900/60 focus:outline-none focus:border-amber-500"
          />
        </div>

        <div>
          <label className="block text-amber-200 mb-2 text-sm">Setting</label>
          <select
            value={setting}
            onChange={(e) => setSetting(e.target.value)}
            className="w-full px-4 py-3 rounded bg-[#3a2815] border border-amber-800/50 text-amber-100 focus:outline-none focus:border-amber-500"
          >
            {SETTINGS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <button
          onClick={handleStart}
          disabled={loading || !playerName.trim()}
          className="w-full py-3 rounded bg-amber-700 hover:bg-amber-600 text-white font-bold text-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Summoning the Dungeon Master...' : 'Start Adventure'}
        </button>
      </div>
    </div>
  )
}
