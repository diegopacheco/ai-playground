import { useState } from 'react'
import GameSetup from './components/GameSetup'
import GamePlay from './components/GamePlay'
import GameHistory from './components/GameHistory'

type Screen = 'setup' | 'play' | 'history'

function App() {
  const [screen, setScreen] = useState<Screen>('setup')
  const [gameId, setGameId] = useState<string>('')

  const onGameCreated = (id: string) => {
    setGameId(id)
    setScreen('play')
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      <nav className="border-b border-amber-900/30 bg-[#0d0d14] px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1
            className="text-2xl font-bold text-amber-400 cursor-pointer"
            onClick={() => setScreen('setup')}
          >
            Dungeon Master AI
          </h1>
          <div className="flex gap-4">
            <button
              onClick={() => setScreen('setup')}
              className={`px-4 py-2 rounded transition-colors ${screen === 'setup' ? 'bg-amber-700 text-white' : 'text-amber-400 hover:bg-amber-900/30'}`}
            >
              New Game
            </button>
            <button
              onClick={() => setScreen('history')}
              className={`px-4 py-2 rounded transition-colors ${screen === 'history' ? 'bg-amber-700 text-white' : 'text-amber-400 hover:bg-amber-900/30'}`}
            >
              History
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto p-6">
        {screen === 'setup' && <GameSetup onGameCreated={onGameCreated} />}
        {screen === 'play' && <GamePlay gameId={gameId} />}
        {screen === 'history' && <GameHistory onResume={(id) => { setGameId(id); setScreen('play') }} />}
      </main>
    </div>
  )
}

export default App
