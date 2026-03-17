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
    <div className="min-h-screen bg-gradient-to-b from-[#1a1220] via-[#1e1528] to-[#1a1220]">
      <nav className="backdrop-blur-md bg-[#1a1220]/80 border-b border-amber-900/20 px-6 py-3 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <h1
            className="text-xl font-semibold cursor-pointer flex items-center gap-2"
            style={{ fontFamily: 'Cinzel, serif' }}
            onClick={() => setScreen('setup')}
          >
            <span className="text-amber-400/90">Dungeon Master</span>
            <span className="text-[10px] font-medium tracking-widest text-amber-600/60 uppercase mt-1">AI</span>
          </h1>
          <div className="flex gap-1">
            <button
              onClick={() => setScreen('setup')}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                screen === 'setup'
                  ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                  : 'text-amber-500/60 hover:text-amber-400 hover:bg-white/5'
              }`}
            >
              New Game
            </button>
            <button
              onClick={() => setScreen('history')}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                screen === 'history'
                  ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                  : 'text-amber-500/60 hover:text-amber-400 hover:bg-white/5'
              }`}
            >
              History
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {screen === 'setup' && <GameSetup onGameCreated={onGameCreated} />}
        {screen === 'play' && <GamePlay gameId={gameId} />}
        {screen === 'history' && <GameHistory onResume={(id) => { setGameId(id); setScreen('play') }} />}
      </main>
    </div>
  )
}

export default App
