import { useState } from 'react'
import { AgentSelector } from './components/AgentSelector'
import { GameBoard } from './components/GameBoard'
import { MatchHistory } from './components/MatchHistory'
import { useGameStream } from './hooks/useGameStream'
import { startGame } from './api/client'

type View = 'select' | 'game' | 'history'

function App() {
  const [view, setView] = useState<View>('select')
  const [gameId, setGameId] = useState<string | null>(null)
  const [agentA, setAgentA] = useState('')
  const [agentB, setAgentB] = useState('')
  const gameState = useGameStream(gameId)

  const handleStartGame = async (a: string, b: string) => {
    setAgentA(a)
    setAgentB(b)
    const result = await startGame(a, b)
    setGameId(result.game_id)
    setView('game')
  }

  const handleNewGame = () => {
    setGameId(null)
    setView('select')
  }

  return (
    <div className="min-h-screen p-4">
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2">Connect Four</h1>
        <p className="text-gray-400">Agent vs Agent</p>
      </header>
      <nav className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setView('select')}
          className={`px-4 py-2 rounded ${view === 'select' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
        >
          New Game
        </button>
        <button
          onClick={() => setView('history')}
          className={`px-4 py-2 rounded ${view === 'history' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
        >
          History
        </button>
      </nav>
      <main className="max-w-4xl mx-auto">
        {view === 'select' && (
          <AgentSelector
            onStart={handleStartGame}
            disabled={gameId !== null && gameState.status === 'playing'}
          />
        )}
        {view === 'game' && gameId && (
          <div className="flex flex-col items-center gap-4">
            <GameBoard
              board={gameState.board}
              currentPlayer={gameState.currentPlayer}
              status={gameState.status}
              winner={gameState.winner}
              isDraw={gameState.isDraw}
              thinkingAgent={gameState.thinkingAgent}
              lastMove={gameState.lastMove}
              durationMs={gameState.durationMs}
              error={gameState.error}
              agentA={agentA}
              agentB={agentB}
            />
            {gameState.status === 'finished' && (
              <button
                onClick={handleNewGame}
                className="bg-green-600 hover:bg-green-700 text-white font-bold px-6 py-3 rounded"
              >
                Play Again
              </button>
            )}
          </div>
        )}
        {view === 'history' && <MatchHistory />}
      </main>
    </div>
  )
}

export default App
