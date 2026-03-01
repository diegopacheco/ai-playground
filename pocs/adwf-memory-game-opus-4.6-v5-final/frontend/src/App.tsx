import { useState } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import PlayerForm from "./components/PlayerForm"
import GameBoard from "./components/GameBoard"
import Leaderboard from "./components/Leaderboard"

const queryClient = new QueryClient()

type Page = "home" | "game" | "leaderboard"

function AppContent() {
  const [page, setPage] = useState<Page>("home")
  const [gameId, setGameId] = useState<number | null>(null)
  const [playerId, setPlayerId] = useState<number | null>(null)

  const handleGameStart = (gId: number, pId: number) => {
    setGameId(gId)
    setPlayerId(pId)
    setPage("game")
  }

  const handlePlayAgain = async () => {
    if (playerId) {
      const { createGame } = await import("./api")
      const game = await createGame(playerId)
      setGameId(game.id)
      queryClient.invalidateQueries({ queryKey: ["game"] })
    }
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {page === "home" && (
        <div>
          <PlayerForm onGameStart={handleGameStart} />
          <div className="fixed bottom-6 left-0 right-0 flex justify-center">
            <button
              onClick={() => setPage("leaderboard")}
              data-testid="view-leaderboard-btn"
              className="px-6 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-xl transition-colors"
            >
              View Leaderboard
            </button>
          </div>
        </div>
      )}
      {page === "game" && gameId && (
        <GameBoard gameId={gameId} onPlayAgain={handlePlayAgain} />
      )}
      {page === "leaderboard" && (
        <Leaderboard onBack={() => setPage("home")} />
      )}
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  )
}
