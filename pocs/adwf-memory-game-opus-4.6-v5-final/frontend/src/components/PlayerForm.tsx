import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { createPlayer, createGame } from "../api"
import type { Game } from "../types"

interface PlayerFormProps {
  onGameStart: (gameId: number, playerId: number) => void
}

export default function PlayerForm({ onGameStart }: PlayerFormProps) {
  const [name, setName] = useState("")

  const mutation = useMutation({
    mutationFn: async (playerName: string) => {
      const player = await createPlayer(playerName)
      const game = await createGame(player.id)
      return { player, game } as { player: { id: number }; game: Game }
    },
    onSuccess: (data) => {
      onGameStart(data.game.id, data.player.id)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (name.trim()) {
      mutation.mutate(name.trim())
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      <div className="bg-gray-800 rounded-2xl p-8 max-w-sm w-full mx-4 shadow-2xl border border-gray-700">
        <h1 className="text-3xl font-bold text-white text-center mb-2">Memory Game</h1>
        <p className="text-gray-400 text-center mb-6">Enter your name to start</p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Your name"
            data-testid="player-name-input"
            className="w-full px-4 py-3 bg-gray-700 text-white rounded-xl border border-gray-600 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
          />
          <button
            type="submit"
            disabled={!name.trim() || mutation.isPending}
            data-testid="start-game-btn"
            className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-colors"
          >
            {mutation.isPending ? "Starting..." : "Start Game"}
          </button>
          {mutation.isError && (
            <p className="text-red-400 text-sm text-center">{mutation.error.message}</p>
          )}
        </form>
      </div>
    </div>
  )
}
