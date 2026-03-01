import { useState, useCallback } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { getGame, flipCard } from "../api"
import type { FlipResponse } from "../types"
import Card from "./Card"
import ScoreBoard from "./ScoreBoard"
import GameOver from "./GameOver"

interface GameBoardProps {
  gameId: number
  onPlayAgain: () => void
}

export default function GameBoard({ gameId, onPlayAgain }: GameBoardProps) {
  const [flipping, setFlipping] = useState(false)
  const queryClient = useQueryClient()

  const { data: game, isLoading } = useQuery({
    queryKey: ["game", gameId],
    queryFn: () => getGame(gameId),
    refetchInterval: false,
  })

  const flipMutation = useMutation({
    mutationFn: (position: number) => flipCard(gameId, position),
    onSuccess: (data: FlipResponse) => {
      queryClient.setQueryData(["game", gameId], data.game)

      const flippedNotMatched = data.game.board.filter(
        (c) => c.flipped && !c.matched
      )

      if (flippedNotMatched.length === 2) {
        setFlipping(true)
        setTimeout(async () => {
          const refreshed = await getGame(gameId)
          queryClient.setQueryData(["game", gameId], refreshed)
          setFlipping(false)
        }, 1000)
      }
    },
  })

  const handleCardClick = useCallback(
    (position: number) => {
      if (flipping || flipMutation.isPending) return
      flipMutation.mutate(position)
    },
    [flipping, flipMutation]
  )

  if (isLoading || !game) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-white text-xl">Loading...</div>
      </div>
    )
  }

  const isComplete = game.status === "completed"

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4 gap-6">
      <h1 className="text-3xl font-bold text-white">Memory Game</h1>
      <ScoreBoard
        moves={game.moves}
        matchesFound={game.matches_found}
        totalPairs={game.total_pairs}
        isComplete={isComplete}
      />
      <div className="grid grid-cols-4 gap-3 max-w-md w-full" data-testid="game-board">
        {game.board.map((card, index) => (
          <Card
            key={card.position}
            card={card}
            index={index}
            onClick={() => handleCardClick(card.position)}
            disabled={flipping || flipMutation.isPending}
          />
        ))}
      </div>
      {isComplete && (
        <GameOver
          score={game.score}
          moves={game.moves}
          onPlayAgain={onPlayAgain}
        />
      )}
    </div>
  )
}
