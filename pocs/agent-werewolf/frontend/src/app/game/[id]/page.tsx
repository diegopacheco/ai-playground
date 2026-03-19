"use client";
import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import { Game } from "@/types";
import { getGame } from "@/lib/api";
import GameLive from "@/components/GameLive";
import GameReplay from "@/components/GameReplay";

export default function GamePage() {
  const params = useParams();
  const gameId = params.id as string;
  const [game, setGame] = useState<Game | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getGame(gameId)
      .then(setGame)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [gameId]);

  if (loading) return <div className="text-gray-500">Loading...</div>;
  if (!game) return <div className="text-red-400">Game not found.</div>;

  if (game.status === "finished") {
    return <GameReplay game={game} />;
  }

  return <GameLive gameId={gameId} />;
}
