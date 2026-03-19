"use client";
import { useParams } from "next/navigation";
import GameLive from "@/components/GameLive";

export default function GamePage() {
  const params = useParams();
  const gameId = params.id as string;

  return <GameLive gameId={gameId} />;
}
