"use client";
import { useState, useEffect, useRef } from "react";
import { getStreamUrl } from "@/lib/api";

export interface GameEvent {
  event: string;
  data: Record<string, unknown>;
  timestamp: number;
}

export function useGameSSE(gameId: string | null) {
  const [events, setEvents] = useState<GameEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!gameId) return;

    const es = new EventSource(getStreamUrl(gameId));
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    const eventTypes = [
      "game_start", "night_phase", "agent_thinking", "elimination",
      "day_phase", "discussion", "voting_phase", "vote", "vote_result",
      "role_reveal", "game_over",
    ];

    eventTypes.forEach((type) => {
      es.addEventListener(type, (e: MessageEvent) => {
        const data = JSON.parse(e.data);
        const event: GameEvent = { event: type, data, timestamp: Date.now() };
        setEvents((prev) => [...prev, event]);
        if (type === "game_over") {
          setGameOver(true);
          es.close();
        }
      });
    });

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [gameId]);

  return { events, connected, gameOver };
}
