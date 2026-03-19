import { useEffect, useRef, useCallback, useState } from "react";
import type {
  CycleUpdate,
  GameStartEvent,
  GameOverEvent,
  MosquitoState,
  EggState,
  KillEvent,
  DateEvent,
  HatchEvent,
} from "~/types";

export interface GameState {
  status: "connecting" | "running" | "finished";
  cycle: number;
  gridSize: number;
  terminator: { x: number; y: number };
  mosquitos: MosquitoState[];
  eggs: EggState[];
  totalKills: number;
  totalHatched: number;
  totalDates: number;
  aliveCount: number;
  eggCount: number;
  winner: string | null;
  recentKills: KillEvent[];
  recentDates: DateEvent[];
  recentHatches: HatchEvent[];
  events: string[];
}

const initialState: GameState = {
  status: "connecting",
  cycle: 0,
  gridSize: 20,
  terminator: { x: 10, y: 10 },
  mosquitos: [],
  eggs: [],
  totalKills: 0,
  totalHatched: 0,
  totalDates: 0,
  aliveCount: 0,
  eggCount: 0,
  winner: null,
  recentKills: [],
  recentDates: [],
  recentHatches: [],
  events: [],
};

export function useGameSSE(gameId: string | null) {
  const [state, setState] = useState<GameState>(initialState);
  const eventSourceRef = useRef<EventSource | null>(null);

  const addEvent = useCallback((msg: string) => {
    setState((prev) => ({
      ...prev,
      events: [`[${String(prev.cycle).padStart(3, "0")}] ${msg}`, ...prev.events].slice(0, 50),
    }));
  }, []);

  useEffect(() => {
    if (!gameId) return;
    setState(initialState);

    const es = new EventSource(`/api/games/${gameId}/stream`);
    eventSourceRef.current = es;

    es.addEventListener("game_start", (e) => {
      const data: GameStartEvent = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        status: "running",
        gridSize: data.grid_size,
        terminator: data.terminator,
        mosquitos: data.mosquitos.map((m) => ({ ...m, age: 0 })),
        aliveCount: data.mosquitos.length,
        events: [`Game started! ${data.terminator_agent} vs ${data.mosquito_agent}`],
      }));
    });

    es.addEventListener("cycle_update", (e) => {
      const data: CycleUpdate = JSON.parse(e.data);
      setState((prev) => {
        const newEvents = [...prev.events];
        for (const k of data.kills) {
          if (k.killed_mosquitos.length > 0)
            newEvents.unshift(`[${String(data.cycle).padStart(3, "0")}] TERMINATED ${k.killed_mosquitos.length} mosquito(s)!`);
          if (k.killed_eggs.length > 0)
            newEvents.unshift(`[${String(data.cycle).padStart(3, "0")}] Destroyed ${k.killed_eggs.length} egg(s)!`);
        }
        for (const d of data.dates) {
          newEvents.unshift(`[${String(data.cycle).padStart(3, "0")}] Mosquitos dating at (${d.position.x},${d.position.y})`);
        }
        for (const h of data.hatches) {
          newEvents.unshift(`[${String(data.cycle).padStart(3, "0")}] Egg hatched at (${h.position.x},${h.position.y})`);
        }
        for (const d of data.deaths) {
          newEvents.unshift(`[${String(data.cycle).padStart(3, "0")}] ${d} died of old age`);
        }
        return {
          ...prev,
          status: "running",
          cycle: data.cycle,
          terminator: data.terminator,
          mosquitos: data.mosquitos,
          eggs: data.eggs,
          totalKills: data.total_kills,
          totalHatched: data.total_hatched,
          totalDates: data.total_dates,
          aliveCount: data.alive_count,
          eggCount: data.egg_count,
          recentKills: data.kills,
          recentDates: data.dates,
          recentHatches: data.hatches,
          events: newEvents.slice(0, 50),
        };
      });
    });

    es.addEventListener("game_over", (e) => {
      const data: GameOverEvent = JSON.parse(e.data);
      setState((prev) => ({
        ...prev,
        status: "finished",
        winner: data.winner,
        events: [
          `GAME OVER! Winner: ${data.winner.toUpperCase()} after ${data.cycles} cycles`,
          ...prev.events,
        ].slice(0, 50),
      }));
      es.close();
    });

    es.onerror = () => {
      es.close();
    };

    return () => {
      es.close();
    };
  }, [gameId]);

  return state;
}
