import { useEffect, useRef, useState } from "react";
import type { GameState } from "~/hooks/useGameSSE";

interface Props {
  state: GameState;
}

interface CellEffect {
  type: "explosion" | "hearts" | "hatch";
  x: number;
  y: number;
  id: string;
}

export default function GameGrid({ state }: Props) {
  const [effects, setEffects] = useState<CellEffect[]>([]);
  const prevCycle = useRef(0);

  useEffect(() => {
    if (state.cycle === prevCycle.current) return;
    prevCycle.current = state.cycle;

    const newEffects: CellEffect[] = [];
    for (const k of state.recentKills) {
      newEffects.push({
        type: "explosion",
        x: k.position.x,
        y: k.position.y,
        id: `exp-${state.cycle}-${k.position.x}-${k.position.y}`,
      });
    }
    for (const d of state.recentDates) {
      newEffects.push({
        type: "hearts",
        x: d.position.x,
        y: d.position.y,
        id: `heart-${state.cycle}-${d.position.x}-${d.position.y}`,
      });
    }
    for (const h of state.recentHatches) {
      newEffects.push({
        type: "hatch",
        x: h.position.x,
        y: h.position.y,
        id: `hatch-${state.cycle}-${h.position.x}-${h.position.y}`,
      });
    }

    if (newEffects.length > 0) {
      setEffects(newEffects);
      setTimeout(() => setEffects([]), 800);
    }
  }, [state.cycle, state.recentKills, state.recentDates, state.recentHatches]);

  const cellSize = Math.min(28, Math.floor(600 / state.gridSize));

  const mosquitoMap = new Map<string, number>();
  for (const m of state.mosquitos) {
    const key = `${m.x},${m.y}`;
    mosquitoMap.set(key, (mosquitoMap.get(key) || 0) + 1);
  }

  const eggMap = new Map<string, number>();
  for (const e of state.eggs) {
    const key = `${e.x},${e.y}`;
    eggMap.set(key, (eggMap.get(key) || 0) + 1);
  }

  const effectMap = new Map<string, CellEffect>();
  for (const e of effects) {
    effectMap.set(`${e.x},${e.y}`, e);
  }

  return (
    <div className="inline-block bg-[#0d0d0d] border border-gray-800 rounded-lg p-2">
      {Array.from({ length: state.gridSize }, (_, y) => (
        <div key={y} className="flex">
          {Array.from({ length: state.gridSize }, (_, x) => {
            const key = `${x},${y}`;
            const isTerminator =
              state.terminator.x === x && state.terminator.y === y;
            const mosquitoCount = mosquitoMap.get(key) || 0;
            const eggCount = eggMap.get(key) || 0;
            const effect = effectMap.get(key);

            return (
              <div
                key={key}
                className={`grid-cell relative flex items-center justify-center border border-[#1a1a1a] ${
                  isTerminator ? "terminator-glow bg-red-950/30" : ""
                }`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  fontSize: cellSize * 0.6,
                }}
              >
                {isTerminator && (
                  <span className="relative z-10" title="Terminator">
                    🤖
                  </span>
                )}
                {!isTerminator && mosquitoCount > 0 && (
                  <span
                    className="mosquito-flutter relative z-10"
                    title={`${mosquitoCount} mosquito(s)`}
                  >
                    🦟
                    {mosquitoCount > 1 && (
                      <span
                        className="absolute -top-1 -right-1 bg-green-600 text-white rounded-full flex items-center justify-center font-bold"
                        style={{ fontSize: cellSize * 0.3, width: cellSize * 0.4, height: cellSize * 0.4 }}
                      >
                        {mosquitoCount}
                      </span>
                    )}
                  </span>
                )}
                {!isTerminator && mosquitoCount === 0 && eggCount > 0 && (
                  <span className="egg-pulse" title={`${eggCount} egg(s)`}>
                    🥚
                    {eggCount > 1 && (
                      <span
                        className="absolute -top-1 -right-1 bg-yellow-600 text-white rounded-full flex items-center justify-center font-bold"
                        style={{ fontSize: cellSize * 0.3, width: cellSize * 0.4, height: cellSize * 0.4 }}
                      >
                        {eggCount}
                      </span>
                    )}
                  </span>
                )}
                {effect && (
                  <span
                    className={`absolute inset-0 flex items-center justify-center z-20 pointer-events-none ${
                      effect.type === "explosion"
                        ? "explosion-effect"
                        : effect.type === "hearts"
                          ? "hearts-effect"
                          : "hatch-effect"
                    }`}
                    style={{ fontSize: cellSize * 0.8 }}
                  >
                    {effect.type === "explosion"
                      ? "💥"
                      : effect.type === "hearts"
                        ? "💕"
                        : "🐣"}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
