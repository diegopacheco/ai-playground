"use client";

import { MOVES, MOVE_IDS } from "@fly-zord/engine";
import type { MoveId } from "@fly-zord/engine";

export interface ControlsProps {
  readonly legal: readonly MoveId[];
  readonly enabled: boolean;
  readonly hint: string;
  readonly onMove: (move: MoveId) => void;
}

export function Controls({ legal, enabled, hint, onMove }: ControlsProps) {
  return (
    <div className="controls">
      <div className="hint">{hint}</div>
      <div className="move-row">
        {MOVE_IDS.map(id => (
          <button key={id} className={`move ${MOVES[id].kind}`} disabled={!enabled || !legal.includes(id)} onClick={() => onMove(id)}>
            <span className="move-name">{MOVES[id].name}</span>
            <span className="move-cost">{MOVES[id].kind === "recharge" ? `+${MOVES[id].power} core` : `${MOVES[id].energy} core`}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
