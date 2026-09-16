"use client";

import { useRef } from "react";
import { MAX_ENERGY, MAX_HP } from "@fly-zord/engine";
import type { Battle, Side } from "@fly-zord/engine";
import { ACTION_FRAMES, HEIGHT, WIDTH, renderBattle } from "../lib/battlefield";
import type { Action } from "../lib/battlefield";
import { PixelCanvas } from "./PixelCanvas";

export interface BattleStageProps {
  readonly battle: Battle;
  readonly action: Action | null;
  readonly onActionDone: () => void;
}

function Gauges({ battle, side }: { battle: Battle; side: Side }) {
  const zord = battle[side];
  return (
    <div className={`gauges ${side}`}>
      <div className="gauge-name">{zord.name}</div>
      <div className="bar hp"><span style={{ width: `${(zord.hp / MAX_HP) * 100}%` }} /></div>
      <div className="bar energy"><span style={{ width: `${(zord.energy / MAX_ENERGY) * 100}%` }} /></div>
      <div className="gauge-read">HP {zord.hp} · CORE {zord.energy}</div>
    </div>
  );
}

export function BattleStage({ battle, action, onActionDone }: BattleStageProps) {
  const timeline = useRef({ action: null as Action | null, start: 0, done: true });
  const doneRef = useRef(onActionDone);
  doneRef.current = onActionDone;

  return (
    <div className="stage">
      <div className="stage-top">
        <Gauges battle={battle} side="left" />
        <Gauges battle={battle} side="right" />
      </div>
      <PixelCanvas
        width={WIDTH}
        height={HEIGHT}
        className="stage-canvas"
        draw={(ctx, frame) => {
          if (timeline.current.action !== action) timeline.current = { action, start: frame, done: action === null };
          const local = frame - timeline.current.start;
          renderBattle(ctx, battle, action, action ? Math.min(local, ACTION_FRAMES) : frame);
          if (action && !timeline.current.done && local >= ACTION_FRAMES) {
            timeline.current.done = true;
            doneRef.current();
          }
        }}
      />
    </div>
  );
}
