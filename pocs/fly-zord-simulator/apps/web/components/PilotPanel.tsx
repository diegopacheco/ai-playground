"use client";

import { MAX_ENERGY, MAX_HP, MOVES, opponentOf } from "@fly-zord/engine";
import type { Battle, Side } from "@fly-zord/engine";
import type { Action } from "../lib/battlefield";
import { CAM_HEIGHT, CAM_WIDTH, renderCockpit } from "../lib/cockpit";
import type { CamView } from "../lib/cockpit";
import { VIEW_HEIGHT, VIEW_WIDTH, renderOnboard } from "../lib/onboard";
import { LOOP_STAGES, STRIP_HEIGHT, STRIP_WIDTH, WING_BEAT, loopStage, renderNeural } from "../lib/neural";
import { PixelCanvas } from "./PixelCanvas";

export interface Telemetry {
  readonly source: string;
  readonly reason: string;
  readonly taunt: string;
  readonly elapsedMs: number | null;
}

export interface PilotPanelProps {
  readonly battle: Battle;
  readonly side: Side;
  readonly view: CamView;
  readonly action: Action | null;
  readonly telemetry: Telemetry | null;
}

function Meter({ label, value, max, tone }: { label: string; value: number; max: number; tone: string }) {
  return (
    <div className="meter">
      <span className="meter-label">{label}</span>
      <div className="meter-bar"><span style={{ width: `${Math.max(0, Math.min(100, (value / max) * 100))}%`, background: tone }} /></div>
      <span className="meter-value">{value}</span>
    </div>
  );
}

export function PilotPanel({ battle, side, view, action, telemetry }: PilotPanelProps) {
  const zord = battle[side];
  const foe = battle[opponentOf(side)];
  const lastMove = [...battle.log].reverse().find(entry => entry.side === side);
  return (
    <section className="pilot-panel" style={{ borderColor: view.accent }}>
      <header className="pilot-head">
        <span style={{ color: view.accent }}>{zord.name}</span>
        <span className="pilot-seat">{zord.pilot.name} · {zord.pilot.kind === "fly" ? zord.pilot.provider : "human"}</span>
      </header>
      <div className="pilot-views">
        <figure>
          <PixelCanvas width={CAM_WIDTH} height={CAM_HEIGHT} className="pixel-view" draw={(ctx, frame) => renderCockpit(ctx, view, frame)} />
          <figcaption>DRIVER · {zord.pilot.name}</figcaption>
        </figure>
        <figure>
          <PixelCanvas width={VIEW_WIDTH} height={VIEW_HEIGHT} className="pixel-view" draw={(ctx, frame) => renderOnboard(ctx, battle, side, action, frame, view.accent)} />
          <figcaption>ONBOARD · WHAT THE {zord.pilot.kind === "fly" ? "FLY" : "RANGER"} SEES</figcaption>
        </figure>
      </div>
      <div className="telemetry">
        <Meter label="hull" value={zord.hp} max={MAX_HP} tone="#ff5a5a" />
        <Meter label="core" value={zord.energy} max={MAX_ENERGY} tone="#3de0a0" />
        <Meter label="threat" value={foe.energy} max={MAX_ENERGY} tone="#ffe14d" />
        <Meter label="target" value={foe.hp} max={MAX_HP} tone="#5aa8ff" />
        <div className="telemetry-rows">
          <div><span>shield</span><b style={{ color: zord.guarding ? "#3de0e0" : "#5a5a80" }}>{zord.guarding ? "raised" : "down"}</b></div>
          <div><span>last</span><b>{lastMove ? MOVES[lastMove.move].name : "none"}</b></div>
          <div><span>link</span><b>{telemetry?.source ?? (zord.pilot.kind === "human" ? "hands on the sticks" : "waiting")}</b></div>
          <div><span>think</span><b>{telemetry?.elapsedMs !== null && telemetry?.elapsedMs !== undefined ? `${telemetry.elapsedMs} ms` : "—"}</b></div>
        </div>
        <div className="neural">
          <div className="neural-head">
            <span>wing beat</span>
            <b style={{ color: view.accent }}>{WING_BEAT[view.mood]} Hz</b>
          </div>
          <PixelCanvas width={STRIP_WIDTH} height={STRIP_HEIGHT} className="pixel-view strip" draw={(ctx, frame) => renderNeural(ctx, view.mood, view.accent, frame)} />
          <div className="loop">
            {LOOP_STAGES.map((stage, index) => (
              <span key={stage} className={index === loopStage(view.mood) ? "loop-step live" : "loop-step"}>{stage}</span>
            ))}
          </div>
        </div>
        <p className="pilot-reason">{telemetry?.reason || "no order sent yet"}</p>
        <p className="pilot-taunt">{telemetry?.taunt ? `“${telemetry.taunt}”` : ""}</p>
      </div>
    </section>
  );
}
