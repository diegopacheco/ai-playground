"use client";

import { useCallback, useEffect, useState } from "react";
import { applyMove, createBattle, legalMoves, opponentOf } from "@fly-zord/engine";
import type { Battle, MoveId, Side } from "@fly-zord/engine";
import type { Action } from "../lib/battlefield";
import type { CamView, Mood } from "../lib/cockpit";
import { PILOT_CHOICES, makePilot } from "../lib/pilots";
import { BattleStage } from "../components/BattleStage";
import { BattleLog } from "../components/BattleLog";
import { PilotPanel } from "../components/PilotPanel";
import { Controls } from "../components/Controls";
import { Setup } from "../components/Setup";
import type { SeatConfig } from "../components/Setup";
import type { Telemetry } from "../components/PilotPanel";

type Phase = "setup" | "human" | "thinking" | "animating" | "over";

interface Decision {
  readonly move: MoveId;
  readonly taunt: string;
  readonly reason: string;
  readonly source: string;
  readonly elapsedMs?: number;
}

const ACCENTS: Readonly<Record<Side, string>> = { left: "#ff5a5a", right: "#5aa8ff" };

export default function Page() {
  const [seats, setSeats] = useState<Record<Side, SeatConfig>>({
    left: { choice: "human", model: "human" },
    right: { choice: "instinct", model: "instinct" }
  });
  const [seed, setSeed] = useState(1987);
  const [battle, setBattle] = useState<Battle | null>(null);
  const [pending, setPending] = useState<Battle | null>(null);
  const [action, setAction] = useState<Action | null>(null);
  const [phase, setPhase] = useState<Phase>("setup");
  const [decisions, setDecisions] = useState<Partial<Record<Side, Decision>>>({});
  const [error, setError] = useState("");

  const start = useCallback(() => {
    const fresh = createBattle(seed, makePilot("left", seats.left.choice, seats.left.model), makePilot("right", seats.right.choice, seats.right.model));
    setBattle(fresh);
    setPending(null);
    setAction(null);
    setDecisions({});
    setError("");
    setPhase(fresh.left.pilot.kind === "human" ? "human" : "thinking");
  }, [seats, seed]);

  const commit = useCallback((current: Battle, move: MoveId, taunt: string) => {
    const next = applyMove(current, move, taunt);
    const event = next.log[next.log.length - 1]!;
    setPending(next);
    setAction({ side: event.side, move: event.move, damage: event.damage, critical: event.critical, blocked: event.blocked });
    setPhase("animating");
  }, []);

  useEffect(() => {
    if (phase !== "thinking" || !battle) return;
    const side = battle.active;
    const pilot = battle[side].pilot;
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch("/api/pilot", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ battle, side, provider: pilot.provider, model: pilot.model })
        });
        const decision = (await response.json()) as Decision;
        if (cancelled) return;
        if (!decision.move) throw new Error("the fly sent nothing back");
        setDecisions(current => ({ ...current, [side]: decision }));
        commit(battle, decision.move, decision.taunt);
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : String(failure));
      }
    })();
    return () => { cancelled = true; };
  }, [phase, battle, commit]);

  const finishAnimation = useCallback(() => {
    if (!pending) return;
    setBattle(pending);
    setPending(null);
    setAction(null);
    setPhase(pending.winner !== "none" ? "over" : pending[pending.active].pilot.kind === "human" ? "human" : "thinking");
  }, [pending]);

  if (!battle) {
    return (
      <main className="arena">
        <header className="head">
          <h1>FLY ZORD SIMULATOR</h1>
          <p>two megazords, two cockpits, one city block left standing</p>
        </header>
        <Setup
          seats={seats}
          seed={seed}
          onSeat={(side, config) => setSeats(current => ({ ...current, [side]: config }))}
          onSeed={setSeed}
          onStart={start}
        />
        <Rules />
      </main>
    );
  }

  const camFor = (side: Side): CamView => {
    const zord = battle[side];
    const isActing = action?.side === side;
    const isStruck = action !== null && action.side !== side && action.damage > 0;
    const mood: Mood =
      battle.winner === opponentOf(side) || pending?.winner === opponentOf(side) ? "down" :
      phase === "thinking" && battle.active === side ? "thinking" :
      phase === "animating" && isActing ? "acting" :
      phase === "animating" && isStruck ? "hit" : "idle";
    return { kind: zord.pilot.kind, mood, accent: ACCENTS[side], label: `${zord.pilot.name} ${zord.pilot.kind === "fly" ? "FLY" : "CAM"}` };
  };

  const telemetryFor = (side: Side): Telemetry | null => {
    const decision = decisions[side];
    if (!decision) return null;
    return { source: decision.source, reason: decision.reason, taunt: decision.taunt, elapsedMs: decision.elapsedMs ?? null };
  };

  const humanTurn = phase === "human" && battle.winner === "none";
  const activePilot = battle[battle.active].pilot;
  const hint =
    phase === "over" ? `${battle[battle.winner === "none" ? battle.active : battle.winner].name} is the last one standing` :
    humanTurn ? "your zord waits for an order" :
    phase === "thinking" ? `${activePilot.name} is reading the battlefield through ${activePilot.provider}` :
    "the city shakes";

  return (
    <main className="arena playing">
      <header className="head bar">
        <h1>FLY ZORD SIMULATOR</h1>
        <span className="head-state">TURN {battle.turn} · {phase.toUpperCase()}</span>
        <span className="head-hint">{hint}</span>
        <button className="start" onClick={start}>REMATCH</button>
        <button className="ghost" onClick={() => { setBattle(null); setPhase("setup"); }}>PILOTS</button>
      </header>
      <div className="deck">
        <PilotPanel battle={battle} side="left" view={camFor("left")} action={action} telemetry={telemetryFor("left")} />
        <div className="stage-column">
          <BattleStage battle={battle} action={action} onActionDone={finishAnimation} />
          <Controls legal={legalMoves(battle, battle.active)} enabled={humanTurn} onMove={move => commit(battle, move, "")} />
        </div>
        <PilotPanel battle={battle} side="right" view={camFor("right")} action={action} telemetry={telemetryFor("right")} />
      </div>
      {error
        ? <div className="error">{error} <button onClick={() => { setError(""); setPhase("thinking"); }}>try again</button></div>
        : <BattleLog battle={battle} />}
    </main>
  );
}

function Rules() {
  return (
    <div className="rules">
      <div className="log-title">HOW A DUEL RUNS</div>
      <ul>
        <li>Turns alternate. Every turn feeds one unit back into the core.</li>
        <li>Piston Jab is free, Sky Saber costs 3, Missile Barrage costs 5.</li>
        <li>Titan Guard cuts the next hit by 60 percent and then drops.</li>
        <li>A fly pilot is a coding agent: {PILOT_CHOICES.filter(choice => choice.kind === "fly").map(choice => choice.id).join(", ")}.</li>
        <li>If a provider CLI is missing the fly falls back to instinct so the duel never stalls.</li>
      </ul>
    </div>
  );
}
