"use client";

import type { Side } from "@fly-zord/engine";
import { PILOT_CHOICES } from "../lib/pilots";

export interface SeatConfig {
  readonly choice: string;
  readonly model: string;
}

export interface SetupProps {
  readonly seats: Readonly<Record<Side, SeatConfig>>;
  readonly seed: number;
  readonly onSeat: (side: Side, config: SeatConfig) => void;
  readonly onSeed: (seed: number) => void;
  readonly onStart: () => void;
}

function Seat({ side, config, onSeat }: { side: Side; config: SeatConfig; onSeat: (side: Side, config: SeatConfig) => void }) {
  const choice = PILOT_CHOICES.find(entry => entry.id === config.choice) ?? PILOT_CHOICES[0]!;
  return (
    <div className={`seat ${side}`}>
      <div className="seat-title">{side === "left" ? "Thunder Titan" : "Dragon Sentinel"}</div>
      <select
        value={config.choice}
        onChange={event => {
          const picked = PILOT_CHOICES.find(entry => entry.id === event.target.value) ?? PILOT_CHOICES[0]!;
          onSeat(side, { choice: picked.id, model: picked.model });
        }}
      >
        {PILOT_CHOICES.map(entry => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
      </select>
      <input
        value={config.model}
        disabled={choice.kind === "human" || choice.id === "instinct"}
        onChange={event => onSeat(side, { choice: config.choice, model: event.target.value })}
      />
    </div>
  );
}

export function Setup({ seats, seed, onSeat, onSeed, onStart }: SetupProps) {
  return (
    <div className="setup">
      <div className="seats">
        <Seat side="left" config={seats.left} onSeat={onSeat} />
        <Seat side="right" config={seats.right} onSeat={onSeat} />
      </div>
      <div className="setup-row">
        <label htmlFor="seed">CITY SEED</label>
        <input id="seed" type="number" value={seed} onChange={event => onSeed(Number(event.target.value) || 0)} />
        <button className="start" onClick={onStart}>DROP THE ZORDS</button>
      </div>
    </div>
  );
}
