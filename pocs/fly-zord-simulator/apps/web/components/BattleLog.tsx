"use client";

import { MOVES } from "@fly-zord/engine";
import type { Battle } from "@fly-zord/engine";

export function BattleLog({ battle }: { battle: Battle }) {
  const entries = [...battle.log].reverse().slice(0, 5);
  return (
    <div className="ticker">
      <span className="ticker-title">FEED</span>
      {entries.length === 0 ? <span className="ticker-empty">the city holds its breath</span> : null}
      {entries.map(entry => (
        <span key={entry.turn} className={`ticker-item ${entry.side}`}>
          <b>T{entry.turn}</b> {MOVES[entry.move].name}
          <i>{entry.damage > 0 ? ` -${entry.damage}${entry.critical ? "!" : ""}${entry.blocked ? " blocked" : ""}` : ""}</i>
          {entry.taunt ? <em>“{entry.taunt}”</em> : null}
        </span>
      ))}
    </div>
  );
}
