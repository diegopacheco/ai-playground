"use client";

import { MOVES } from "@fly-zord/engine";
import type { Battle } from "@fly-zord/engine";

export function BattleLog({ battle }: { battle: Battle }) {
  const entries = [...battle.log].reverse().slice(0, 12);
  return (
    <div className="log">
      <div className="log-title">COMBAT FEED</div>
      {entries.length === 0 ? <div className="log-empty">the city holds its breath</div> : null}
      {entries.map(entry => (
        <div key={entry.turn} className={`log-row ${entry.side}`}>
          <span className="log-turn">T{entry.turn}</span>
          <span className="log-move">{battle[entry.side].name} · {MOVES[entry.move].name}</span>
          <span className="log-damage">{entry.damage > 0 ? `-${entry.damage}${entry.critical ? " CRIT" : ""}${entry.blocked ? " BLOCKED" : ""}` : "—"}</span>
          {entry.taunt ? <span className="log-taunt">“{entry.taunt}”</span> : null}
        </div>
      ))}
    </div>
  );
}
