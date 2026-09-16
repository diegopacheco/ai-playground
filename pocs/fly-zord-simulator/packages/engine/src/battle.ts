import { CRITICAL_CHANCE, CRITICAL_MULTIPLIER, ENERGY_PER_TURN, GUARD_REDUCTION, MAX_ENERGY, MAX_HP, MOVES, MOVE_IDS, VARIANCE } from "./moves.js";
import { random } from "./rng.js";
import type { Battle, MoveId, Pilot, Side, TurnEvent, Zord } from "./types.js";

export function opponentOf(side: Side): Side {
  return side === "left" ? "right" : "left";
}

function makeZord(name: string, palette: string, pilot: Pilot): Zord {
  return { name, palette, hp: MAX_HP, energy: 4, guarding: false, pilot };
}

export function createBattle(seed: number, leftPilot: Pilot, rightPilot: Pilot): Battle {
  return {
    seed,
    turn: 1,
    active: "left",
    left: makeZord("Thunder Titan", "crimson", leftPilot),
    right: makeZord("Dragon Sentinel", "cobalt", rightPilot),
    log: [],
    winner: "none"
  };
}

export function legalMoves(battle: Battle, side: Side): MoveId[] {
  const zord = battle[side];
  return MOVE_IDS.filter(id => MOVES[id].energy <= zord.energy);
}

function damageFor(power: number, roll: number, critical: boolean, blocked: boolean): number {
  const spread = 1 - VARIANCE + roll * VARIANCE * 2;
  const raw = power * spread * (critical ? CRITICAL_MULTIPLIER : 1) * (blocked ? 1 - GUARD_REDUCTION : 1);
  return Math.max(1, Math.round(raw));
}

export function applyMove(battle: Battle, moveId: MoveId, taunt = ""): Battle {
  if (battle.winner !== "none") throw new Error("the battle is already decided");
  const side = battle.active;
  if (!legalMoves(battle, side).includes(moveId)) throw new Error(`${moveId} needs more energy than the zord holds`);

  const move = MOVES[moveId];
  const attacker: Zord = { ...battle[side], guarding: false };
  const defenderSide = opponentOf(side);
  const defender = battle[defenderSide];

  const spreadRoll = random(battle.seed + battle.turn * 2);
  const criticalRoll = random(battle.seed + battle.turn * 2 + 1);
  const critical = move.kind === "attack" && criticalRoll < CRITICAL_CHANCE;
  const blocked = move.kind === "attack" && defender.guarding;
  const damage = move.kind === "attack" ? damageFor(move.power, spreadRoll, critical, blocked) : 0;

  const energyAfter = Math.min(MAX_ENERGY, attacker.energy - move.energy + (move.kind === "recharge" ? move.power : 0) + ENERGY_PER_TURN);
  const nextAttacker: Zord = { ...attacker, energy: energyAfter, guarding: move.kind === "defend" };
  const nextDefender: Zord = { ...defender, hp: Math.max(0, defender.hp - damage), guarding: defender.guarding && damage === 0 };

  const event: TurnEvent = { turn: battle.turn, side, move: moveId, damage, critical, blocked, energyAfter, taunt };
  const winner: Side | "none" = nextDefender.hp === 0 ? side : "none";

  return {
    ...battle,
    turn: battle.turn + 1,
    active: winner === "none" ? defenderSide : side,
    left: side === "left" ? nextAttacker : nextDefender,
    right: side === "right" ? nextAttacker : nextDefender,
    log: [...battle.log, event],
    winner
  };
}

export function describeBattle(battle: Battle, side: Side): string {
  const self = battle[side];
  const foe = battle[opponentOf(side)];
  const recent = battle.log.slice(-3).map(entry => `turn ${entry.turn}: ${entry.side} used ${entry.move} for ${entry.damage} damage`);
  return [
    `You pilot ${self.name} on the ${side} side. HP ${self.hp}/${MAX_HP}, energy ${self.energy}/${MAX_ENERGY}, guarding ${self.guarding}.`,
    `Enemy ${foe.name}. HP ${foe.hp}/${MAX_HP}, energy ${foe.energy}/${MAX_ENERGY}, guarding ${foe.guarding}.`,
    `Turn ${battle.turn}. Moves you can afford: ${legalMoves(battle, side).join(", ")}.`,
    recent.length > 0 ? `Recent: ${recent.join(" | ")}` : "Recent: the city is still quiet."
  ].join("\n");
}
