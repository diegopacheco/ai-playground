import type { Move, MoveId } from "./types.js";

export const MAX_HP = 100;
export const MAX_ENERGY = 10;
export const ENERGY_PER_TURN = 1;
export const GUARD_REDUCTION = 0.6;
export const CRITICAL_CHANCE = 0.12;
export const CRITICAL_MULTIPLIER = 1.6;
export const VARIANCE = 0.15;

export const MOVES: Readonly<Record<MoveId, Move>> = {
  jab: { id: "jab", name: "Piston Jab", kind: "attack", energy: 0, power: 8, frames: 6 },
  saber: { id: "saber", name: "Sky Saber", kind: "attack", energy: 3, power: 18, frames: 8 },
  missiles: { id: "missiles", name: "Missile Barrage", kind: "attack", energy: 5, power: 26, frames: 10 },
  guard: { id: "guard", name: "Titan Guard", kind: "defend", energy: 0, power: 0, frames: 5 },
  charge: { id: "charge", name: "Core Charge", kind: "recharge", energy: 0, power: 4, frames: 5 }
};

export const MOVE_IDS: readonly MoveId[] = ["jab", "saber", "missiles", "guard", "charge"];

export function isMoveId(value: string): value is MoveId {
  return (MOVE_IDS as readonly string[]).includes(value);
}
