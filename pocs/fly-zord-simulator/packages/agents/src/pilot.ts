import { MAX_ENERGY, MOVES, describeBattle, isMoveId, legalMoves, opponentOf } from "@fly-zord/engine";
import type { Battle, MoveId, Side } from "@fly-zord/engine";
import { AgyAgent, ClaudeCodeAgent, CodexAgent, OllamaAgent, jsonObject } from "./agent-sdk.js";
import type { Agent } from "./agent-sdk.js";

export const INSTINCT = "instinct";

export interface Decision {
  readonly move: MoveId;
  readonly taunt: string;
  readonly reason: string;
  readonly source: string;
}

export type AgentFactory = () => Agent;

export const PROVIDERS: Readonly<Record<string, AgentFactory>> = {
  claude: () => new ClaudeCodeAgent(),
  codex: () => new CodexAgent(),
  agy: () => new AgyAgent(),
  ollama: () => new OllamaAgent()
};

export function buildPrompt(battle: Battle, side: Side): string {
  const catalog = legalMoves(battle, side).map(id => `${id} (${MOVES[id].kind}, energy ${MOVES[id].energy}, power ${MOVES[id].power})`);
  return [
    "You are a house fly strapped into the cockpit of a giant battle zord in an 8-bit city brawl.",
    "You command the zord with tiny levers and you never leave the cockpit.",
    describeBattle(battle, side),
    `Pick exactly one move from: ${catalog.join("; ")}.`,
    "Answer with one JSON object and nothing else:",
    '{"move":"<move id>","taunt":"<up to 8 words shouted through the cockpit speaker>","reason":"<up to 15 words>"}'
  ].join("\n");
}

export function parseDecision(output: string, legal: readonly MoveId[], source: string): Decision | null {
  const data = jsonObject(output);
  const move = data.move;
  if (typeof move !== "string" || !isMoveId(move) || !legal.includes(move)) return null;
  const taunt = typeof data.taunt === "string" ? data.taunt.trim().slice(0, 60) : "";
  const reason = typeof data.reason === "string" ? data.reason.trim().slice(0, 120) : "";
  return { move, taunt, reason, source };
}

export function instinct(battle: Battle, side: Side): Decision {
  const self = battle[side];
  const foe = battle[opponentOf(side)];
  const legal = legalMoves(battle, side);
  const pick = (id: MoveId, reason: string): Decision => ({ move: id, taunt: TAUNTS[id], reason, source: INSTINCT });

  if (self.hp <= 25 && foe.energy >= MOVES.missiles.energy && !self.guarding) return pick("guard", "a barrage is loaded and my hull is thin");
  if (legal.includes("missiles") && !foe.guarding) return pick("missiles", "full energy and an open target");
  if (legal.includes("saber") && !foe.guarding && self.energy >= MOVES.saber.energy + 1) return pick("saber", "cheap damage while the core refills");
  if (foe.guarding && self.energy < MAX_ENERGY) return pick("charge", "hitting a raised shield wastes the core");
  if (self.energy < MOVES.saber.energy) return pick("charge", "the core is too cold to swing");
  return pick("jab", "close range and free to throw");
}

const TAUNTS: Readonly<Record<MoveId, string>> = {
  jab: "Bzzt! Take the piston!",
  saber: "Sky saber, slice him!",
  missiles: "Every tube, fire!",
  guard: "Shields up, little wings!",
  charge: "Feeding the core, hold on!"
};

export function flyDecision(battle: Battle, side: Side, provider: string, model: string, factory: AgentFactory | undefined = PROVIDERS[provider]): Decision {
  if (!factory || provider === INSTINCT) return instinct(battle, side);
  const legal = legalMoves(battle, side);
  try {
    const output = factory().call(model, buildPrompt(battle, side));
    const decision = parseDecision(output, legal, `${provider}:${model}`);
    if (decision) return decision;
  } catch {
  }
  return { ...instinct(battle, side), source: `${INSTINCT} (${provider} unavailable)` };
}
