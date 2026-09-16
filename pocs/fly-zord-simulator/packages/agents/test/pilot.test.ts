import { expect, test } from "bun:test";
import { MOVE_IDS, createBattle, legalMoves } from "@fly-zord/engine";
import type { Battle, MoveId, Pilot } from "@fly-zord/engine";
import { OllamaAgent } from "../src/agent-sdk.js";
import { buildPrompt, flyDecision, instinct, parseDecision } from "../src/pilot.js";

const fly: Pilot = { name: "Buzz", kind: "fly", provider: "ollama", model: "llama3.2" };
const base = createBattle(7, fly, fly);

function withZords(left: Partial<Battle["left"]>, right: Partial<Battle["right"]>): Battle {
  return { ...base, left: { ...base.left, ...left }, right: { ...base.right, ...right } };
}

test("the fly is told only the moves its core can pay for", () => {
  const prompt = buildPrompt(withZords({ energy: 1 }, {}), "left");
  expect(prompt).toContain("jab (attack, energy 0, power 8)");
  expect(prompt).not.toContain("missiles");
  expect(prompt).toContain('{"move"');
});

test("a move the fly cannot afford is refused instead of crashing the turn", () => {
  const drained = withZords({ energy: 0 }, {});
  expect(parseDecision('{"move":"missiles"}', legalMoves(drained, "left"), "test")).toBeNull();
  expect(parseDecision('{"move":"teleport"}', MOVE_IDS, "test")).toBeNull();
});

test("a decision is read out of whatever chatter the cli prints around it", () => {
  const output = 'thinking...\n{"move":"saber","taunt":"Bzzt!","reason":"open target"}\ndone\n';
  const decision = parseDecision(output, MOVE_IDS, "ollama:llama3.2");
  expect(decision?.move).toBe("saber");
  expect(decision?.taunt).toBe("Bzzt!");
  expect(decision?.source).toBe("ollama:llama3.2");
});

test("the fly is asked through the provider cli with the model the player picked", () => {
  const commands: string[][] = [];
  const runner = (command: string[]) => {
    commands.push(command);
    return '{"move":"guard","taunt":"Shields!","reason":"hull is thin"}';
  };
  const decision = flyDecision(base, "left", "ollama", "llama3.2", () => new OllamaAgent(runner));
  expect(commands[0]?.slice(0, 3)).toEqual(["ollama", "run", "llama3.2"]);
  expect(commands[0]?.[3]).toContain("Thunder Titan");
  expect(decision.move).toBe("guard");
  expect(decision.source).toBe("ollama:llama3.2");
});

test("a missing cli drops the fly onto instinct and says so instead of stalling the fight", () => {
  const broken = () => new OllamaAgent(() => { throw new Error("ollama is not installed"); });
  const decision = flyDecision(base, "left", "ollama", "llama3.2", broken);
  expect(decision.move).toBe(instinct(base, "left").move);
  expect(decision.source).toContain("instinct");
  expect(decision.source).toContain("ollama");
});

test("an unknown provider still flies the zord on instinct", () => {
  expect(flyDecision(base, "left", "instinct", "instinct").source).toBe("instinct");
});

test("instinct raises the guard when the hull is thin and the enemy holds a barrage", () => {
  expect(instinct(withZords({ hp: 20 }, { energy: 6 }), "left").move).toBe("guard");
});

test("instinct feeds the core rather than spending it on a raised shield", () => {
  expect(instinct(withZords({ energy: 8 }, { guarding: true }), "left").move).toBe("charge");
});

test("instinct fires the barrage the moment the core can pay and the enemy is open", () => {
  expect(instinct(withZords({ energy: 5 }, {}), "left").move).toBe("missiles");
});

test("instinct never picks a move the core cannot pay for, whatever the state", () => {
  for (let hp = 5; hp <= 100; hp += 19) {
    for (let energy = 0; energy <= 10; energy += 1) {
      for (const guarding of [false, true]) {
        const state = withZords({ hp, energy }, { guarding, energy: 10 - energy });
        const move: MoveId = instinct(state, "left").move;
        expect(`${hp}/${energy}/${guarding}:${legalMoves(state, "left").includes(move)}`).toBe(`${hp}/${energy}/${guarding}:true`);
      }
    }
  }
});
