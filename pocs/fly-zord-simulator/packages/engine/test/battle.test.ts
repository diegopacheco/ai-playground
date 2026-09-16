import { expect, test } from "bun:test";
import { MAX_ENERGY, MAX_HP, MOVES, applyMove, createBattle, describeBattle, legalMoves } from "../src/index.js";
import type { Battle, MoveId, Pilot } from "../src/index.js";

const human: Pilot = { name: "Pilot", kind: "human", provider: "human", model: "human" };
const fly: Pilot = { name: "Buzz", kind: "fly", provider: "instinct", model: "instinct" };

function battleWith(changes: Partial<Battle>): Battle {
  return { ...createBattle(7, human, fly), ...changes };
}

function run(battle: Battle, moves: MoveId[]): Battle {
  return moves.reduce((state, move) => applyMove(state, move), battle);
}

test("a move the core cannot pay for is never offered and never accepted", () => {
  const drained = battleWith({ left: { ...createBattle(7, human, fly).left, energy: 2 } });
  expect(legalMoves(drained, "left")).toEqual(["jab", "guard", "charge"]);
  expect(() => applyMove(drained, "missiles")).toThrow(/more energy/);
});

test("raising the guard cuts the next hit and the shield drops once it absorbs one", () => {
  const guarded = applyMove(createBattle(11, human, fly), "guard");
  expect(guarded.left.guarding).toBe(true);
  const hit = applyMove(guarded, "jab");
  expect(hit.log[1]?.blocked).toBe(true);
  expect(hit.left.hp).toBeGreaterThan(MAX_HP - MOVES.jab.power);
  expect(hit.left.guarding).toBe(false);
});

test("charging the core is what makes a barrage affordable", () => {
  const start = battleWith({ left: { ...createBattle(7, human, fly).left, energy: 0 } });
  expect(legalMoves(start, "left")).not.toContain("missiles");
  const charged = run(start, ["charge", "jab"]);
  expect(legalMoves(charged, "left")).toContain("missiles");
});

test("every turn feeds the core one unit and never past the cap", () => {
  const start = battleWith({ left: { ...createBattle(7, human, fly).left, energy: MAX_ENERGY } });
  const after = applyMove(start, "jab");
  expect(after.left.energy).toBe(MAX_ENERGY);
  expect(applyMove(after, "jab").right.energy).toBe(5);
});

test("the battle stops the moment a zord falls so no turn lands after the win", () => {
  const nearlyDead = battleWith({ right: { ...createBattle(7, human, fly).right, hp: 1 } });
  const finished = applyMove(nearlyDead, "jab");
  expect(finished.winner).toBe("left");
  expect(finished.right.hp).toBe(0);
  expect(() => applyMove(finished, "jab")).toThrow(/already decided/);
});

test("one seed replays one battle so a fly rematch is judged on its choices alone", () => {
  const script: MoveId[] = ["charge", "guard", "missiles", "jab", "saber", "missiles"];
  const first = run(createBattle(4242, fly, fly), script);
  const second = run(createBattle(4242, fly, fly), script);
  const other = run(createBattle(9, fly, fly), script);
  expect(first.log).toEqual(second.log);
  expect(first.log).not.toEqual(other.log);
});

test("the briefing a fly reads names only the moves it can afford right now", () => {
  const drained = battleWith({ left: { ...createBattle(7, human, fly).left, energy: 1 } });
  const briefing = describeBattle(drained, "left");
  expect(briefing).toContain("Moves you can afford: jab, guard, charge");
  expect(briefing).toContain("Thunder Titan");
  expect(briefing).toContain("Dragon Sentinel");
});

test("turns alternate so neither zord can act twice in a row", () => {
  const first = applyMove(createBattle(7, human, fly), "jab");
  expect(first.active).toBe("right");
  expect(applyMove(first, "jab").active).toBe("left");
});
