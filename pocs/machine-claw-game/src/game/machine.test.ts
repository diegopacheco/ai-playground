import { describe, expect, test } from "bun:test";
import { type GameState, type Input, type Sensors, INITIAL_STATE, RULES, advance, awardPrize, command } from "./machine";

const NOTHING: Input = { moveX: 0, moveZ: 0, drop: false, coin: false };
const CLEAR: Sensors = { touching: false, atTop: false, atHome: false, gripClosed: false, gripOpen: true };
const DT = 1 / 60;

function playing(): GameState {
  return advance(INITIAL_STATE, { ...NOTHING, coin: true }, CLEAR, DT);
}

describe("coins", () => {
  test("the claw never plays for free", () => {
    let state = INITIAL_STATE;
    for (let i = 0; i < 600; i += 1) state = advance(state, { ...NOTHING, moveX: 1, drop: true }, CLEAR, DT);
    expect(state.phase).toBe("idle");
    expect(state.plays).toBe(0);
  });

  test("one coin buys exactly one try", () => {
    const state = playing();
    expect(state.phase).toBe("aiming");
    expect(state.credits).toBe(0);
    expect(state.plays).toBe(1);
    expect(state.timeLeft).toBe(RULES.roundSeconds);
  });

  test("a coin inserted during a try is kept for the next try", () => {
    const state = advance(playing(), { ...NOTHING, coin: true }, CLEAR, DT);
    expect(state.phase).toBe("aiming");
    expect(state.credits).toBe(1);
  });

  test("credits stop at the limit so coins are not eaten without bound", () => {
    let state = playing();
    for (let i = 0; i < 30; i += 1) state = advance(state, { ...NOTHING, coin: true }, CLEAR, DT);
    expect(state.credits).toBe(RULES.maxCredits);
  });
});

describe("round", () => {
  test("the claw drops by itself when the timer runs out", () => {
    let state = playing();
    for (let i = 0; i < RULES.roundSeconds * 60 + 5; i += 1) state = advance(state, NOTHING, CLEAR, DT);
    expect(state.phase).toBe("descending");
  });

  test("a round walks every phase in order and ends idle", () => {
    const seen: string[] = [];
    let state = playing();
    const script: readonly (readonly [Input, Sensors])[] = [
      [{ ...NOTHING, drop: true }, CLEAR],
      [NOTHING, { ...CLEAR, touching: true }],
      [NOTHING, { ...CLEAR, gripClosed: true, gripOpen: false }],
      [NOTHING, { ...CLEAR, atTop: true }],
      [NOTHING, { ...CLEAR, atHome: true }],
      [NOTHING, { ...CLEAR, gripOpen: true }],
    ];
    for (const [input, sensors] of script) {
      state = advance(state, input, sensors, 2);
      seen.push(state.phase);
    }
    expect(seen).toEqual(["descending", "closing", "lifting", "returning", "releasing", "idle"]);
  });

  test("the claw does not lift before the fingers are closed", () => {
    let state = advance(playing(), { ...NOTHING, drop: true }, CLEAR, DT);
    state = advance(state, NOTHING, { ...CLEAR, touching: true }, DT);
    state = advance(state, NOTHING, { ...CLEAR, gripClosed: false }, 5);
    expect(state.phase).toBe("closing");
  });

  test("the player cannot steer once the claw is dropping", () => {
    const order = command("descending", { ...NOTHING, moveX: 1, moveZ: 1 });
    expect(order.vx).toBe(0);
    expect(order.vz).toBe(0);
    expect(order.vy).toBeLessThan(0);
  });

  test("the claw holds the toy closed until it is over the chute", () => {
    expect(command("lifting", NOTHING).grip).toBe(1);
    expect(command("returning", NOTHING).grip).toBe(1);
    expect(command("releasing", NOTHING).grip).toBe(0);
  });

  test("a prize adds to the count without touching credits", () => {
    const state = awardPrize(playing());
    expect(state.prizes).toBe(1);
    expect(state.credits).toBe(0);
  });
});
