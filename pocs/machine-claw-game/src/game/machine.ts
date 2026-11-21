export type Phase = "idle" | "aiming" | "descending" | "closing" | "lifting" | "returning" | "releasing";

export type GameState = {
  readonly phase: Phase;
  readonly credits: number;
  readonly timeLeft: number;
  readonly phaseTime: number;
  readonly prizes: number;
  readonly plays: number;
};

export type Input = {
  readonly moveX: number;
  readonly moveZ: number;
  readonly drop: boolean;
  readonly coin: boolean;
};

export type Sensors = {
  readonly touching: boolean;
  readonly atTop: boolean;
  readonly atHome: boolean;
  readonly gripClosed: boolean;
  readonly gripOpen: boolean;
};

export type ClawCommand = {
  readonly vx: number;
  readonly vy: number;
  readonly vz: number;
  readonly grip: number;
  readonly homing: boolean;
};

export const RULES = {
  roundSeconds: 20,
  maxCredits: 9,
  moveSpeed: 0.38,
  descendSpeed: 0.4,
  liftSpeed: 0.32,
  returnSpeed: 0.3,
  closeHold: 0.35,
  releaseHold: 1.4,
} as const;

export const INITIAL_STATE: GameState = {
  phase: "idle",
  credits: 0,
  timeLeft: 0,
  phaseTime: 0,
  prizes: 0,
  plays: 0,
};

function enter(state: GameState, phase: Phase): GameState {
  return { ...state, phase, phaseTime: 0 };
}

export function advance(state: GameState, input: Input, sensors: Sensors, dt: number): GameState {
  const credits = input.coin ? Math.min(state.credits + 1, RULES.maxCredits) : state.credits;
  const s: GameState = { ...state, credits, phaseTime: state.phaseTime + dt };
  switch (s.phase) {
    case "idle":
      return s.credits > 0
        ? { ...enter(s, "aiming"), credits: s.credits - 1, timeLeft: RULES.roundSeconds, plays: s.plays + 1 }
        : s;
    case "aiming": {
      const timeLeft = Math.max(0, s.timeLeft - dt);
      const timed: GameState = { ...s, timeLeft };
      return input.drop || timeLeft === 0 ? enter(timed, "descending") : timed;
    }
    case "descending":
      return sensors.touching ? enter(s, "closing") : s;
    case "closing":
      return sensors.gripClosed && s.phaseTime >= RULES.closeHold ? enter(s, "lifting") : s;
    case "lifting":
      return sensors.atTop ? enter(s, "returning") : s;
    case "returning":
      return sensors.atHome ? enter(s, "releasing") : s;
    case "releasing":
      return sensors.gripOpen && s.phaseTime >= RULES.releaseHold ? enter(s, "idle") : s;
    default: {
      const unreachable: never = s.phase;
      return unreachable;
    }
  }
}

export function awardPrize(state: GameState): GameState {
  return { ...state, prizes: state.prizes + 1 };
}

const STILL = { vx: 0, vy: 0, vz: 0, homing: false } as const;

export function command(phase: Phase, input: Input): ClawCommand {
  switch (phase) {
    case "idle":
      return { ...STILL, grip: 0, homing: true };
    case "aiming":
      return { vx: input.moveX * RULES.moveSpeed, vy: 0, vz: input.moveZ * RULES.moveSpeed, grip: 0, homing: false };
    case "descending":
      return { ...STILL, vy: -RULES.descendSpeed, grip: 0 };
    case "closing":
      return { ...STILL, grip: 1 };
    case "lifting":
      return { ...STILL, vy: RULES.liftSpeed, grip: 1 };
    case "returning":
      return { ...STILL, grip: 1, homing: true };
    case "releasing":
      return { ...STILL, grip: 0, homing: true };
    default: {
      const unreachable: never = phase;
      return unreachable;
    }
  }
}
