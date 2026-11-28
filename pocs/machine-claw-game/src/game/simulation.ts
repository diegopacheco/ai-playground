import { type Arena, type ClawPose, type ToySpec, createArena } from "../physics/arena";
import { CLAW, insideChute } from "./layout";
import { type GameState, type Input, type Phase, INITIAL_STATE, RULES, advance, awardPrize, command } from "./machine";
import { PLUSH_LOOKS } from "./plush";

export type SimEvent =
  | { readonly kind: "phase"; readonly phase: Phase }
  | { readonly kind: "prize"; readonly look: number }
  | { readonly kind: "roundEnd"; readonly won: boolean };

export type SimOptions = {
  readonly random: () => number;
  readonly toys?: readonly ToySpec[];
  readonly restock?: boolean;
};

export type Simulation = {
  readonly arena: Arena;
  state(): GameState;
  claw(): ClawPose;
  step(input: Input, dt: number): readonly SimEvent[];
};

export const BASE_PILE_SIZE = 16;
export const PILE_SIZE = Math.round(BASE_PILE_SIZE * 2.5);
export const TOY_RADIUS = { min: 0.078, max: 0.1 } as const;
const CELL = 0.21;
const GRIP_SPEED = 1.7;
const HOME_TOLERANCE = 0.01;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toyRadius(random: () => number): number {
  return TOY_RADIUS.min + random() * (TOY_RADIUS.max - TOY_RADIUS.min);
}

function randomToy(random: () => number, y: number, minX: number): ToySpec {
  const x = minX + random() * (0.42 - minX);
  const z = -0.42 + random() * 0.84;
  if (insideChute(x, z, 0.14)) return randomToy(random, y, minX);
  return { x, y, z, radius: toyRadius(random), look: Math.floor(random() * PLUSH_LOOKS.length), spin: random() * Math.PI * 2 };
}

function pileSpots(): readonly { readonly x: number; readonly z: number }[] {
  const steps = [-2, -1, 0, 1, 2].map((step) => step * CELL);
  return steps.flatMap((x) => steps.map((z) => ({ x, z }))).filter((spot) => !insideChute(spot.x, spot.z, 0.14));
}

function shuffledLooks(random: () => number, count: number): number[] {
  const looks = Array.from({ length: count }, (_, i) => i % PLUSH_LOOKS.length);
  for (let i = looks.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [looks[i], looks[j]] = [looks[j] ?? 0, looks[i] ?? 0];
  }
  return looks;
}

export function stackPile(random: () => number, count: number): readonly ToySpec[] {
  const spots = pileSpots();
  const looks = shuffledLooks(random, count);
  return looks.map((look, i) => {
    const spot = spots[i % spots.length] ?? { x: 0, z: 0 };
    const layer = Math.floor(i / spots.length);
    return {
      x: spot.x + (random() - 0.5) * 0.01,
      y: 0.12 + layer * CELL,
      z: spot.z + (random() - 0.5) * 0.01,
      radius: toyRadius(random),
      look,
      spin: random() * Math.PI * 2,
    };
  });
}

export function createSimulation(options: SimOptions): Simulation {
  const arena = createArena();
  const pile = options.toys ?? stackPile(options.random, PILE_SIZE);
  for (const spec of pile) arena.addToy(spec);

  let state = INITIAL_STATE;
  let pose: ClawPose = { x: CLAW.homeX, y: CLAW.topY, z: CLAW.homeZ, grip: 0 };
  let velocity = { x: 0, y: 0, z: 0 };
  let roundPrizes = 0;
  arena.placeClaw(pose, true);

  function sensors() {
    const descending = state.phase === "descending";
    return {
      touching: descending && (pose.y <= CLAW.minY + 0.001 || arena.clearance(pose) <= CLAW.touchDistance),
      atTop: pose.y >= CLAW.topY - 0.002,
      atHome: Math.hypot(CLAW.homeX - pose.x, CLAW.homeZ - pose.z) <= HOME_TOLERANCE,
      gripClosed: pose.grip >= 1,
      gripOpen: pose.grip <= 0,
    };
  }

  function move(input: Input, dt: number): void {
    const order = command(state.phase, input);
    const toHomeX = CLAW.homeX - pose.x;
    const toHomeZ = CLAW.homeZ - pose.z;
    const distance = Math.hypot(toHomeX, toHomeZ);
    const homeSpeed = distance > 0 ? Math.min(RULES.returnSpeed, distance * 6) / distance : 0;
    const wanted = order.homing
      ? { x: toHomeX * homeSpeed, y: order.vy, z: toHomeZ * homeSpeed }
      : { x: order.vx, y: order.vy, z: order.vz };
    const blend = Math.min(1, dt * 12);
    velocity = {
      x: velocity.x + (wanted.x - velocity.x) * blend,
      y: velocity.y + (wanted.y - velocity.y) * Math.min(1, dt * 24),
      z: velocity.z + (wanted.z - velocity.z) * blend,
    };
    const gripStep = Math.sign(order.grip - pose.grip) * GRIP_SPEED * dt;
    pose = {
      x: clamp(pose.x + velocity.x * dt, -CLAW.reach, CLAW.reach),
      y: clamp(pose.y + velocity.y * dt, CLAW.minY, CLAW.topY),
      z: clamp(pose.z + velocity.z * dt, -CLAW.reach, CLAW.reach),
      grip: clamp(pose.grip + gripStep, 0, 1),
    };
  }

  function step(input: Input, dt: number): readonly SimEvent[] {
    const events: SimEvent[] = [];
    const next = advance(state, input, sensors(), dt);
    if (next.phase !== state.phase) {
      if (state.phase === "releasing") events.push({ kind: "roundEnd", won: roundPrizes > 0 });
      if (next.phase === "aiming") roundPrizes = 0;
      events.push({ kind: "phase", phase: next.phase });
    }
    state = next;
    move(input, dt);
    arena.placeClaw(pose, false);
    arena.step(dt);
    for (const toy of arena.takeFallen()) {
      state = awardPrize(state);
      roundPrizes += 1;
      events.push({ kind: "prize", look: toy.look });
      if (options.restock !== false) arena.addToy(randomToy(options.random, 1.2, 0));
    }
    return events;
  }

  return { arena, state: () => state, claw: () => pose, step };
}
