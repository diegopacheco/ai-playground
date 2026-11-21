import { beforeAll, describe, expect, test } from "bun:test";
import { initPhysics } from "../physics/arena";
import { CABINET, CLAW } from "./layout";
import type { Input } from "./machine";
import { type Simulation, createSimulation } from "./simulation";

const DT = 1 / 60;
const NOTHING: Input = { moveX: 0, moveZ: 0, drop: false, coin: false };

function seeded(seed: number): () => number {
  let value = seed;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
}

function run(sim: Simulation, input: Input, steps: number): void {
  for (let i = 0; i < steps; i += 1) sim.step(input, DT);
}

function steerTo(sim: Simulation, x: number, z: number): void {
  for (let i = 0; i < 600; i += 1) {
    const claw = sim.claw();
    const dx = x - claw.x;
    const dz = z - claw.z;
    if (Math.hypot(dx, dz) < 0.004) return;
    sim.step({ ...NOTHING, moveX: Math.max(-1, Math.min(1, dx * 12)), moveZ: Math.max(-1, Math.min(1, dz * 12)) }, DT);
  }
}

function playRound(sim: Simulation, x: number, z: number): void {
  sim.step({ ...NOTHING, coin: true }, DT);
  sim.step(NOTHING, DT);
  steerTo(sim, x, z);
  sim.step({ ...NOTHING, drop: true }, DT);
  for (let i = 0; i < 1800 && sim.state().phase !== "idle"; i += 1) sim.step(NOTHING, DT);
}

beforeAll(async () => {
  await initPhysics();
});

describe("claw physics", () => {
  test("a toy right under the claw is carried to the chute and counted as a prize", () => {
    const sim = createSimulation({
      random: seeded(1),
      restock: false,
      toys: [{ x: 0.2, y: 0.12, z: -0.1, radius: 0.095, look: 0, spin: 0 }],
    });
    run(sim, NOTHING, 120);
    playRound(sim, 0.2, -0.1);
    expect(sim.state().phase).toBe("idle");
    expect(sim.state().prizes).toBe(1);
    expect(sim.arena.toys().length).toBe(0);
  });

  test("dropping on an empty spot wins nothing", () => {
    const sim = createSimulation({
      random: seeded(2),
      restock: false,
      toys: [{ x: 0.3, y: 0.12, z: -0.3, radius: 0.095, look: 0, spin: 0 }],
    });
    run(sim, NOTHING, 120);
    playRound(sim, -0.1, -0.1);
    expect(sim.state().prizes).toBe(0);
    expect(sim.arena.toys().length).toBe(1);
  });

  test("without a coin the claw stays parked over the chute", () => {
    const sim = createSimulation({ random: seeded(3), toys: [] });
    run(sim, { ...NOTHING, moveX: 1, moveZ: -1, drop: true }, 120);
    expect(sim.claw().x).toBeCloseTo(CLAW.homeX, 3);
    expect(sim.claw().z).toBeCloseTo(CLAW.homeZ, 3);
    expect(sim.claw().y).toBeCloseTo(CLAW.topY, 3);
  });

  test("a full pile stays inside the cabinet and calm while the claw digs into it", () => {
    const sim = createSimulation({ random: seeded(4), restock: false });
    run(sim, NOTHING, 360);
    const before = sim.arena.toys().length;
    playRound(sim, 0.1, -0.1);
    run(sim, NOTHING, 120);
    const limit = CABINET.half + 0.001;
    for (const toy of sim.arena.toys()) {
      const at = toy.body.translation();
      const speed = toy.body.linvel();
      expect(Math.abs(at.x)).toBeLessThan(limit);
      expect(Math.abs(at.z)).toBeLessThan(limit);
      expect(at.y).toBeLessThan(CABINET.wallHeight);
      expect(Math.hypot(speed.x, speed.y, speed.z)).toBeLessThan(1);
    }
    expect(sim.arena.toys().length + sim.state().prizes).toBe(before);
  });

  test("the claw stops on top of a toy instead of pushing through it", () => {
    const sim = createSimulation({
      random: seeded(5),
      restock: false,
      toys: [{ x: 0.1, y: 0.25, z: 0, radius: 0.2, look: 0, spin: 0 }],
    });
    run(sim, NOTHING, 120);
    sim.step({ ...NOTHING, coin: true }, DT);
    sim.step(NOTHING, DT);
    steerTo(sim, 0.1, 0);
    sim.step({ ...NOTHING, drop: true }, DT);
    for (let i = 0; i < 600 && sim.state().phase !== "closing"; i += 1) sim.step(NOTHING, DT);
    expect(sim.state().phase).toBe("closing");
    expect(sim.claw().y).toBeGreaterThan(CLAW.minY + 0.02);
  });
});
