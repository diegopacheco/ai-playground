import { describe, test, expect } from "bun:test";
import { LimitGuard } from "../src/limits.ts";

describe("LimitGuard", () => {
  test("ticks while under budget", () => {
    const guard = new LimitGuard({ maxSteps: 3, wallClockMs: 60_000 });
    expect(guard.tick()).toEqual({ kind: "ok" });
    expect(guard.tick()).toEqual({ kind: "ok" });
    expect(guard.tick()).toEqual({ kind: "ok" });
    expect(guard.currentStep).toBe(3);
  });

  test("trips on step_budget at step 26 when maxSteps=25", () => {
    const guard = new LimitGuard({ maxSteps: 25, wallClockMs: 60_000 });
    for (let i = 0; i < 25; i++) {
      expect(guard.tick()).toEqual({ kind: "ok" });
    }
    expect(guard.tick()).toEqual({ kind: "trip", reason: "step_budget" });
  });

  test("trips on wall_clock when time exceeds wallClockMs", () => {
    let t = 1000;
    const guard = new LimitGuard({
      maxSteps: 1000,
      wallClockMs: 5_000,
      now: () => t,
    });
    expect(guard.tick()).toEqual({ kind: "ok" });
    t = 1000 + 5_001;
    expect(guard.tick()).toEqual({ kind: "trip", reason: "wall_clock" });
  });

  test("peek does not consume a step", () => {
    const guard = new LimitGuard({ maxSteps: 2, wallClockMs: 60_000 });
    expect(guard.peek()).toEqual({ kind: "ok" });
    expect(guard.peek()).toEqual({ kind: "ok" });
    expect(guard.currentStep).toBe(0);
    guard.tick();
    guard.tick();
    expect(guard.peek()).toEqual({ kind: "trip", reason: "step_budget" });
  });

  test("rejects non-positive maxSteps", () => {
    expect(() => new LimitGuard({ maxSteps: 0, wallClockMs: 1000 })).toThrow();
    expect(() => new LimitGuard({ maxSteps: -1, wallClockMs: 1000 })).toThrow();
  });

  test("rejects non-positive wallClockMs", () => {
    expect(() => new LimitGuard({ maxSteps: 1, wallClockMs: 0 })).toThrow();
  });

  test("budget reports remaining steps", () => {
    const guard = new LimitGuard({ maxSteps: 5, wallClockMs: 60_000 });
    expect(guard.budget).toEqual({ max: 5, remaining: 5 });
    guard.tick();
    guard.tick();
    expect(guard.budget).toEqual({ max: 5, remaining: 3 });
  });
});
