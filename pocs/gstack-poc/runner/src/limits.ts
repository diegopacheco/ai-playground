import type { StopReason } from "./types.ts";

export type LimitVerdict =
  | { kind: "ok" }
  | { kind: "trip"; reason: Extract<StopReason, "step_budget" | "wall_clock"> };

export interface LimitGuardOptions {
  maxSteps: number;
  wallClockMs: number;
  now?: () => number;
}

export class LimitGuard {
  private readonly maxSteps: number;
  private readonly wallClockMs: number;
  private readonly now: () => number;
  private readonly startedAt: number;
  private stepsConsumed = 0;

  constructor(options: LimitGuardOptions) {
    if (options.maxSteps <= 0) {
      throw new Error(`maxSteps must be positive, got ${options.maxSteps}`);
    }
    if (options.wallClockMs <= 0) {
      throw new Error(`wallClockMs must be positive, got ${options.wallClockMs}`);
    }
    this.maxSteps = options.maxSteps;
    this.wallClockMs = options.wallClockMs;
    this.now = options.now ?? Date.now;
    this.startedAt = this.now();
  }

  tick(): LimitVerdict {
    this.stepsConsumed += 1;
    if (this.stepsConsumed > this.maxSteps) {
      return { kind: "trip", reason: "step_budget" };
    }
    if (this.now() - this.startedAt > this.wallClockMs) {
      return { kind: "trip", reason: "wall_clock" };
    }
    return { kind: "ok" };
  }

  peek(): LimitVerdict {
    if (this.stepsConsumed >= this.maxSteps) {
      return { kind: "trip", reason: "step_budget" };
    }
    if (this.now() - this.startedAt > this.wallClockMs) {
      return { kind: "trip", reason: "wall_clock" };
    }
    return { kind: "ok" };
  }

  get currentStep(): number {
    return this.stepsConsumed;
  }

  get elapsedMs(): number {
    return this.now() - this.startedAt;
  }

  get budget(): { max: number; remaining: number } {
    return {
      max: this.maxSteps,
      remaining: Math.max(0, this.maxSteps - this.stepsConsumed),
    };
  }
}
