import { describe, expect, test } from "bun:test";
import { completedWork, failed, finished, startRun, stepDone, stepRunning, summarizing, withPlan } from "../src/run-state.ts";

describe("run state the UI polls", () => {
  const planned = withPlan(startRun("plan a trip", "claude", "m"), ["flights", "hotel"]);

  test("a new run starts in planning with no steps", () => {
    expect(startRun("g", "claude", "m")).toMatchObject({ phase: "planning", steps: [] });
  });

  test("a plan turns every task into a pending step and moves to working", () => {
    expect(planned.phase).toBe("working");
    expect(planned.steps.map(step => step.status)).toEqual(["pending", "pending"]);
  });

  test("only the targeted step changes and earlier states are not mutated", () => {
    const running = stepRunning(planned, 1);
    expect(running.steps.map(step => step.status)).toEqual(["pending", "running"]);
    expect(planned.steps[1].status).toBe("pending");
  });

  test("completed work feeds only finished steps to the next agent", () => {
    const state = stepRunning(stepDone(planned, 0, "LIS to OPO"), 1);
    expect(completedWork(state)).toBe("Task 1: flights\nResult: LIS to OPO");
  });

  test("the run ends as done with a summary or failed with the error", () => {
    expect(finished(summarizing(planned), "all set")).toMatchObject({ phase: "done", summary: "all set" });
    expect(failed(planned, "boom")).toMatchObject({ phase: "failed", error: "boom" });
  });
});
