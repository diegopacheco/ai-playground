import { describe, expect, test } from "bun:test";
import { isAgentStep, isFinal, isFinalWorkflow, pillClass, progress, stepDuration } from "../src/dbos.ts";
import type { JournalStep, RunState } from "../src/types.ts";

const base: RunState = { goal: "g", provider: "claude", model: "m", phase: "planning", steps: [], summary: "", error: "" };
const step: JournalStep = { id: 1, name: "plan", output: "", error: "", started_at: 1000, completed_at: 3500 };

describe("dbos helpers used by the UI", () => {
  test("progress counts plan and summary as steps and only reaches 100 when done", () => {
    const working = { ...base, phase: "working" as const, steps: [{ task: "a", status: "done" as const, output: "x" }, { task: "b", status: "running" as const, output: "" }] };
    expect(progress(base)).toBe(0);
    expect(progress(working)).toBe(50);
    expect(progress({ ...working, phase: "summarizing" })).toBeLessThan(100);
    expect(progress({ ...working, phase: "done" })).toBe(100);
  });

  test("run polling stops only on done or failed", () => {
    expect(isFinal({ ...base, phase: "done" })).toBe(true);
    expect(isFinal({ ...base, phase: "failed" })).toBe(true);
    expect(isFinal({ ...base, phase: "summarizing" })).toBe(false);
    expect(isFinal(null)).toBe(false);
  });

  test("journal polling keeps going while DBOS may still recover the workflow", () => {
    expect(isFinalWorkflow("PENDING")).toBe(false);
    expect(isFinalWorkflow(undefined)).toBe(false);
    expect(isFinalWorkflow("SUCCESS")).toBe(true);
    expect(isFinalWorkflow("MAX_RECOVERY_ATTEMPTS_EXCEEDED")).toBe(true);
  });

  test("DBOS statuses map to the css pill classes and a pending workflow shows as running", () => {
    expect(pillClass("SUCCESS")).toBe("success");
    expect(pillClass("PENDING")).toBe("running");
    expect(pillClass("MAX_RECOVERY_ATTEMPTS_EXCEEDED")).toBe("max-recovery-attempts-exceeded");
  });

  test("step duration reads the DBOS timestamps and stays blank while a step runs", () => {
    expect(stepDuration(step)).toBe("2.5s");
    expect(stepDuration({ ...step, completed_at: 1200 })).toBe("200ms");
    expect(stepDuration({ ...step, completed_at: null })).toBe("");
  });

  test("DBOS internal operations are told apart from agent steps", () => {
    expect(isAgentStep(step)).toBe(true);
    expect(isAgentStep({ ...step, name: "DBOS.setEvent" })).toBe(false);
  });
});
