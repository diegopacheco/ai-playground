import { describe, expect, test } from "bun:test";
import { isFinal, pillClass, progress } from "../src/dbos.ts";
import type { RunState } from "../src/types.ts";

const base: RunState = { goal: "g", provider: "claude", model: "m", phase: "planning", steps: [], summary: "", error: "" };

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

  test("DBOS statuses map to the css pill classes and a pending workflow shows as running", () => {
    expect(pillClass("SUCCESS")).toBe("success");
    expect(pillClass("PENDING")).toBe("running");
    expect(pillClass("MAX_RECOVERY_ATTEMPTS_EXCEEDED")).toBe("max-recovery-attempts-exceeded");
  });
});
