import { describe, expect, test } from "bun:test";
import { isFinalRun, pillClass, progress } from "../src/temporal.ts";
import type { RunState, RunView } from "../src/types.ts";

const base: RunState = { goal: "g", provider: "claude", model: "m", phase: "planning", steps: [], summary: "", error: "" };
const view = (status: string): RunView => ({ run_id: "run-1", status, temporal_url: "u", state: base });

describe("temporal helpers used by the UI", () => {
  test("progress counts plan and summary as steps and only reaches 100 when done", () => {
    const working = { ...base, phase: "working" as const, steps: [{ task: "a", status: "done" as const, output: "x" }, { task: "b", status: "running" as const, output: "" }] };
    expect(progress(base)).toBe(0);
    expect(progress(working)).toBe(50);
    expect(progress({ ...working, phase: "done" })).toBe(100);
  });

  test("polling keeps going while the workflow is open, even with no worker alive", () => {
    expect(isFinalRun(view("RUNNING"))).toBe(false);
    expect(isFinalRun(undefined)).toBe(false);
  });

  test("polling stops once the workflow is closed for any reason", () => {
    expect(isFinalRun(view("COMPLETED"))).toBe(true);
    expect(isFinalRun(view("FAILED"))).toBe(true);
    expect(isFinalRun(view("TERMINATED"))).toBe(true);
  });

  test("workflow statuses map to done, failed or running pills", () => {
    expect(pillClass("COMPLETED")).toBe("done");
    expect(pillClass("TIMED_OUT")).toBe("failed");
    expect(pillClass("RUNNING")).toBe("running");
  });
});
