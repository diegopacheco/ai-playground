import { describe, expect, test } from "bun:test";
import { isFinalRun, pillClass, progress } from "../src/trigger.ts";
import type { RunState, RunView } from "../src/types.ts";

const base: RunState = { goal: "g", provider: "claude", model: "m", phase: "planning", steps: [], summary: "", error: "" };
const view = (status: string, state: RunState | null): RunView => ({ run_id: "run_1", status, dashboard_url: "u", state });

describe("trigger helpers used by the UI", () => {
  test("progress counts plan and summary as steps and only reaches 100 when done", () => {
    const working = { ...base, phase: "working" as const, steps: [{ task: "a", status: "done" as const, output: "x" }, { task: "b", status: "running" as const, output: "" }] };
    expect(progress(base)).toBe(0);
    expect(progress(working)).toBe(50);
    expect(progress({ ...working, phase: "done" })).toBe(100);
  });

  test("polling keeps going while a crashed attempt is being retried", () => {
    expect(isFinalRun(view("EXECUTING", base))).toBe(false);
    expect(isFinalRun(view("REATTEMPTING", { ...base, phase: "failed" }))).toBe(false);
    expect(isFinalRun(undefined)).toBe(false);
  });

  test("polling stops on a crashed or failed run, and on a completed run once the final state is published", () => {
    expect(isFinalRun(view("COMPLETED", { ...base, phase: "done" }))).toBe(true);
    expect(isFinalRun(view("COMPLETED", { ...base, phase: "summarizing" }))).toBe(false);
    expect(isFinalRun(view("FAILED", null))).toBe(true);
    expect(isFinalRun(view("CRASHED", { ...base, phase: "working" }))).toBe(true);
  });

  test("run statuses map to done, failed or running pills", () => {
    expect(pillClass("COMPLETED")).toBe("done");
    expect(pillClass("CRASHED")).toBe("failed");
    expect(pillClass("QUEUED")).toBe("running");
  });
});
