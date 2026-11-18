import { describe, expect, test } from "bun:test";
import { isFinal, newRunId, progress, restateUiUrl, runsQuery, sqlText, toRunRows } from "../src/restate.ts";
import type { RunState } from "../src/types.ts";

const base: RunState = { goal: "g", provider: "claude", model: "m", phase: "planning", steps: [], summary: "", error: "" };

describe("restate helpers used by the UI", () => {
  test("a run id from the URL cannot break out of the admin SQL string", () => {
    expect(sqlText("x' OR '1'='1")).toBe("'x'' OR ''1''=''1'");
    expect(runsQuery("x'--")).toContain("target_service_key = 'x''--'");
  });

  test("without a run id the query lists every run of the workflow, newest first", () => {
    const query = runsQuery();
    expect(query).not.toContain("target_service_key =");
    expect(query).toContain("ORDER BY created_at DESC");
  });

  test("the Restate button deep links to the invocation page of the Restate UI", () => {
    expect(restateUiUrl("http://localhost:9070", "inv_1a2b")).toBe("http://localhost:9070/ui/invocations/inv_1a2b");
  });

  test("run ids are unique per start time so each goal is a new workflow", () => {
    expect(newRunId(1)).not.toBe(newRunId(2));
    expect(newRunId(1)).toMatch(/^run-/);
  });

  test("progress counts plan and summary as steps and only reaches 100 when done", () => {
    const working = { ...base, phase: "working" as const, steps: [{ task: "a", status: "done" as const, output: "x" }, { task: "b", status: "running" as const, output: "" }] };
    expect(progress(base)).toBe(0);
    expect(progress(working)).toBe(50);
    expect(progress({ ...working, phase: "summarizing" })).toBeLessThan(100);
    expect(progress({ ...working, phase: "done" })).toBe(100);
  });

  test("polling stops only on done or failed", () => {
    expect(isFinal({ ...base, phase: "done" })).toBe(true);
    expect(isFinal({ ...base, phase: "failed" })).toBe(true);
    expect(isFinal({ ...base, phase: "summarizing" })).toBe(false);
    expect(isFinal(null)).toBe(false);
  });

  test("admin SQL rows map to UI rows", () => {
    expect(toRunRows([{ id: "inv_1", target_service_key: "run-1", status: "running", created_at: "2026-11-17" }]))
      .toEqual([{ runId: "run-1", invocationId: "inv_1", status: "running", createdAt: "2026-11-17" }]);
  });
});
