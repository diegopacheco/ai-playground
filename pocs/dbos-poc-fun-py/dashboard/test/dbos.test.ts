import { describe, expect, test } from "bun:test";
import { formatDuration, isFinal, pillClass, preview, stepValue, timeline, workflowEnd } from "../src/dbos.ts";
import type { StepRow, Stored, WorkflowDetail, WorkflowRow } from "../src/types.ts";

const none: Stored = { format: "agent_json", readable: true, value: null };
const json = (value: Stored["value"]): Stored => ({ format: "agent_json", readable: true, value });
const workflow: WorkflowRow = {
  id: "run-1", name: "AgentWorkflow", status: "PENDING", created_at: 1000, updated_at: 1000, completed_at: null,
  executor: "local", app_version: "v", recovery_attempts: 1, queue: "", steps: 2,
};
const step = (id: number, name: string, start: number, end: number, output: Stored = none): StepRow => ({
  id, name, output, error: none, child_workflow: "", started_at: start, completed_at: end,
});
const detail: WorkflowDetail = {
  workflow, serialization: "agent_json", input: none, output: none, error: none,
  steps: [step(1, "read config", 1000, 1100), step(2, "DBOS.setEvent", 1100, 1110), step(3, "plan", 1110, 3000)],
  events: [], history: [{ function_id: 2, key: "state", value: json({ phase: "planning" }) }],
};

describe("dashboard helpers", () => {
  test("polling keeps going while DBOS may still recover a pending workflow", () => {
    expect(isFinal("PENDING")).toBe(false);
    expect(isFinal(undefined)).toBe(false);
    expect(isFinal("SUCCESS")).toBe(true);
    expect(isFinal("MAX_RECOVERY_ATTEMPTS_EXCEEDED")).toBe(true);
    expect(pillClass("PENDING")).toBe("running");
    expect(pillClass("MAX_RECOVERY_ATTEMPTS_EXCEEDED")).toBe("max-recovery-attempts-exceeded");
  });

  test("a running workflow is measured up to now and a finished one up to its end", () => {
    expect(workflowEnd(workflow, 5000)).toBe(5000);
    expect(workflowEnd({ ...workflow, status: "SUCCESS", completed_at: 4000 }, 9000)).toBe(4000);
    expect(formatDuration(250)).toBe("250ms");
    expect(formatDuration(2500)).toBe("2.5s");
    expect(formatDuration(65_000)).toBe("1m 05s");
  });

  test("the timeline hides DBOS internals by default and shows the step DBOS has not recorded yet", () => {
    const bars = timeline(detail, 5000, false);
    expect(bars.map(bar => bar.label)).toEqual(["1 read config", "3 plan", "next step running"]);
    const running = bars.at(-1)!;
    expect(running.kind).toBe("running");
    expect(running.left).toBeCloseTo(50);
    expect(running.left + running.width).toBeCloseTo(100);
    expect(timeline(detail, 5000, true)).toHaveLength(4);
    expect(timeline({ ...detail, workflow: { ...workflow, status: "SUCCESS", completed_at: 3000 } }, 9000, false)).toHaveLength(2);
  });

  test("a setEvent step shows the value it published, and pickled data is flagged instead of shown", () => {
    expect(stepValue(detail.steps[1]!, detail.history).value).toEqual({ phase: "planning" });
    expect(preview(json("plain text"))).toBe("plain text");
    expect(preview({ format: "py_pickle", readable: false, value: null })).toBe("stored as py_pickle, not readable outside Python");
  });
});
