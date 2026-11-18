import { describe, expect, test } from "bun:test";
import { MAX_TASKS, parsePlan } from "../src/plan.ts";

describe("agent plan parsing", () => {
  test("reads the task list even when the model wraps it in prose or a code fence", () => {
    expect(parsePlan('Sure!\n```json\n["book flight", "find hotel"]\n```')).toEqual(["book flight", "find hotel"]);
  });

  test("caps the plan so a chatty model cannot make the workflow run forever", () => {
    const many = JSON.stringify(Array.from({ length: 10 }, (_, i) => `task ${i}`));
    expect(parsePlan(many)).toHaveLength(MAX_TASKS);
  });

  test("drops blank and non text items instead of running empty tasks", () => {
    expect(parsePlan('["  a  ", "", 3, null, "b"]')).toEqual(["a", "b"]);
  });

  test("throws when there is no usable plan so Restate retries the plan step", () => {
    expect(() => parsePlan("I cannot help")).toThrow("no JSON task list");
    expect(() => parsePlan("[]")).toThrow("no tasks");
  });
});
