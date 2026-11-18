import { describe, expect, test } from "bun:test";
import { loadPrompt, renderPrompt } from "../src/prompts.ts";

const template = "---\nname: t\nparams: $goal, $goal_owner\n---\nOwner $goal_owner wants $goal.\n";

describe("prompt templates", () => {
  test("replaces every declared parameter, longest name first so $goal does not eat $goal_owner", () => {
    expect(renderPrompt(template, { goal: "a trip", goal_owner: "Ana" })).toBe("Owner Ana wants a trip.");
  });

  test("fails loud when a declared parameter has no value, so the agent never sees a raw $placeholder", () => {
    expect(() => renderPrompt(template, { goal: "a trip" })).toThrow("$goal_owner");
  });

  test("rejects a prompt file without the --- header that documents its parameters", () => {
    expect(() => renderPrompt("Do $goal", { goal: "x" })).toThrow("header");
  });

  test("every shipped prompt renders with the parameters the workflow passes", () => {
    expect(loadPrompt("plan-goal", { goal: "learn rust" })).toContain("learn rust");
    expect(loadPrompt("execute-task", { goal: "g", task: "t1", previous_work: "none yet" })).toContain("t1");
    expect(loadPrompt("summarize-run", { goal: "g", work: "w1" })).toContain("w1");
  });
});
