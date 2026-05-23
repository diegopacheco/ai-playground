import { describe, test, expect } from "bun:test";
import { actionLogToScript } from "../src/templater.ts";
import type { ActionLogEntry, RunResult } from "../src/types.ts";

function ok(action: ActionLogEntry["action"], step: number): ActionLogEntry {
  return {
    step,
    action,
    result: { status: "ok", action, tookMs: 10 },
    timestamp: 1000 + step,
  };
}

function failed(action: ActionLogEntry["action"], step: number): ActionLogEntry {
  return {
    step,
    action,
    result: { status: "failed", action, error: "boom", tookMs: 10 },
    timestamp: 1000 + step,
  };
}

const baseRun: Omit<RunResult, "log" | "stopReason"> = {
  url: "https://www.saucedemo.com",
  prompt: "log in as standard_user and see the inventory page",
  startedAt: 0,
  endedAt: 100,
};

describe("actionLogToScript", () => {
  test("emits an empty test with just the goto for an empty log", () => {
    const script = actionLogToScript({ ...baseRun, log: [], stopReason: "done" });
    expect(script).toContain("import { test, expect } from '@playwright/test';");
    expect(script).toContain("test('log in as standard_user and see the inventory page', async ({ page }) =>");
    expect(script).toContain("await page.goto('https://www.saucedemo.com')");
    expect(script).toContain("});");
  });

  test("renders a complete login flow", () => {
    const log: ActionLogEntry[] = [
      ok(
        {
          tool: "type",
          selector: { kind: "placeholder", text: "Username" },
          text: "standard_user",
          reason: "username from the prompt",
        },
        1,
      ),
      ok(
        {
          tool: "type",
          selector: { kind: "placeholder", text: "Password" },
          text: "secret_sauce",
          reason: "matching password",
        },
        2,
      ),
      ok(
        {
          tool: "click",
          selector: { kind: "role", role: "button", name: "Login" },
          reason: "submit",
        },
        3,
      ),
      ok(
        {
          tool: "assert_text",
          selector: { kind: "text", text: "Products" },
          text: "Products",
          reason: "verify inventory",
        },
        4,
      ),
      ok({ tool: "done", reason: "test passed" }, 5),
    ];
    const script = actionLogToScript({ ...baseRun, log, stopReason: "done" });
    expect(script).toContain("page.getByPlaceholder('Username').fill('standard_user')");
    expect(script).toContain("page.getByPlaceholder('Password').fill('secret_sauce')");
    expect(script).toContain("page.getByRole('button', { name: 'Login' }).click()");
    expect(script).toContain("expect(page.getByText('Products')).toContainText('Products')");
    expect(script).not.toContain("done");
  });

  test("filters failed steps", () => {
    const log: ActionLogEntry[] = [
      ok(
        {
          tool: "click",
          selector: { kind: "role", role: "button", name: "Login" },
          reason: "first try",
        },
        1,
      ),
      failed(
        {
          tool: "click",
          selector: { kind: "role", role: "button", name: "Wrong" },
          reason: "this fails",
        },
        2,
      ),
      ok({ tool: "done", reason: "done" }, 3),
    ];
    const script = actionLogToScript({ ...baseRun, log, stopReason: "done" });
    expect(script).toContain("name: 'Login' ");
    expect(script).not.toContain("name: 'Wrong'");
  });

  test("annotates partial scripts with stop reason", () => {
    const log: ActionLogEntry[] = [
      ok(
        {
          tool: "click",
          selector: { kind: "role", role: "button", name: "Login" },
          reason: "first try",
        },
        1,
      ),
    ];
    const script = actionLogToScript({
      ...baseRun,
      log,
      stopReason: "step_budget",
    });
    expect(script).toContain("// stopped early: step_budget");
  });

  test("is deterministic — same input, same output", () => {
    const log: ActionLogEntry[] = [
      ok(
        {
          tool: "click",
          selector: { kind: "role", role: "button", name: "Login" },
          reason: "click it",
        },
        1,
      ),
    ];
    const run: RunResult = { ...baseRun, log, stopReason: "done" };
    const a = actionLogToScript(run);
    const b = actionLogToScript(run);
    expect(a).toBe(b);
  });

  test("emits all five selector kinds correctly", () => {
    const log: ActionLogEntry[] = [
      ok(
        { tool: "click", selector: { kind: "role", role: "button" }, reason: "r" },
        1,
      ),
      ok(
        {
          tool: "click",
          selector: { kind: "placeholder", text: "Email" },
          reason: "p",
        },
        2,
      ),
      ok({ tool: "click", selector: { kind: "text", text: "Hi" }, reason: "t" }, 3),
      ok(
        { tool: "click", selector: { kind: "label", text: "Name" }, reason: "l" },
        4,
      ),
      ok(
        {
          tool: "click",
          selector: { kind: "test_id", id: "submit-btn" },
          reason: "i",
        },
        5,
      ),
    ];
    const script = actionLogToScript({ ...baseRun, log, stopReason: "done" });
    expect(script).toContain("page.getByRole('button')");
    expect(script).toContain("page.getByPlaceholder('Email')");
    expect(script).toContain("page.getByText('Hi')");
    expect(script).toContain("page.getByLabel('Name')");
    expect(script).toContain("page.getByTestId('submit-btn')");
  });

  test("truncates long prompts in test name", () => {
    const longPrompt =
      "this is a really really long prompt that goes on and on and on and exceeds the eighty character cap by a comfortable margin to verify truncation works";
    const script = actionLogToScript({
      ...baseRun,
      prompt: longPrompt,
      log: [],
      stopReason: "done",
    });
    expect(script).toContain("...");
    const testNameMatch = script.match(/test\('([^']+)'/);
    expect(testNameMatch).not.toBeNull();
    expect(testNameMatch![1]!.length).toBeLessThanOrEqual(80);
  });

  test("escapes single quotes in strings via double-quote fallback", () => {
    const log: ActionLogEntry[] = [
      ok(
        {
          tool: "type",
          selector: { kind: "placeholder", text: "Search" },
          text: "it's working",
          reason: "single quote in input",
        },
        1,
      ),
    ];
    const script = actionLogToScript({ ...baseRun, log, stopReason: "done" });
    expect(script).toContain('"it\'s working"');
  });
});
