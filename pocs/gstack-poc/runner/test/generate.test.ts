import { describe, test, expect } from "bun:test";
import { runGenerate } from "../src/generate.ts";
import type { BrowserAdapter } from "../src/browser.ts";
import type { OllamaClient, OllamaChatRequest, OllamaChatResponse } from "../src/ollama.ts";
import type { Action, RunEvent } from "../src/types.ts";

class FakeBrowser implements BrowserAdapter {
  public readonly calls: string[] = [];
  public failOn: { selector: string } | undefined;
  constructor(private url: string = "https://example.com") {}
  async goto(url: string): Promise<void> {
    this.url = url;
    this.calls.push(`goto:${url}`);
  }
  async screenshot(): Promise<string> {
    this.calls.push("screenshot");
    return "base64-png-placeholder";
  }
  async click(selector: { kind: string }): Promise<void> {
    this.calls.push(`click:${selector.kind}`);
  }
  async type(selector: { kind: string }, text: string): Promise<void> {
    this.calls.push(`type:${selector.kind}:${text}`);
  }
  async waitFor(selector: { kind: string }): Promise<void> {
    this.calls.push(`wait:${selector.kind}`);
  }
  async assertText(selector: { kind: string }, text: string): Promise<void> {
    if (this.failOn?.selector === selector.kind) {
      throw new Error("assertion failed");
    }
    this.calls.push(`assert:${selector.kind}:${text}`);
  }
  async currentUrl(): Promise<string> {
    return this.url;
  }
}

class ScriptedOllama implements OllamaClient {
  public callCount = 0;
  constructor(private readonly responses: Array<Action | string>) {}
  async chat(_request: OllamaChatRequest): Promise<OllamaChatResponse> {
    const idx = this.callCount;
    this.callCount += 1;
    const next = this.responses[idx];
    if (next === undefined) {
      throw new Error(`ScriptedOllama: no scripted response at index ${idx}`);
    }
    const content = typeof next === "string" ? next : JSON.stringify(actionToJson(next));
    return {
      model: "mock",
      message: { role: "assistant", content },
      done: true,
    };
  }
}

function actionToJson(action: Action): Record<string, unknown> {
  const base: Record<string, unknown> = { action: action.tool, reason: action.reason };
  if ("selector" in action) base.selector = action.selector;
  if ("text" in action) base.text = action.text;
  return base;
}

describe("runGenerate", () => {
  test("happy path: type → click → assert → done", async () => {
    const browser = new FakeBrowser();
    const ollama = new ScriptedOllama([
      {
        tool: "type",
        selector: { kind: "placeholder", text: "Username" },
        text: "standard_user",
        reason: "fill the username from the prompt",
      },
      {
        tool: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "click the login button to submit",
      },
      {
        tool: "assert_text",
        selector: { kind: "text", text: "Products" },
        text: "Products",
        reason: "verify we landed on the inventory page",
      },
      { tool: "done", reason: "test passed" },
    ]);
    const events: RunEvent[] = [];
    const result = await runGenerate(
      {
        prompt: "log in",
        url: "https://www.saucedemo.com",
        maxSteps: 25,
        wallClockMs: 60_000,
        onEvent: (e) => events.push(e),
      },
      { browser, ollama },
    );
    expect(result.stopReason).toBe("done");
    expect(result.log.length).toBe(4);
    expect(browser.calls).toContain("goto:https://www.saucedemo.com");
    expect(browser.calls).toContain("type:placeholder:standard_user");
    expect(browser.calls).toContain("click:role");
    expect(browser.calls).toContain("assert:text:Products");
    const stepEvents = events.filter((e) => e.type === "step");
    expect(stepEvents.length).toBe(4);
    const statusEvents = events.filter((e) => e.type === "status");
    expect(statusEvents[0]).toMatchObject({ status: "started" });
    expect(statusEvents[statusEvents.length - 1]).toMatchObject({ status: "complete" });
  });

  test("trips on step_budget when LLM keeps clicking", async () => {
    const browser = new FakeBrowser();
    const clickForever: Action = {
      tool: "click",
      selector: { kind: "role", role: "button", name: "Login" },
      reason: "keep clicking forever",
    };
    const ollama = new ScriptedOllama(Array.from({ length: 50 }, () => clickForever));
    const result = await runGenerate(
      {
        prompt: "log in",
        url: "https://www.saucedemo.com",
        maxSteps: 3,
        wallClockMs: 60_000,
      },
      { browser, ollama },
    );
    expect(result.stopReason).toBe("step_budget");
    expect(result.log.length).toBe(3);
  });

  test("trips on wall_clock when controlled time exceeds budget", async () => {
    let t = 1000;
    const now = () => t;
    const browser = new FakeBrowser();
    const ollama = new ScriptedOllama([
      {
        tool: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "first action",
      },
      {
        tool: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "would be second action but time runs out",
      },
    ]);
    const result = await runGenerate(
      {
        prompt: "log in",
        url: "https://www.saucedemo.com",
        maxSteps: 25,
        wallClockMs: 1000,
      },
      { browser, ollama, now: () => { t += 600; return t; } },
    );
    expect(result.stopReason).toBe("wall_clock");
  });

  test("stops on model_error when JSON is malformed", async () => {
    const browser = new FakeBrowser();
    const ollama = new ScriptedOllama(["this is not json"]);
    const result = await runGenerate(
      {
        prompt: "log in",
        url: "https://www.saucedemo.com",
        maxSteps: 25,
        wallClockMs: 60_000,
      },
      { browser, ollama },
    );
    expect(result.stopReason).toBe("model_error");
  });

  test("continues past a failed action and eventually finishes", async () => {
    const browser = new FakeBrowser();
    browser.failOn = { selector: "text" };
    const ollama = new ScriptedOllama([
      {
        tool: "assert_text",
        selector: { kind: "text", text: "wrong" },
        text: "wrong",
        reason: "this will fail",
      },
      { tool: "done", reason: "give up and call it done" },
    ]);
    const result = await runGenerate(
      {
        prompt: "anything",
        url: "https://example.com",
        maxSteps: 25,
        wallClockMs: 60_000,
      },
      { browser, ollama },
    );
    expect(result.stopReason).toBe("done");
    expect(result.log[0]!.result.status).toBe("failed");
    expect(result.log[1]!.action.tool).toBe("done");
  });

  test("emits step events with action + reason microcopy", async () => {
    const browser = new FakeBrowser();
    const ollama = new ScriptedOllama([
      {
        tool: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "because the prompt says log in",
      },
      { tool: "done", reason: "done" },
    ]);
    const events: RunEvent[] = [];
    await runGenerate(
      {
        prompt: "log in",
        url: "https://www.saucedemo.com",
        maxSteps: 25,
        wallClockMs: 60_000,
        onEvent: (e) => events.push(e),
      },
      { browser, ollama },
    );
    const stepEvents = events.filter((e) => e.type === "step");
    expect(stepEvents[0]).toMatchObject({
      verb: expect.stringContaining("Clicking"),
      reason: "because the prompt says log in",
    });
  });

  test("reports browser_error when goto fails", async () => {
    const browser = new FakeBrowser();
    browser.goto = async () => {
      throw new Error("dns failure");
    };
    const ollama = new ScriptedOllama([]);
    const result = await runGenerate(
      {
        prompt: "log in",
        url: "https://invalid.invalid",
        maxSteps: 25,
        wallClockMs: 60_000,
      },
      { browser, ollama },
    );
    expect(result.stopReason).toBe("browser_error");
    expect(result.log.length).toBe(0);
  });
});
