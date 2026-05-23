import { describe, test, expect } from "bun:test";
import {
  DEFAULTS,
  RUNNER_VERSION,
  generateScript,
  actionLogToScript,
  runGenerate,
  LimitGuard,
  parseAction,
  type BrowserAdapter,
  type OllamaClient,
} from "../src/index.ts";

describe("public surface", () => {
  test("exports a version string", () => {
    expect(RUNNER_VERSION).toBe("0.0.1");
  });

  test("exposes the expected defaults", () => {
    expect(DEFAULTS.maxSteps).toBe(25);
    expect(DEFAULTS.wallClockMs).toBe(1_200_000);
    expect(DEFAULTS.ollamaUrl).toBe("http://127.0.0.1:11434");
    expect(DEFAULTS.model).toMatch(/^qwen2\.5vl:/);
  });

  test("re-exports the core runner symbols", () => {
    expect(typeof generateScript).toBe("function");
    expect(typeof runGenerate).toBe("function");
    expect(typeof actionLogToScript).toBe("function");
    expect(typeof parseAction).toBe("function");
    expect(LimitGuard).toBeDefined();
  });

  test("generateScript wires the runner to the templater end-to-end", async () => {
    const browser: BrowserAdapter = {
      async goto() {},
      async screenshot() {
        return "base64";
      },
      async click() {},
      async type() {},
      async waitFor() {},
      async assertText() {},
      async currentUrl() {
        return "https://example.com";
      },
    };
    const responses = [
      JSON.stringify({
        action: "click",
        selector: { kind: "role", role: "button", name: "Login" },
        reason: "click the login button",
      }),
      JSON.stringify({ action: "done", reason: "done" }),
    ];
    let i = 0;
    const ollama: OllamaClient = {
      async chat() {
        const content = responses[i] ?? responses[responses.length - 1]!;
        i += 1;
        return {
          model: "mock",
          message: { role: "assistant", content },
          done: true,
        };
      },
    };
    const { script, run } = await generateScript(
      {
        prompt: "press login",
        url: "https://example.com",
        maxSteps: 5,
        wallClockMs: 10_000,
      },
      { browser, ollama },
    );
    expect(run.stopReason).toBe("done");
    expect(script).toContain("import { test, expect } from '@playwright/test';");
    expect(script).toContain("page.getByRole('button', { name: 'Login' }).click()");
  });
});
