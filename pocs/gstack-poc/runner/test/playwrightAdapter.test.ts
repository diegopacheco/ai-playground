import { describe, test, expect } from "bun:test";
import type { Browser, BrowserContext, Page } from "playwright";
import {
  PlaywrightSession,
  withPlaywrightSession,
  type BrowserLauncher,
} from "../src/playwrightAdapter.ts";

interface CallLog {
  events: string[];
}

function makeFakeStack(log: CallLog, opts: { failOnPageNew?: boolean } = {}): {
  launcher: BrowserLauncher;
  browser: Browser;
  context: BrowserContext;
  page: Page;
} {
  const page = {
    setDefaultNavigationTimeout: (_ms: number) => {
      log.events.push("page.setDefaultNavigationTimeout");
    },
    goto: async (url: string) => {
      log.events.push(`page.goto:${url}`);
    },
    screenshot: async (_opts: { type: string; quality: number }) => {
      log.events.push("page.screenshot");
      return Buffer.from([0xff, 0xd8, 0xff, 0xe0]);
    },
    url: () => "https://example.com/current",
    close: async () => {
      log.events.push("page.close");
    },
    getByRole: () => ({ click: async () => log.events.push("locator.click") }),
    getByPlaceholder: () => ({
      fill: async (t: string) => log.events.push(`locator.fill:${t}`),
    }),
    getByText: () => ({
      textContent: async () => "the inventory is full of Products",
      waitFor: async () => log.events.push("locator.waitFor"),
    }),
    getByLabel: () => ({ fill: async () => undefined }),
    getByTestId: () => ({ click: async () => undefined }),
  } as unknown as Page;

  const context = {
    newPage: async () => {
      if (opts.failOnPageNew) throw new Error("context.newPage failed");
      log.events.push("context.newPage");
      return page;
    },
    close: async () => {
      log.events.push("context.close");
    },
  } as unknown as BrowserContext;

  const browser = {
    newContext: async () => {
      log.events.push("browser.newContext");
      return context;
    },
    close: async () => {
      log.events.push("browser.close");
    },
  } as unknown as Browser;

  const launcher: BrowserLauncher = {
    async launch({ headless }) {
      log.events.push(`launcher.launch:${headless}`);
      return browser;
    },
  };

  return { launcher, browser, context, page };
}

describe("PlaywrightSession", () => {
  test("launches headless by default and seeds page navigation timeout", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const session = await PlaywrightSession.launch(launcher);
    expect(log.events).toContain("launcher.launch:true");
    expect(log.events).toContain("browser.newContext");
    expect(log.events).toContain("context.newPage");
    expect(log.events).toContain("page.setDefaultNavigationTimeout");
    await session.close();
  });

  test("close tears down page → context → browser in order", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const session = await PlaywrightSession.launch(launcher);
    log.events.length = 0;
    await session.close();
    expect(log.events).toEqual(["page.close", "context.close", "browser.close"]);
  });

  test("close is idempotent — second call does not throw", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const session = await PlaywrightSession.launch(launcher);
    await session.close();
    await expect(session.close()).resolves.toBeUndefined();
  });

  test("launch failure during page creation cleans up partial state", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log, { failOnPageNew: true });
    await expect(PlaywrightSession.launch(launcher)).rejects.toThrow(
      "context.newPage failed",
    );
    expect(log.events).toContain("context.close");
    expect(log.events).toContain("browser.close");
  });

  test("goto, screenshot, click, type all flow through Playwright correctly", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const session = await PlaywrightSession.launch(launcher);
    log.events.length = 0;
    await session.goto("https://example.com");
    const shot = await session.screenshot();
    expect(shot).toBe(Buffer.from([0xff, 0xd8, 0xff, 0xe0]).toString("base64"));
    await session.click({ kind: "role", role: "button", name: "Login" });
    await session.type({ kind: "placeholder", text: "Username" }, "standard_user");
    expect(log.events).toContain("page.goto:https://example.com");
    expect(log.events).toContain("page.screenshot");
    expect(log.events).toContain("locator.click");
    expect(log.events).toContain("locator.fill:standard_user");
    await session.close();
  });

  test("assertText throws when content does not include expected substring", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const session = await PlaywrightSession.launch(launcher);
    await expect(
      session.assertText({ kind: "text", text: "Products" }, "NotPresent"),
    ).rejects.toThrow(/assertText failed/);
    await session.close();
  });

  test("assertText passes when content includes expected substring", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const session = await PlaywrightSession.launch(launcher);
    await expect(
      session.assertText({ kind: "text", text: "Products" }, "Products"),
    ).resolves.toBeUndefined();
    await session.close();
  });
});

describe("withPlaywrightSession — critical dispose-on-error guarantee", () => {
  test("closes session when body throws — prevents the leak gap from plan-eng-review", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    await expect(
      withPlaywrightSession(launcher, {}, async () => {
        throw new Error("simulated timeout mid-run");
      }),
    ).rejects.toThrow("simulated timeout mid-run");
    expect(log.events).toContain("page.close");
    expect(log.events).toContain("context.close");
    expect(log.events).toContain("browser.close");
  });

  test("closes session on normal completion", async () => {
    const log: CallLog = { events: [] };
    const { launcher } = makeFakeStack(log);
    const result = await withPlaywrightSession(launcher, {}, async () => 42);
    expect(result).toBe(42);
    expect(log.events).toContain("browser.close");
  });
});
