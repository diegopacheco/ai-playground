import type { Browser, BrowserContext, CDPSession, Page } from "playwright";
import type { BrowserAdapter } from "./browser.ts";
import type { Selector } from "./types.ts";
import { withDisposable } from "./lifecycle.ts";

export interface ScreencastFrame {
  data: string;
  timestamp: number;
}

export interface PlaywrightLaunchOptions {
  headless?: boolean;
  viewport?: { width: number; height: number };
  navigationTimeoutMs?: number;
}

export interface BrowserLauncher {
  launch(options: { headless: boolean }): Promise<Browser>;
}

const DEFAULT_OPTIONS: Required<PlaywrightLaunchOptions> = {
  headless: true,
  viewport: { width: 1280, height: 800 },
  navigationTimeoutMs: 15_000,
};

export class PlaywrightSession implements BrowserAdapter {
  private cdp: CDPSession | undefined;
  private screencastActive = false;
  private constructor(
    private readonly browser: Browser,
    private readonly context: BrowserContext,
    private readonly page: Page,
    private readonly navigationTimeoutMs: number,
  ) {}

  static async launch(
    launcher: BrowserLauncher,
    options: PlaywrightLaunchOptions = {},
  ): Promise<PlaywrightSession> {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const browser = await launcher.launch({ headless: opts.headless });
    let context: BrowserContext | undefined;
    let page: Page | undefined;
    try {
      context = await browser.newContext({ viewport: opts.viewport });
      page = await context.newPage();
      page.setDefaultNavigationTimeout(opts.navigationTimeoutMs);
      return new PlaywrightSession(browser, context, page, opts.navigationTimeoutMs);
    } catch (e) {
      if (page !== undefined) await page.close().catch(() => undefined);
      if (context !== undefined) await context.close().catch(() => undefined);
      await browser.close().catch(() => undefined);
      throw e;
    }
  }

  async goto(url: string): Promise<void> {
    await this.page.goto(url, { waitUntil: "domcontentloaded" });
  }

  async screenshot(): Promise<string> {
    const buffer = await this.page.screenshot({ type: "jpeg", quality: 70 });
    return buffer.toString("base64");
  }

  async click(selector: Selector): Promise<void> {
    await this.locator(selector).click();
  }

  async type(selector: Selector, text: string): Promise<void> {
    await this.locator(selector).fill(text);
  }

  async waitFor(selector: Selector): Promise<void> {
    await this.locator(selector).waitFor();
  }

  async assertText(selector: Selector, text: string): Promise<void> {
    const content = (await this.locator(selector).textContent()) ?? "";
    if (!content.includes(text)) {
      throw new Error(
        `assertText failed: expected ${JSON.stringify(text)} in ${JSON.stringify(content)}`,
      );
    }
  }

  async currentUrl(): Promise<string> {
    return this.page.url();
  }

  async close(): Promise<void> {
    if (this.screencastActive) {
      await this.stopScreencast().catch(() => undefined);
    }
    try {
      await this.page.close();
    } catch {
      // already closed
    }
    try {
      await this.context.close();
    } catch {
      // already closed
    }
    try {
      await this.browser.close();
    } catch {
      // already closed
    }
  }

  async startScreencast(onFrame: (frame: ScreencastFrame) => void): Promise<void> {
    if (this.screencastActive) return;
    if (this.cdp === undefined) {
      this.cdp = await this.context.newCDPSession(this.page);
    }
    const cdp = this.cdp;
    cdp.on("Page.screencastFrame", (event: { data: string; sessionId: number }) => {
      onFrame({ data: event.data, timestamp: Date.now() });
      cdp.send("Page.screencastFrameAck", { sessionId: event.sessionId }).catch(
        () => undefined,
      );
    });
    await cdp.send("Page.startScreencast", {
      format: "jpeg",
      quality: 70,
      everyNthFrame: 1,
    });
    this.screencastActive = true;
  }

  async stopScreencast(): Promise<void> {
    if (!this.screencastActive || this.cdp === undefined) return;
    await this.cdp.send("Page.stopScreencast").catch(() => undefined);
    this.screencastActive = false;
  }

  pageHandle(): Page {
    return this.page;
  }

  private locator(selector: Selector) {
    switch (selector.kind) {
      case "role":
        return selector.name !== undefined
          ? this.page.getByRole(selector.role as Parameters<Page["getByRole"]>[0], {
              name: selector.name,
            })
          : this.page.getByRole(selector.role as Parameters<Page["getByRole"]>[0]);
      case "placeholder":
        return this.page.getByPlaceholder(selector.text);
      case "text":
        return this.page.getByText(selector.text);
      case "label":
        return this.page.getByLabel(selector.text);
      case "test_id":
        return this.page.getByTestId(selector.id);
    }
  }
}

export async function withPlaywrightSession<T>(
  launcher: BrowserLauncher,
  options: PlaywrightLaunchOptions,
  use: (session: PlaywrightSession) => Promise<T>,
): Promise<T> {
  return withDisposable(() => PlaywrightSession.launch(launcher, options), use);
}
