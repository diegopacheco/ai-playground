import { chromium, type Browser } from "playwright";
import type { BrowserLauncher } from "./playwrightAdapter.ts";

export const chromiumLauncher: BrowserLauncher = {
  async launch({ headless }: { headless: boolean }): Promise<Browser> {
    return chromium.launch({ headless });
  },
};
