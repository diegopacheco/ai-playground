import { defineConfig } from "playwright/test";
import { readFileSync } from "node:fs";
const port = Number(
  readFileSync(new URL("./scripts/ports.env", import.meta.url), "utf8").match(
    /^WEB=(\d+)$/m,
  )[1],
);
export default defineConfig({
  testDir: "./tests",
  testMatch: "*.spec.mjs",
  timeout: 45000,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: `http://localhost:${port}`,
    headless: true,
    launchOptions: {
      args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    },
    viewport: { width: 1440, height: 1000 },
  },
  webServer: {
    command: "node server.mjs",
    url: `http://localhost:${port}/health`,
    reuseExistingServer: true,
    timeout: 30000,
  },
  projects: [
    { name: "desktop" },
    {
      name: "mobile",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
        deviceScaleFactor: 1,
      },
    },
  ],
});
