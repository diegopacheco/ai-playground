import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  retries: 0,
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  webServer: [
    {
      command: "cd ../backend && cargo run",
      port: 3000,
      timeout: 60000,
      reuseExistingServer: true,
    },
    {
      command: "bun run dev",
      port: 5173,
      timeout: 30000,
      reuseExistingServer: true,
    },
  ],
});
