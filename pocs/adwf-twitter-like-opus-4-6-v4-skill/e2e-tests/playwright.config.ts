import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  retries: 0,
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
  },
  webServer: [
    {
      command: "cd ../backend && cargo run",
      port: 8080,
      timeout: 60000,
      reuseExistingServer: true,
    },
    {
      command: "cd ../frontend && bun run dev",
      port: 5173,
      timeout: 30000,
      reuseExistingServer: true,
    },
  ],
});
