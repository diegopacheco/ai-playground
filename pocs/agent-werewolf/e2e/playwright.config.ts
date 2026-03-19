import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  timeout: 30000,
  use: {
    baseURL: "http://localhost:3001",
    screenshot: "only-on-failure",
  },
});
