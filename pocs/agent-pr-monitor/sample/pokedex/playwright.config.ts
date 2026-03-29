import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 60000,
  expect: { timeout: 15000 },
  use: {
    baseURL: 'http://localhost:4200',
    headless: true,
  },
  webServer: {
    command: 'npx ng serve --port 4200',
    port: 4200,
    timeout: 120000,
    reuseExistingServer: true,
  },
});
