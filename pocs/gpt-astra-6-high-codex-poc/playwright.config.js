import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: 'browser.spec.js',
  timeout: 30000,
  use: { baseURL: 'http://localhost:3100', headless: true, viewport: { width: 1440, height: 1120 } },
  webServer: {
    command: 'bash build.sh && node server.js dist',
    env: { PORT: '3100' },
    url: 'http://localhost:3100',
    timeout: 30000,
    reuseExistingServer: false
  },
  reporter: 'list'
});
