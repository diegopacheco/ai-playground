import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  retries: 0,
  use: {
    baseURL: 'http://localhost:5173',
    headless: true,
    screenshot: 'only-on-failure',
  },
  reporter: [['html', { open: 'never' }]],
  webServer: [
    {
      command: 'cd ../backend && cargo run',
      port: 8080,
      reuseExistingServer: true,
      timeout: 60000,
    },
    {
      command: 'npm run dev',
      port: 5173,
      reuseExistingServer: true,
      timeout: 30000,
    },
  ],
})
