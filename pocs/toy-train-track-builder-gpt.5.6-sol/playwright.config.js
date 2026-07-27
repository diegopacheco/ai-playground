import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  use: {
    baseURL: 'http://127.0.0.1:5191',
    channel: 'chrome',
    viewport: { width: 1440, height: 900 },
    screenshot: 'only-on-failure'
  },
  webServer: {
    command: 'npm run start -- --host 127.0.0.1 --port 5191',
    url: 'http://127.0.0.1:5191',
    reuseExistingServer: true
  }
})
