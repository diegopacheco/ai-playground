import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  use: {
    baseURL: 'http://127.0.0.1:5188',
    channel: 'chrome',
    viewport: { width: 1440, height: 900 },
    screenshot: 'only-on-failure'
  },
  webServer: {
    command: 'python3 -m http.server 5188 --bind 127.0.0.1',
    url: 'http://127.0.0.1:5188',
    reuseExistingServer: true
  }
})
