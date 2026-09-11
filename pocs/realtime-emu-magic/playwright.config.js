import { defineConfig } from '@playwright/test'
import { readFileSync } from 'node:fs'

const port = process.env.PORT || readFileSync(new URL('./scripts/ports.env', import.meta.url), 'utf8').match(/^APP=(\d+)$/m)[1]
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './test/browser',
  timeout: 180000,
  expect: { timeout: 60000 },
  workers: 1,
  use: { baseURL, viewport: { width: 1440, height: 1200 }, headless: true },
  webServer: { command: 'node --env-file-if-exists=.env.local server.mjs', url: `${baseURL}/health`, reuseExistingServer: !process.env.CI },
  reporter: 'list'
})
