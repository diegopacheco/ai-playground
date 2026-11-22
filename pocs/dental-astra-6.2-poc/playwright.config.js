import { defineConfig } from '@playwright/test'
import { readFileSync } from 'node:fs'
const port = readFileSync(
  new URL('./scripts/ports.env', import.meta.url),
  'utf8',
)
  .trim()
  .split('=')[1]
const baseURL = process.env.ATLAS_URL || 'http://127.0.0.1:' + port
export default defineConfig({
  testDir: 'tests',
  workers: 1,
  use: {
    baseURL,
    viewport: { width: 1320, height: 900 },
    launchOptions: {
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
  webServer: {
    command: 'python3 -m http.server ' + port + ' --bind 127.0.0.1',
    url: baseURL,
    reuseExistingServer: true,
  },
})
