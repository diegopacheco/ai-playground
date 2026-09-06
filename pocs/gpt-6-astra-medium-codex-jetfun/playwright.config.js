import { defineConfig } from '@playwright/test';
import { readFileSync } from 'node:fs';
const port = Number(readFileSync(new URL('./scripts/ports.env', import.meta.url), 'utf8').match(/^WEB=(\d+)$/m)[1]);
export default defineConfig({
  testDir: './tests',
  timeout: 60000,
  workers: 1,
  use: { baseURL: `http://localhost:${port}`, viewport: { width: 1440, height: 1080 }, launchOptions: { args: ['--use-angle=swiftshader', '--enable-webgl'] } },
  webServer: { command: 'node server.mjs', url: `http://localhost:${port}`, reuseExistingServer: true, timeout: 30000 },
  reporter: 'list'
});
