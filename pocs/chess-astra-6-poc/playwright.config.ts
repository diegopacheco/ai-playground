import { defineConfig } from '@playwright/test';
import { readFileSync } from 'node:fs';

const port = Number(readFileSync(new URL('./scripts/ports.env', import.meta.url), 'utf8').trim().split('=')[1]);
export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.ts',
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 1100 },
    launchOptions: { args: ['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader'] },
    trace: 'retain-on-failure',
  },
  webServer: { command: process.env.PLAYWRIGHT_PREVIEW ? 'bun run preview' : 'bun run dev', url: `http://127.0.0.1:${port}`, reuseExistingServer: !process.env.CI, timeout: 30000 },
});
