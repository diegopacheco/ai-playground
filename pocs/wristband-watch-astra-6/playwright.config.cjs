const { defineConfig } = require('@playwright/test');
const fs = require('node:fs');
const port = fs.readFileSync('scripts/ports.env', 'utf8').match(/^FRONTEND=(\d+)$/m)[1];
module.exports = defineConfig({
  testDir: './tests',
  fullyParallel: false,
  reporter: 'list',
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium', viewport: { width: 1440, height: 1100 }, reducedMotion: 'reduce' },
  webServer: { command: 'node server.mjs', url: `http://127.0.0.1:${port}`, reuseExistingServer: true, timeout: 30000 }
});
