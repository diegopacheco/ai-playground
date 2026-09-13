import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';

const port = Number(readFileSync(new URL('./scripts/ports.env', import.meta.url), 'utf8').trim().split('=')[1]);

export default defineConfig({
  server: { host: '127.0.0.1', port, strictPort: true },
  preview: { host: '127.0.0.1', port, strictPort: true },
});
