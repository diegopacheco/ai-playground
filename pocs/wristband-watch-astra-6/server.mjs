import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('.', import.meta.url));
const config = await readFile(new URL('scripts/ports.env', import.meta.url), 'utf8');
const port = Number(config.match(/^FRONTEND=(\d+)$/m)?.[1]);
if (!port) throw new Error('FRONTEND port is missing in scripts/ports.env');
const types = { html: 'text/html; charset=utf-8', svg: 'image/svg+xml', png: 'image/png' };
http.createServer(async (request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  const path = pathname === '/' ? 'index.html' : pathname.slice(1);
  if (path !== 'index.html' && !/^assets\/[a-z0-9-]+\.svg$/.test(path) && !/^printscreens\/[a-z0-9-]+\.png$/.test(path)) {
    response.writeHead(404).end('Not found');
    return;
  }
  try {
    const body = await readFile(root + path);
    response.writeHead(200, { 'Content-Type': types[path.split('.').pop()], 'Cache-Control': 'no-store' });
    response.end(body);
  } catch {
    response.writeHead(404).end('Not found');
  }
}).listen(port, '127.0.0.1', () => process.stdout.write(`ASTRA / ONE at http://127.0.0.1:${port}\n`));
