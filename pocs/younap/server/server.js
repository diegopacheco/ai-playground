import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { search } from './youtube.js';

const PUBLIC = fileURLToPath(new URL('../public', import.meta.url));

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml'
};

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

async function handleSearch(url, res, searchFn) {
  const query = (url.searchParams.get('q') || '').trim();
  if (!query) return sendJson(res, 400, { error: 'q is required' });
  try {
    sendJson(res, 200, await searchFn(query));
  } catch (error) {
    sendJson(res, 502, { error: error.message });
  }
}

async function handleStatic(url, res) {
  const path = normalize(join(PUBLIC, url.pathname === '/' ? 'index.html' : url.pathname));
  if (!path.startsWith(PUBLIC)) return sendJson(res, 403, { error: 'forbidden' });
  try {
    const body = await readFile(path);
    res.writeHead(200, { 'Content-Type': TYPES[extname(path)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    sendJson(res, 404, { error: 'not found' });
  }
}

export function createApp(searchFn = search) {
  return createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/api/search') return handleSearch(url, res, searchFn);
    return handleStatic(url, res);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT);
  if (!port) {
    console.error('PORT is required');
    process.exit(1);
  }
  createApp().listen(port, () => console.log(`younap listening on http://localhost:${port}`));
}
