import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('.', import.meta.url));

const MOUNTS = [
  ['/vendor/three/', join(ROOT, 'node_modules', 'three') + sep],
  ['/', join(ROOT, 'public') + sep],
];

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

export function resolvePath(url) {
  let path;
  try {
    path = decodeURIComponent(new URL(url, 'http://localhost').pathname);
  } catch {
    return null;
  }
  for (const [prefix, dir] of MOUNTS) {
    if (!path.startsWith(prefix)) continue;
    const rel = path.slice(prefix.length) || 'index.html';
    const full = normalize(join(dir, rel));
    return full.startsWith(dir) ? full : null;
  }
  return null;
}

export function createServer() {
  return http.createServer(async (req, res) => {
    const file = resolvePath(req.url);
    if (!file) {
      res.writeHead(403).end('forbidden');
      return;
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, {
        'Content-Type': TYPES[extname(file)] || 'application/octet-stream',
        'Cache-Control': 'no-cache',
      });
      res.end(body);
    } catch {
      res.writeHead(404).end('not found');
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 7707);
  createServer().listen(port, () => console.log(`rally game on http://localhost:${port}`));
}
