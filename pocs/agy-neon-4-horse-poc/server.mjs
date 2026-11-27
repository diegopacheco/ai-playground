import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PRESET_SUMMARY } from './public/js/presets.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PUBLIC_DIR = path.join(__dirname, 'public');
const THREE_DIR = path.join(__dirname, 'node_modules', 'three');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon'
};

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store'
  });
  res.end(body);
}

async function serveFile(res, rootDir, relPath) {
  const safePath = path.normalize(relPath).replace(/^(\.\.(\/|\\|$))+/, '');
  const fullPath = path.join(rootDir, safePath);
  if (!fullPath.startsWith(rootDir)) {
    sendJson(res, 403, { error: 'forbidden' });
    return;
  }
  try {
    const stat = await fs.stat(fullPath);
    if (!stat.isFile()) {
      sendJson(res, 404, { error: 'not_found' });
      return;
    }
    const ext = path.extname(fullPath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    const data = await fs.readFile(fullPath);
    res.writeHead(200, {
      'Content-Type': contentType,
      'Content-Length': data.length,
      'Cache-Control': ext === '.html' ? 'no-store' : 'public, max-age=300'
    });
    res.end(data);
  } catch {
    sendJson(res, 404, { error: 'not_found' });
  }
}

export function createAppServer() {
  return http.createServer(async (req, res) => {
    const reqUrl = new URL(req.url || '/', 'http://localhost');
    const pathname = decodeURIComponent(reqUrl.pathname);

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      sendJson(res, 405, { error: 'method_not_allowed' });
      return;
    }

    if (pathname === '/api/health') {
      sendJson(res, 200, {
        status: 'ok',
        service: 'colorado-summer-horse'
      });
      return;
    }

    if (pathname === '/api/presets') {
      sendJson(res, 200, PRESET_SUMMARY);
      return;
    }

    if (pathname.startsWith('/vendor/three/')) {
      const vendorPath = pathname.slice('/vendor/three/'.length);
      await serveFile(res, THREE_DIR, vendorPath);
      return;
    }

    if (pathname === '/' || pathname === '') {
      await serveFile(res, PUBLIC_DIR, 'index.html');
      return;
    }

    await serveFile(res, PUBLIC_DIR, pathname.slice(1));
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  const port = Number(process.env.PORT || 3410);
  const server = createAppServer();
  server.listen(port, '127.0.0.1', () => {
    process.stdout.write(`colorado-summer-horse listening on http://localhost:${port}\n`);
  });
}
