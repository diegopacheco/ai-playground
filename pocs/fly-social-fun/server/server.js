import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSimulation, SimulationError } from './simulation.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MOUNTS = [
  { prefix: '/vendor/three/', dir: path.join(ROOT, 'node_modules/three/build') },
  { prefix: '/vendor/addons/', dir: path.join(ROOT, 'node_modules/three/examples/jsm') },
  { prefix: '/', dir: path.join(ROOT, 'public') },
];
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};
const MAX_BODY_BYTES = 10_000;
const HEARTBEAT_MS = 15_000;

function sendJson(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > MAX_BODY_BYTES) reject(new SimulationError('request body too large'));
    });
    req.on('end', () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        reject(new SimulationError('request body is not valid JSON'));
      }
    });
    req.on('error', reject);
  });
}

function resolveStatic(pathname) {
  const mount = MOUNTS.find((candidate) => pathname.startsWith(candidate.prefix));
  let relative;
  try {
    relative = decodeURIComponent(pathname.slice(mount.prefix.length)) || 'index.html';
  } catch {
    return null;
  }
  const file = path.resolve(mount.dir, relative);
  return file.startsWith(mount.dir + path.sep) ? file : null;
}

function serveStatic(res, pathname) {
  const file = resolveStatic(pathname);
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    return sendJson(res, 404, { error: 'not found' });
  }
  res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
}

export function createApp({ simulation = createSimulation(), tickMs = 1500 } = {}) {
  const clients = new Set();
  const timers = [];

  simulation.subscribe((event) => {
    const frame = `data: ${JSON.stringify(event)}\n\n`;
    for (const res of clients) res.write(frame);
  });

  function stream(req, res) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.write(`data: ${JSON.stringify({ type: 'hello', tickMs, ...simulation.snapshot() })}\n\n`);
    clients.add(res);
    req.on('close', () => clients.delete(res));
  }

  async function action(req, res, run) {
    const body = await readJson(req);
    sendJson(res, 202, run(body.spot));
  }

  async function api(req, res, pathname) {
    const profileMatch = pathname.match(/^\/api\/flies\/([\w-]+)$/);
    if (req.method === 'GET' && pathname === '/api/state') return sendJson(res, 200, { tickMs, ...simulation.snapshot() });
    if (req.method === 'GET' && pathname === '/api/stream') return stream(req, res);
    if (req.method === 'GET' && profileMatch) {
      const profile = simulation.profile(profileMatch[1]);
      return profile ? sendJson(res, 200, profile) : sendJson(res, 404, { error: 'fly not found' });
    }
    if (req.method === 'POST' && pathname === '/api/swat') return action(req, res, simulation.swat);
    if (req.method === 'POST' && pathname === '/api/snack') return action(req, res, simulation.dropSnack);
    return sendJson(res, 404, { error: 'not found' });
  }

  async function route(req, res) {
    const { pathname } = new URL(req.url, 'http://localhost');
    if (pathname.startsWith('/api/')) return api(req, res, pathname);
    if (req.method !== 'GET') return sendJson(res, 405, { error: 'method not allowed' });
    return serveStatic(res, pathname);
  }

  const server = http.createServer((req, res) => {
    route(req, res).catch((error) => {
      const status = error instanceof SimulationError ? 400 : 500;
      sendJson(res, status, { error: error.message });
    });
  });

  function listen(port) {
    return new Promise((resolve) => {
      server.listen(port, () => {
        timers.push(setInterval(simulation.tick, tickMs));
        timers.push(setInterval(() => clients.forEach((res) => res.write(': ping\n\n')), HEARTBEAT_MS));
        resolve(server.address().port);
      });
    });
  }

  function close() {
    timers.forEach(clearInterval);
    clients.forEach((res) => res.end());
    clients.clear();
    server.closeAllConnections();
    return new Promise((resolve) => server.close(resolve));
  }

  return { listen, close };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const app = createApp();
  const port = await app.listen(Number(process.env.PORT || 4747));
  console.log(`Buzzr is buzzing on http://localhost:${port}`);
  const shutdown = () => app.close().then(() => process.exit(0));
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}
