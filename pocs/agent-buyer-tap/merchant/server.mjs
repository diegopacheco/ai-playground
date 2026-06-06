import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { importPublic } from '../shared/jwk.mjs';
import { verifyRequest } from '../shared/rfc9421.mjs';
import { verifyConsent } from '../shared/consent.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8802);
const DIRECTORY_URL = process.env.DIRECTORY_URL || 'http://key-directory:8801';
const PUBLIC_DIR = path.join(__dirname, 'public');
const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, 'catalog.json')));
const priceBySku = Object.fromEntries(catalog.map((p) => [p.sku, p.priceCents]));

let keyMap = {};
async function refreshKeys() {
  const res = await fetch(`${DIRECTORY_URL}/.well-known/jwks.json`);
  const jwks = await res.json();
  const next = {};
  for (const jwk of jwks.keys) next[jwk.kid] = importPublic(jwk);
  keyMap = next;
}
function resolveKey(kid) { return keyMap[kid]; }

const orders = new Map();
const seenNonce = new Set();
const seenJti = new Set();

const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.map': 'application/json' };

function json(res, code, obj) {
  const body = Buffer.from(JSON.stringify(obj));
  res.writeHead(code, { 'content-type': 'application/json' });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
  });
}

function computeAmount(items) {
  let sum = 0;
  for (const it of items) {
    const pc = priceBySku[it.sku];
    if (pc == null) return null;
    sum += pc * it.qty;
  }
  return sum;
}

function itemsMatch(a, b) {
  const norm = (x) => JSON.stringify([...x].map((i) => ({ sku: i.sku, qty: i.qty })).sort((m, n) => (m.sku < n.sku ? -1 : 1)));
  return norm(a) === norm(b);
}

function handleCatalog(req, res) {
  if (req.headers['signature-input']) {
    const v = verifyRequest({ method: req.method, path: '/api/catalog', authority: req.headers.host, headers: req.headers, resolveKey, seenNonce });
    if (!v.ok) return json(res, 401, { reason: v.reason });
    res.setHeader('x-trusted-agent', v.keyid);
  }
  return json(res, 200, { products: catalog });
}

async function handlePurchase(req, res) {
  const bodyBuf = await readBody(req);
  const v = verifyRequest({ method: req.method, path: '/api/purchase', authority: req.headers.host, headers: req.headers, bodyBuf, resolveKey, seenNonce });
  if (!v.ok) return json(res, 401, { status: 'rejected', reason: v.reason });

  const consent = verifyConsent(req.headers['consent-claim'], resolveKey);
  if (!consent.ok) return json(res, 401, { status: 'rejected', reason: consent.reason });

  let order;
  try { order = JSON.parse(bodyBuf.toString()); } catch { return json(res, 400, { status: 'rejected', reason: 'bad json' }); }

  const now = Math.floor(Date.now() / 1000);
  const claim = consent.payload;
  if (claim.aud !== 'merchant.local') return json(res, 422, { status: 'rejected', reason: 'consent aud mismatch' });
  if (claim.exp <= now) return json(res, 422, { status: 'rejected', reason: 'consent expired' });
  if (seenJti.has(claim.jti)) return json(res, 422, { status: 'rejected', reason: 'consent replay' });

  const computed = computeAmount(order.items || []);
  if (computed == null) return json(res, 422, { status: 'rejected', reason: 'unknown sku' });
  if (computed !== order.amount || claim.amount !== order.amount) return json(res, 422, { status: 'rejected', reason: 'amount mismatch' });
  if (!itemsMatch(order.items, claim.items)) return json(res, 422, { status: 'rejected', reason: 'items mismatch' });

  seenJti.add(claim.jti);
  const id = crypto.randomUUID();
  orders.set(id, {
    id, buyer: claim.sub, items: order.items, amountCents: order.amount, currency: order.currency,
    status: 'accepted', createdAt: new Date().toISOString(), agentKeyid: v.keyid, walletKid: consent.kid
  });
  return json(res, 200, { orderId: id, status: 'accepted' });
}

function handleOrder(res, id) {
  const order = orders.get(id);
  if (!order) return json(res, 404, { reason: 'not found' });
  return json(res, 200, order);
}

function serveStatic(req, res) {
  let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (rel === '/') rel = '/index.html';
  const filePath = path.join(PUBLIC_DIR, rel);
  if (!filePath.startsWith(PUBLIC_DIR)) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(filePath, (err, buf) => {
    if (err) {
      fs.readFile(path.join(PUBLIC_DIR, 'index.html'), (e2, html) => {
        if (e2) { res.writeHead(404); res.end('not found'); return; }
        res.writeHead(200, { 'content-type': 'text/html' });
        res.end(html);
      });
      return;
    }
    res.writeHead(200, { 'content-type': mime[path.extname(filePath)] || 'application/octet-stream' });
    res.end(buf);
  });
}

const server = http.createServer((req, res) => {
  const p = new URL(req.url, 'http://x').pathname;
  if (p === '/health') { res.writeHead(200); res.end('ok'); return; }
  if (p === '/api/catalog' && req.method === 'GET') return handleCatalog(req, res);
  if (p === '/api/purchase' && req.method === 'POST') return handlePurchase(req, res);
  if (p.startsWith('/api/orders/') && req.method === 'GET') return handleOrder(res, p.slice('/api/orders/'.length));
  return serveStatic(req, res);
});

async function start() {
  for (let i = 0; i < 120; i++) {
    try { await refreshKeys(); if (Object.keys(keyMap).length) break; } catch {}
    await new Promise((r) => setTimeout(r, 1000));
  }
  server.listen(PORT, () => console.log(`merchant on ${PORT}`));
}
start();
