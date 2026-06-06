import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { chromium } from 'playwright';
import { importPrivate } from '../shared/jwk.mjs';
import { signRequest } from '../shared/rfc9421.mjs';
import { mintConsent } from '../shared/consent.mjs';

const MERCHANT_URL = process.env.MERCHANT_URL || 'http://merchant:8802';
const KEYS_DIR = process.env.KEYS_DIR || '/keys';
const SHOTS_DIR = process.env.PRINTSCREENS_DIR || '/printscreens';
const MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(fn, name) {
  for (let i = 0; i < 120; i++) {
    try { if (await fn()) return; } catch {}
    await sleep(1000);
  }
  throw new Error(`timeout waiting for ${name}`);
}

function loadKey(file) {
  const jwk = JSON.parse(fs.readFileSync(path.join(KEYS_DIR, file)));
  return { key: importPrivate(jwk), kid: jwk.kid };
}

async function browse(agent) {
  const url = `${MERCHANT_URL}/api/catalog`;
  const headers = signRequest({ method: 'GET', url, privateKey: agent.key, keyid: agent.kid });
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`signed browse rejected: ${res.status}`);
  console.log(`signed browse ok, trusted-agent header: ${res.headers.get('x-trusted-agent')}`);
  return (await res.json()).products;
}

async function selectProduct(products) {
  const inStock = products.filter((p) => p.stock > 0);
  if (!process.env.OPENAI_API_KEY) {
    console.log('OPENAI_API_KEY not set, using deterministic selection');
    return { sku: inStock[0].sku, qty: 1 };
  }
  try {
    const { default: OpenAI } = await import('openai');
    const client = new OpenAI();
    const resp = await client.chat.completions.create({
      model: MODEL,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: 'You are a buyer agent shopping for one item under 60 dollars. Reply only with JSON {"sku":"<sku>","qty":1} choosing one sku from the list.' },
        { role: 'user', content: JSON.stringify(inStock.map((p) => ({ sku: p.sku, name: p.name, priceCents: p.priceCents }))) }
      ]
    });
    const choice = JSON.parse(resp.choices[0].message.content);
    if (!inStock.find((p) => p.sku === choice.sku)) throw new Error('model picked unknown sku');
    console.log(`OpenAI (${MODEL}) chose ${choice.sku}`);
    return { sku: choice.sku, qty: choice.qty || 1 };
  } catch (err) {
    console.log(`OpenAI selection failed (${err.message}), using deterministic selection`);
    return { sku: inStock[0].sku, qty: 1 };
  }
}

async function purchase(agent, wallet, items, amount, currency) {
  const now = Math.floor(Date.now() / 1000);
  const consentToken = mintConsent({
    iss: 'wallet.local', sub: 'buyer-001', aud: 'merchant.local',
    items, amount, currency, iat: now, exp: now + 300, jti: crypto.randomUUID()
  }, wallet.key, wallet.kid);

  const url = `${MERCHANT_URL}/api/purchase`;
  const bodyBuf = Buffer.from(JSON.stringify({ items, amount, currency }));
  const headers = signRequest({ method: 'POST', url, bodyBuf, privateKey: agent.key, keyid: agent.kid });
  headers['content-type'] = 'application/json';
  headers['consent-claim'] = consentToken;

  const res = await fetch(url, { method: 'POST', headers, body: bodyBuf });
  return res.json();
}

async function main() {
  fs.mkdirSync(SHOTS_DIR, { recursive: true });
  await waitFor(() => fs.existsSync(path.join(KEYS_DIR, 'agent.private.jwk')) && fs.existsSync(path.join(KEYS_DIR, 'wallet.private.jwk')), 'keys');
  await waitFor(async () => (await fetch(`${MERCHANT_URL}/health`)).ok, 'merchant');

  const agent = loadKey('agent.private.jwk');
  const wallet = loadKey('wallet.private.jwk');

  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(MERCHANT_URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.card');
  const seen = await page.$$eval('.card', (els) => els.map((e) => ({ sku: e.getAttribute('data-sku'), priceCents: Number(e.getAttribute('data-price')) })));
  console.log(`navigated storefront, saw ${seen.length} products`);
  await page.screenshot({ path: path.join(SHOTS_DIR, '01-storefront.png'), fullPage: true });

  const products = await browse(agent);
  const choice = await selectProduct(products);
  const product = products.find((p) => p.sku === choice.sku);
  const items = [{ sku: choice.sku, qty: choice.qty }];
  const amount = product.priceCents * choice.qty;
  console.log(`buying ${choice.qty} x ${product.name} for ${amount} ${product.currency}`);

  const result = await purchase(agent, wallet, items, amount, product.currency);
  if (result.status !== 'accepted') {
    console.log(`PURCHASE REJECTED reason=${result.reason}`);
    await browser.close();
    process.exit(1);
  }

  await page.goto(`${MERCHANT_URL}/orders/${result.orderId}`, { waitUntil: 'networkidle' });
  await page.waitForSelector('.confirm');
  await page.screenshot({ path: path.join(SHOTS_DIR, '02-confirmation.png'), fullPage: true });
  console.log(`PURCHASE ACCEPTED orderId=${result.orderId}`);
  await browser.close();
}

main().catch((err) => { console.error(err); process.exit(1); });
