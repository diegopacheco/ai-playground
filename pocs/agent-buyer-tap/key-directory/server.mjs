import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { genEd25519 } from '../shared/jwk.mjs';

const KEYS_DIR = process.env.KEYS_DIR || '/keys';
const PORT = Number(process.env.PORT || 8801);

function ensureKeys() {
  fs.mkdirSync(KEYS_DIR, { recursive: true });
  const jwksPath = path.join(KEYS_DIR, 'jwks.json');
  if (fs.existsSync(jwksPath)) return;
  const agent = genEd25519();
  const wallet = genEd25519();
  fs.writeFileSync(path.join(KEYS_DIR, 'agent.private.jwk'), JSON.stringify(agent.privateJwk));
  fs.writeFileSync(path.join(KEYS_DIR, 'wallet.private.jwk'), JSON.stringify(wallet.privateJwk));
  fs.writeFileSync(jwksPath, JSON.stringify({ keys: [agent.publicJwk, wallet.publicJwk] }, null, 2));
}

ensureKeys();
const jwks = fs.readFileSync(path.join(KEYS_DIR, 'jwks.json'));

http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200); res.end('ok'); return; }
  if (req.url === '/.well-known/jwks.json') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(jwks);
    return;
  }
  res.writeHead(404); res.end('not found');
}).listen(PORT, () => console.log(`key-directory on ${PORT}`));
