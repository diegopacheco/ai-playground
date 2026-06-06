import crypto from 'node:crypto';

const TAG = 'visa-tap';
const ALG = 'ed25519';

export function contentDigest(bodyBuf) {
  const hash = crypto.createHash('sha256').update(bodyBuf).digest('base64');
  return `sha-256=:${hash}:`;
}

function signatureParams(components, params) {
  const list = components.map((c) => `"${c}"`).join(' ');
  return `(${list})` +
    `;created=${params.created}` +
    `;keyid="${params.keyid}"` +
    `;nonce="${params.nonce}"` +
    `;tag="${params.tag}"` +
    `;alg="${params.alg}"`;
}

function componentValue(name, ctx) {
  if (name === '@method') return ctx.method.toUpperCase();
  if (name === '@path') return ctx.path;
  if (name === '@authority') return ctx.authority;
  return ctx.headers[name];
}

function signatureBase(components, sigParams, ctx) {
  const lines = components.map((c) => `"${c}": ${componentValue(c, ctx)}`);
  lines.push(`"@signature-params": ${sigParams}`);
  return lines.join('\n');
}

function parseSignatureInput(value) {
  const eq = value.indexOf('=');
  const sigParams = value.slice(eq + 1).trim();
  const inner = sigParams.match(/\(([^)]*)\)/)[1];
  const components = inner.length ? inner.split(' ').map((s) => s.replace(/"/g, '')) : [];
  const grab = (re) => (sigParams.match(re) || [])[1];
  return {
    sigParams,
    components,
    created: grab(/created=(\d+)/),
    keyid: grab(/keyid="([^"]*)"/),
    nonce: grab(/nonce="([^"]*)"/),
    tag: grab(/tag="([^"]*)"/),
    alg: grab(/alg="([^"]*)"/)
  };
}

function parseSignature(value) {
  return (value.match(/sig1=:([^:]*):/) || [])[1];
}

export function signRequest({ method, url, bodyBuf, privateKey, keyid }) {
  const u = new URL(url);
  const headers = {};
  const components = ['@method', '@path', '@authority'];
  if (bodyBuf && bodyBuf.length) {
    headers['content-digest'] = contentDigest(bodyBuf);
    components.push('content-digest');
  }
  const params = {
    created: Math.floor(Date.now() / 1000),
    keyid,
    nonce: crypto.randomBytes(16).toString('hex'),
    tag: TAG,
    alg: ALG
  };
  const sigParams = signatureParams(components, params);
  const ctx = { method, path: u.pathname, authority: u.host, headers };
  const base = signatureBase(components, sigParams, ctx);
  const signature = crypto.sign(null, Buffer.from(base, 'utf8'), privateKey).toString('base64');
  headers['signature-input'] = `sig1=${sigParams}`;
  headers['signature'] = `sig1=:${signature}:`;
  return headers;
}

export function verifyRequest({ method, path, authority, headers, bodyBuf, resolveKey, seenNonce, window = 60 }) {
  const si = headers['signature-input'];
  const sg = headers['signature'];
  if (!si || !sg) return { ok: false, reason: 'missing signature headers' };
  const p = parseSignatureInput(si);
  if (p.tag !== TAG) return { ok: false, reason: 'bad tag' };
  if (p.alg !== ALG) return { ok: false, reason: 'bad alg' };
  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - Number(p.created)) > window) return { ok: false, reason: 'stale created' };
  if (seenNonce && seenNonce.has(p.nonce)) return { ok: false, reason: 'replay nonce' };
  if (p.components.includes('content-digest')) {
    const expected = contentDigest(bodyBuf || Buffer.alloc(0));
    if (headers['content-digest'] !== expected) return { ok: false, reason: 'digest mismatch' };
  }
  const pub = resolveKey(p.keyid);
  if (!pub) return { ok: false, reason: 'unknown keyid' };
  const base = signatureBase(p.components, p.sigParams, { method, path, authority, headers });
  const signature = Buffer.from(parseSignature(sg), 'base64');
  const ok = crypto.verify(null, Buffer.from(base, 'utf8'), pub, signature);
  if (ok && seenNonce) seenNonce.add(p.nonce);
  return { ok, reason: ok ? 'ok' : 'bad signature', keyid: p.keyid, params: p };
}
