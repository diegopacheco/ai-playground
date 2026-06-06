import crypto from 'node:crypto';

const b64u = (buf) => Buffer.from(buf).toString('base64url');
const b64uJson = (obj) => b64u(Buffer.from(JSON.stringify(obj)));

export function mintConsent(payload, walletPrivateKey, kid) {
  const header = { alg: 'EdDSA', typ: 'JWT', kid };
  const signingInput = `${b64uJson(header)}.${b64uJson(payload)}`;
  const signature = crypto.sign(null, Buffer.from(signingInput), walletPrivateKey).toString('base64url');
  return `${signingInput}.${signature}`;
}

export function verifyConsent(token, resolveKey) {
  const parts = (token || '').split('.');
  if (parts.length !== 3) return { ok: false, reason: 'malformed consent' };
  const [h, p, s] = parts;
  const header = JSON.parse(Buffer.from(h, 'base64url'));
  const pub = resolveKey(header.kid);
  if (!pub) return { ok: false, reason: 'unknown consent kid' };
  const ok = crypto.verify(null, Buffer.from(`${h}.${p}`), pub, Buffer.from(s, 'base64url'));
  if (!ok) return { ok: false, reason: 'bad consent signature' };
  return { ok: true, payload: JSON.parse(Buffer.from(p, 'base64url')), kid: header.kid };
}
