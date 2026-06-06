import { genEd25519, importPrivate, importPublic } from './jwk.mjs';
import { signRequest, verifyRequest } from './rfc9421.mjs';
import { mintConsent, verifyConsent } from './consent.mjs';

let failures = 0;
const check = (name, cond) => {
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name}`);
  if (!cond) failures++;
};

const agent = genEd25519();
const wallet = genEd25519();
const agentPriv = importPrivate(agent.privateJwk);
const walletPriv = importPrivate(wallet.privateJwk);
const directory = {
  [agent.kid]: importPublic(agent.publicJwk),
  [wallet.kid]: importPublic(wallet.publicJwk)
};
const resolveKey = (kid) => directory[kid];

const url = 'http://merchant:8802/purchase';
const body = Buffer.from(JSON.stringify({ items: [{ sku: 'sku-42', qty: 1 }], amount: 1299, currency: 'USD' }));
const signed = signRequest({ method: 'POST', url, bodyBuf: body, privateKey: agentPriv, keyid: agent.kid });

const seen = new Set();
const good = verifyRequest({
  method: 'POST', path: '/purchase', authority: 'merchant:8802',
  headers: signed, bodyBuf: body, resolveKey, seenNonce: seen
});
check('valid signed purchase verifies', good.ok);

const replay = verifyRequest({
  method: 'POST', path: '/purchase', authority: 'merchant:8802',
  headers: signed, bodyBuf: body, resolveKey, seenNonce: seen
});
check('replayed nonce rejected', !replay.ok && replay.reason === 'replay nonce');

const tamperedBody = Buffer.from(JSON.stringify({ items: [{ sku: 'sku-42', qty: 99 }], amount: 1299, currency: 'USD' }));
const tamper = verifyRequest({
  method: 'POST', path: '/purchase', authority: 'merchant:8802',
  headers: signed, bodyBuf: tamperedBody, resolveKey, seenNonce: new Set()
});
check('tampered body rejected', !tamper.ok && tamper.reason === 'digest mismatch');

const unknown = verifyRequest({
  method: 'POST', path: '/purchase', authority: 'merchant:8802',
  headers: signed, bodyBuf: body, resolveKey: () => null, seenNonce: new Set()
});
check('unknown keyid rejected', !unknown.ok && unknown.reason === 'unknown keyid');

const consent = mintConsent(
  { iss: 'wallet.local', sub: 'buyer-001', aud: 'merchant.local', items: [{ sku: 'sku-42', qty: 1 }], amount: 1299, currency: 'USD', iat: 1, exp: 9999999999, jti: 'abc' },
  walletPriv, wallet.kid
);
const consentOk = verifyConsent(consent, resolveKey);
check('consent verifies with wallet key', consentOk.ok && consentOk.payload.amount === 1299);

const forged = consent.slice(0, -4) + 'AAAA';
const consentBad = verifyConsent(forged, resolveKey);
check('forged consent rejected', !consentBad.ok);

check('agent and wallet kids differ', agent.kid !== wallet.kid);

console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
