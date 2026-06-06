import crypto from 'node:crypto';

export function genEd25519() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519');
  const privateJwk = privateKey.export({ format: 'jwk' });
  const publicJwk = publicKey.export({ format: 'jwk' });
  const kid = thumbprint(publicJwk);
  publicJwk.kid = kid;
  privateJwk.kid = kid;
  return { publicJwk, privateJwk, kid };
}

export function thumbprint(jwk) {
  const ordered = { crv: jwk.crv, kty: jwk.kty, x: jwk.x };
  return crypto.createHash('sha256').update(JSON.stringify(ordered)).digest('base64url');
}

export function importPrivate(jwk) {
  return crypto.createPrivateKey({ key: jwk, format: 'jwk' });
}

export function importPublic(jwk) {
  return crypto.createPublicKey({ key: jwk, format: 'jwk' });
}
