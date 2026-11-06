import { randomId } from "./b64";
import { generateKeyPair, publicJwk, signJwt, thumbprint } from "./jwt";
import type { EcJwk, JsonObject } from "./types";

const DELEGATION_SECONDS = 3600;

export type Identity = {
  clientId: string;
  rootJwk: EcJwk;
  delegatedKid: string;
  delegatedKey: CryptoKey;
  delegation: string;
  sid: string;
  users: Map<string, string>;
};

export type RequestTarget = { audience: string; method: string; url: string };

const now = (): number => Math.floor(Date.now() / 1000);

export async function createIdentity(clientId: string): Promise<Identity> {
  const root = await generateKeyPair();
  const delegated = await generateKeyPair();
  const rootJwk = await publicJwk(root.publicKey);
  const rootKid = await thumbprint(rootJwk);
  const delegatedJwk = await publicJwk(delegated.publicKey);
  const issuedAt = now();
  const delegation = await signJwt(
    root.privateKey,
    { alg: "ES256", typ: "poppy-delegation+jwt", kid: rootKid },
    { iss: clientId, cnf: { jwk: delegatedJwk }, iat: issuedAt, exp: issuedAt + DELEGATION_SECONDS, jti: randomId("") },
  );
  return {
    clientId,
    rootJwk: { ...rootJwk, kid: rootKid },
    delegatedKid: await thumbprint(delegatedJwk),
    delegatedKey: delegated.privateKey,
    delegation,
    sid: randomId(""),
    users: new Map(),
  };
}

export function userIdFor(identity: Identity, companyId: string): string {
  const known = identity.users.get(companyId);
  if (known) return known;
  const created = randomId("usr_", 9);
  identity.users.set(companyId, created);
  return created;
}

export function newUserFor(identity: Identity, companyId: string): string {
  identity.users.delete(companyId);
  return userIdFor(identity, companyId);
}

export function clientMetadata(identity: Identity): JsonObject {
  return { client_id: identity.clientId, client_name: "Atlas Personal Agent", jwks: { keys: [identity.rootJwk] } };
}

export async function sessionToken(identity: Identity, target: RequestTarget): Promise<string> {
  return signJwt(
    identity.delegatedKey,
    { typ: "poppy-session+jwt", alg: "ES256", kid: identity.delegatedKid, poppy_dlg: identity.delegation },
    {
      iss: identity.clientId,
      aud: target.audience,
      sid: identity.sid,
      sub: userIdFor(identity, target.audience),
      htm: target.method,
      htu: target.url.split("?")[0],
      iat: now(),
      jti: randomId(""),
    },
  );
}
