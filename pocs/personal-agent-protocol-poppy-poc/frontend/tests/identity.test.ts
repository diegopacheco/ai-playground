import { describe, expect, test } from "bun:test";
import { createIdentity, newUserFor, sessionToken, userIdFor } from "../src/poppy/identity";
import { decodeJwt, thumbprint } from "../src/poppy/jwt";
import type { EcJwk } from "../src/poppy/types";

const CLIENT = "http://localhost:8741/agent-host/atlas/agent.json";
const WILLOW = "http://localhost:8741";

describe("identity", () => {
  test("User IDs are pairwise: stable at one Company, different at another, so Companies cannot correlate a User", async () => {
    const identity = await createIdentity(CLIENT);
    const willow = userIdFor(identity, WILLOW);
    expect(userIdFor(identity, WILLOW)).toBe(willow);
    expect(userIdFor(identity, "https://other.example")).not.toBe(willow);
    expect(newUserFor(identity, WILLOW)).not.toBe(willow);
  });

  test("Session JWT is bound to one request and never repeats a jti", async () => {
    const identity = await createIdentity(CLIENT);
    const target = { audience: WILLOW, method: "GET", url: `${WILLOW}/api/rooms?pets=true` };
    const first = decodeJwt(await sessionToken(identity, target));
    const second = decodeJwt(await sessionToken(identity, target));
    expect(first.header.typ).toBe("poppy-session+jwt");
    expect(first.payload).toMatchObject({ iss: CLIENT, aud: WILLOW, htm: "GET", htu: `${WILLOW}/api/rooms`, sub: userIdFor(identity, WILLOW) });
    expect(first.payload.jti).not.toBe(second.payload.jti);
  });

  test("Session kid is the delegated key thumbprint and the delegation is vouched by the published root key", async () => {
    const identity = await createIdentity(CLIENT);
    const session = decodeJwt(await sessionToken(identity, { audience: WILLOW, method: "POST", url: `${WILLOW}/api/holds` }));
    const delegation = decodeJwt(String(session.header.poppy_dlg));
    const cnf = delegation.payload.cnf as { jwk: EcJwk };
    expect(session.header.kid).toBe(await thumbprint(cnf.jwk));
    expect(delegation.header).toMatchObject({ typ: "poppy-delegation+jwt", kid: identity.rootJwk.kid });
    expect(Number(delegation.payload.exp) - Number(delegation.payload.iat)).toBe(3600);
  });
});
