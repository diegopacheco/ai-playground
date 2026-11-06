import { describe, expect, test } from "bun:test";
import { b64urlDecode, b64urlEncode } from "../src/poppy/b64";
import { decodeJwt, generateKeyPair, publicJwk, signJwt, thumbprint, verifyJwt } from "../src/poppy/jwt";

describe("jwt", () => {
  test("base64url round trips bytes that need padding and url safe characters", () => {
    const bytes = Uint8Array.from([251, 255, 191, 0, 1]);
    expect(b64urlEncode(bytes)).not.toMatch(/[+/=]/);
    expect([...b64urlDecode(b64urlEncode(bytes))]).toEqual([...bytes]);
  });

  test("thumbprint ignores member order and extra members, so kid matches what the Company computes", async () => {
    const jwk = await publicJwk((await generateKeyPair()).publicKey);
    const reordered = { y: jwk.y, x: jwk.x, crv: jwk.crv, kty: jwk.kty, kid: "ignored" } as const;
    expect(await thumbprint(reordered)).toBe(await thumbprint(jwk));
  });

  test("ES256 signature is raw 64 byte r||s and verifies with the public key only", async () => {
    const pair = await generateKeyPair();
    const token = await signJwt(pair.privateKey, { alg: "ES256" }, { sub: "usr_1" });
    expect(b64urlDecode(token.split(".")[2]).length).toBe(64);
    expect(await verifyJwt(pair.publicKey, token)).toBe(true);
    expect(await verifyJwt((await generateKeyPair()).publicKey, token)).toBe(false);
    expect(decodeJwt(token).payload.sub).toBe("usr_1");
  });
});
