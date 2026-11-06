import { b64urlDecode, b64urlEncode, b64urlJson } from "./b64";
import type { EcJwk, JsonObject } from "./types";

const ES256: EcdsaParams = { name: "ECDSA", hash: "SHA-256" };

export async function generateKeyPair(): Promise<CryptoKeyPair> {
  return crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
}

export async function publicJwk(key: CryptoKey): Promise<EcJwk> {
  const { x, y } = await crypto.subtle.exportKey("jwk", key);
  if (!x || !y) throw new Error("not an EC public key");
  return { kty: "EC", crv: "P-256", x, y };
}

export async function thumbprint(jwk: EcJwk): Promise<string> {
  const canonical = JSON.stringify({ crv: jwk.crv, kty: jwk.kty, x: jwk.x, y: jwk.y });
  return b64urlEncode(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical))));
}

export async function signJwt(key: CryptoKey, header: JsonObject, payload: JsonObject): Promise<string> {
  const input = `${b64urlJson(header)}.${b64urlJson(payload)}`;
  const signature = await crypto.subtle.sign(ES256, key, new TextEncoder().encode(input));
  return `${input}.${b64urlEncode(new Uint8Array(signature))}`;
}

export async function verifyJwt(key: CryptoKey, token: string): Promise<boolean> {
  const [head, body, signature] = token.split(".");
  return crypto.subtle.verify(ES256, key, b64urlDecode(signature), new TextEncoder().encode(`${head}.${body}`));
}

export type DecodedJwt = { header: JsonObject; payload: JsonObject };

export function decodeJwt(token: string): DecodedJwt {
  const [head, body] = token.split(".");
  const parse = (part: string): JsonObject => JSON.parse(new TextDecoder().decode(b64urlDecode(part)));
  return { header: parse(head), payload: parse(body) };
}
