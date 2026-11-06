export function b64urlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function b64urlDecode(text: string): Uint8Array<ArrayBuffer> {
  const padded = text.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (text.length % 4)) % 4);
  return Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
}

export function b64urlJson(value: unknown): string {
  return b64urlEncode(new TextEncoder().encode(JSON.stringify(value)));
}

export function randomId(prefix: string, size = 12): string {
  return prefix + b64urlEncode(crypto.getRandomValues(new Uint8Array(size)));
}
