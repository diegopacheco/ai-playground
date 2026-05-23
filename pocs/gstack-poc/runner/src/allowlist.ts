export const ALLOWLIST: ReadonlyArray<string> = [
  "www.saucedemo.com",
  "the-internet.herokuapp.com",
  "automationexercise.com",
  "www.automationexercise.com",
  "demo.playwright.dev",
  "todomvc.com",
];

export const SAFETY_BLOCKLIST: ReadonlyArray<string> = [
  "bankofamerica.com",
  "chase.com",
  "wellsfargo.com",
  "paypal.com",
  "stripe.com",
];

export type UrlCheck =
  | { ok: true; host: string; allowlisted: boolean }
  | { ok: false; reason: "malformed_url" | "blocked_by_safety" | "attestation_required" };

export function checkUrl(rawUrl: string, attested: boolean): UrlCheck {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return { ok: false, reason: "malformed_url" };
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return { ok: false, reason: "malformed_url" };
  }
  const host = parsed.host.toLowerCase();
  const hostOnly = parsed.hostname.toLowerCase();

  if (matchesAny(hostOnly, SAFETY_BLOCKLIST) || hostOnly.endsWith(".gov")) {
    return { ok: false, reason: "blocked_by_safety" };
  }

  const allowlisted = ALLOWLIST.includes(host) || ALLOWLIST.includes(hostOnly);
  if (allowlisted) {
    return { ok: true, host, allowlisted: true };
  }
  if (!attested) {
    return { ok: false, reason: "attestation_required" };
  }
  return { ok: true, host, allowlisted: false };
}

function matchesAny(host: string, list: ReadonlyArray<string>): boolean {
  return list.some((entry) => host === entry || host.endsWith(`.${entry}`));
}
