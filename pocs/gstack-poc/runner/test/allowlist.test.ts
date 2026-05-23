import { describe, test, expect } from "bun:test";
import { checkUrl, ALLOWLIST, SAFETY_BLOCKLIST } from "../src/allowlist.ts";

describe("checkUrl", () => {
  test("allowlisted host passes without attestation", () => {
    const result = checkUrl("https://www.saucedemo.com/", false);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.allowlisted).toBe(true);
      expect(result.host).toBe("www.saucedemo.com");
    }
  });

  test("allowlisted host with path still passes", () => {
    const result = checkUrl(
      "https://the-internet.herokuapp.com/login",
      false,
    );
    expect(result.ok).toBe(true);
  });

  test("non-allowlisted host with attestation passes", () => {
    const result = checkUrl("https://example.com/my-app", true);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.allowlisted).toBe(false);
      expect(result.host).toBe("example.com");
    }
  });

  test("non-allowlisted host without attestation requires attestation", () => {
    const result = checkUrl("https://example.com/my-app", false);
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.reason).toBe("attestation_required");
    }
  });

  test("malformed URL is rejected before any allowlist check", () => {
    const result = checkUrl("not a url at all", true);
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.reason).toBe("malformed_url");
    }
  });

  test("non-http(s) schemes are rejected", () => {
    const fileResult = checkUrl("file:///etc/passwd", true);
    expect(fileResult.ok).toBe(false);
    const ftpResult = checkUrl("ftp://files.example.com", true);
    expect(ftpResult.ok).toBe(false);
  });

  test("safety blocklist overrides attestation", () => {
    const result = checkUrl("https://www.bankofamerica.com/login", true);
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.reason).toBe("blocked_by_safety");
    }
  });

  test("safety blocklist matches subdomains", () => {
    const result = checkUrl("https://login.chase.com/", true);
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.reason).toBe("blocked_by_safety");
    }
  });

  test(".gov is always blocked even with attestation", () => {
    const result = checkUrl("https://my-app.gov/", true);
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.reason).toBe("blocked_by_safety");
    }
  });

  test("host comparison is case-insensitive", () => {
    const result = checkUrl("https://WWW.SAUCEDEMO.COM/", false);
    expect(result.ok).toBe(true);
  });

  test("ALLOWLIST exposes a non-empty curated set", () => {
    expect(ALLOWLIST.length).toBeGreaterThan(0);
    expect(ALLOWLIST).toContain("www.saucedemo.com");
  });

  test("SAFETY_BLOCKLIST exposes a non-empty curated set", () => {
    expect(SAFETY_BLOCKLIST.length).toBeGreaterThan(0);
    expect(SAFETY_BLOCKLIST).toContain("bankofamerica.com");
  });
});
