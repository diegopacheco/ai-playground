import { describe, test, expect } from "bun:test";
import { EVAL_CASES } from "../eval/cases.ts";
import { checkUrl } from "../src/index.ts";

describe("eval cases (structural — does not run the model)", () => {
  test("has exactly 10 cases", () => {
    expect(EVAL_CASES.length).toBe(10);
  });

  test("every case has a unique id", () => {
    const ids = new Set(EVAL_CASES.map((c) => c.id));
    expect(ids.size).toBe(EVAL_CASES.length);
  });

  test("every case has a non-empty prompt and url", () => {
    for (const c of EVAL_CASES) {
      expect(c.prompt.length).toBeGreaterThan(10);
      expect(c.url.startsWith("https://")).toBe(true);
    }
  });

  test("every case targets an allowlisted URL — no attestation needed for evals", () => {
    for (const c of EVAL_CASES) {
      const verdict = checkUrl(c.url, false);
      expect(verdict.ok).toBe(true);
      if (verdict.ok) {
        expect(verdict.allowlisted).toBe(true);
      }
    }
  });

  test("every case has at least one expectInScript needle", () => {
    for (const c of EVAL_CASES) {
      expect(c.expectInScript.length).toBeGreaterThan(0);
    }
  });
});
