import { describe, expect, test } from "bun:test";
import { parseVerdict } from "./parse";
import { findingTone } from "./outcome";

const failed = {
  latencyMs: 29,
  passed: false,
  findings: [
    { name: "helpfulness", outcome: "PASSED", measure: "level of 0 to 3", value: 1.81, threshold: 2, detail: "" },
    { name: "is_plausible", outcome: "FAILED", measure: "probability of yes", value: 0.21, threshold: 0.5, detail: "" },
  ],
  feedback: "- is_plausible: At least one value is impossible",
};

describe("parseVerdict", () => {
  test("a failed verdict carries the feedback a second try needs", () => {
    const v = parseVerdict(failed);
    expect(v.passed).toBe(false);
    expect(v.feedback).toContain("is_plausible");
    expect(v.findings.map((f) => findingTone(f.outcome))).toEqual(["good", "bad"]);
  });

  test("a finding with an unknown outcome is refused", () => {
    const broken = { ...failed, findings: [{ ...failed.findings[0], outcome: "MAYBE" }] };
    expect(() => parseVerdict(broken)).toThrow("outcome has unknown value MAYBE");
  });
});
