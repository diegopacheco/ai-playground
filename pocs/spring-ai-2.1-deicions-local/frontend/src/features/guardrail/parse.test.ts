import { describe, expect, test } from "bun:test";
import { parseScreening } from "./parse";
import { outcomeInfo } from "./outcome";

const blocked = {
  battery: "INPUT",
  latencyMs: 34,
  outcome: "BLOCK",
  blocked: true,
  severity: 0.67,
  scores: { jailbreak: 0.79, physical_harm: 0.0, illegal: 0.0, self_harm: 0.05 },
  triggered: ["jailbreak"],
  flagged: [],
  summary: "block [jailbreak]",
};

describe("parseScreening", () => {
  test("a block names the hazard that triggered it", () => {
    const s = parseScreening(blocked);
    expect(s.blocked).toBe(true);
    expect(s.triggered).toEqual(["jailbreak"]);
    expect(outcomeInfo(s.outcome).tone).toBe("bad");
  });

  test("a hazard score that is not a number is refused", () => {
    expect(() => parseScreening({ ...blocked, scores: { jailbreak: "high" } })).toThrow("scores.jailbreak is not a number");
  });
});
