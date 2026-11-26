import { describe, expect, test } from "bun:test";
import { ApiError } from "../../http";
import { parseTriage } from "./parse";

const valid = {
  model: "d1-omni:600m",
  latencyMs: 23,
  defect: { yes: false, probability: 0.45 },
  topic: { value: "delivery", probabilities: { delivery: 0.96, product: 0.01, account: 0.01, other: 0.02 } },
  satisfaction: { label: "unhappy", nearestLevel: 0, expectedLevel: 0.38, levels: ["unhappy", "neutral", "happy"], probabilities: [0.68, 0.26, 0.06] },
  route: "HUMAN_REVIEW",
  thresholds: { auto: 0.9, review: 0.4 },
};

describe("parseTriage", () => {
  test("keeps the probability next to the boolean so a coin-flip no stays visible", () => {
    const t = parseTriage(valid);
    expect(t.defect.yes).toBe(false);
    expect(t.defect.probability).toBe(0.45);
    expect(t.route).toBe("HUMAN_REVIEW");
  });

  test("rejects a payload without a probability instead of showing an invented confidence", () => {
    expect(() => parseTriage({ ...valid, defect: { yes: true } })).toThrow(ApiError);
  });

  test("rejects a route the UI does not know how to act on", () => {
    expect(() => parseTriage({ ...valid, route: "ESCALATE" })).toThrow("route has unknown value ESCALATE");
  });
});
