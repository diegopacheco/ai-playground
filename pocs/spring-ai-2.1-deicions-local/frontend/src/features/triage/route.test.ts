import { describe, expect, test } from "bun:test";
import { defectZones, routeInfo } from "./route";

describe("defectZones", () => {
  test("the review band sits between the two backend thresholds", () => {
    const zones = defectZones({ auto: 0.9, review: 0.4 });
    expect(zones.map((z) => [z.from, z.to])).toEqual([
      [0, 0.4],
      [0.4, 0.9],
      [0.9, 1],
    ]);
  });
});

describe("routeInfo", () => {
  test("only a sure defect is presented as acting without a person", () => {
    expect(routeInfo("AUTO_REPAIR_TICKET").action).toContain("without a person");
    expect(routeInfo("HUMAN_REVIEW").action).toContain("a person reads");
    expect(routeInfo("HUMAN_REVIEW").tone).toBe("warn");
  });
});
