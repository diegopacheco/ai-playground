import { describe, expect, test } from "bun:test";
import { score } from "./scorecard";
import type { ScenarioResult } from "./types";

const result = (id: string, intent: "attack" | "legit", secure: boolean): ScenarioResult => ({
  id, title: id, intent, layer: "network proxy", command: "x", verdict: secure ? "BLOCKED" : "ALLOWED", secure, exitCode: 0, evidence: "",
});

describe("score", () => {
  test("legit reads never count against the sandbox, only attacks are graded", () => {
    expect(score([result("a", "attack", true), result("read", "legit", true)])).toMatchObject({ attacks: 1, blocked: 1, grade: "A", legit: 1, legitAllowed: 0 });
  });

  test("one attack getting through drops the grade and is listed as a leak", () => {
    const s = score([result("a", "attack", true), result("exfil", "attack", false)]);
    expect(s.grade).toBe("B");
    expect(s.leaks.map((l) => l.id)).toEqual(["exfil"]);
  });

  test("two or more leaks fail the sandbox", () => {
    expect(score([result("a", "attack", false), result("b", "attack", false)]).grade).toBe("F");
  });
});
