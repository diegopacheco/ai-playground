import type { ScenarioResult } from "./types";

export type Score = {
  attacks: number;
  blocked: number;
  leaks: ScenarioResult[];
  legit: number;
  legitAllowed: number;
  grade: "A" | "B" | "F";
};

export function score(results: ScenarioResult[]): Score {
  const attacks = results.filter((r) => r.intent === "attack");
  const leaks = attacks.filter((r) => !r.secure);
  const blocked = attacks.length - leaks.length;
  const grade = leaks.length === 0 ? "A" : leaks.length === 1 ? "B" : "F";
  const legit = results.filter((r) => r.intent === "legit");
  const legitAllowed = legit.filter((r) => r.verdict === "ALLOWED").length;
  return { attacks: attacks.length, blocked, leaks, grade, legit: legit.length, legitAllowed };
}
