import { asString, bool, list, num, numbers, object, oneOf, str } from "../../parse";
import { BATTERIES, OUTCOMES, type Screening } from "./types";

export function parseScreening(raw: unknown): Screening {
  const o = object(raw, "screening");
  return {
    battery: oneOf(o, "battery", BATTERIES),
    latencyMs: num(o, "latencyMs"),
    outcome: oneOf(o, "outcome", OUTCOMES),
    blocked: bool(o, "blocked"),
    severity: num(o, "severity"),
    scores: numbers(o, "scores"),
    triggered: list(o, "triggered", asString),
    flagged: list(o, "flagged", asString),
    summary: str(o, "summary"),
  };
}
