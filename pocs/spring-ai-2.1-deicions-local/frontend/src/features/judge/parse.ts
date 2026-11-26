import { bool, list, num, object, oneOf, str } from "../../parse";
import { FINDING_OUTCOMES, type Finding, type Verdict } from "./types";

function parseFinding(raw: unknown, what: string): Finding {
  const o = object(raw, what);
  return {
    name: str(o, "name"),
    outcome: oneOf(o, "outcome", FINDING_OUTCOMES),
    measure: str(o, "measure"),
    value: num(o, "value"),
    threshold: num(o, "threshold"),
    detail: str(o, "detail"),
  };
}

export function parseVerdict(raw: unknown): Verdict {
  const o = object(raw, "verdict");
  return {
    latencyMs: num(o, "latencyMs"),
    passed: bool(o, "passed"),
    findings: list(o, "findings", parseFinding),
    feedback: str(o, "feedback"),
  };
}
