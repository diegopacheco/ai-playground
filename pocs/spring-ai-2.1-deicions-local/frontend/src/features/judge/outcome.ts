import type { Tone } from "../../components/Pill";
import type { FindingOutcome } from "./types";

export function findingTone(outcome: FindingOutcome): Tone {
  switch (outcome) {
    case "PASSED":
      return "good";
    case "FAILED":
      return "bad";
    case "INCONCLUSIVE":
    case "NOT_APPLICABLE":
      return "warn";
    case "ERROR":
      return "bad";
    default: {
      const unreachable: never = outcome;
      return unreachable;
    }
  }
}
