import type { Tone } from "../../components/Pill";
import type { Outcome } from "./types";

export function outcomeInfo(outcome: Outcome): { readonly tone: Tone; readonly meaning: string } {
  switch (outcome) {
    case "PASS":
      return { tone: "good", meaning: "No hazard: the chat model may be called." };
    case "REVIEW":
      return { tone: "warn", meaning: "A hazard sits in the borderline band: it passes unless block-on-review is on." };
    case "BLOCK":
      return { tone: "bad", meaning: "A hazard is likely: the turn is refused before the chat model runs." };
    case "SUPPORT":
      return { tone: "info", meaning: "Signs of self-harm: the user gets a support message instead of an answer." };
    default: {
      const unreachable: never = outcome;
      return unreachable;
    }
  }
}
