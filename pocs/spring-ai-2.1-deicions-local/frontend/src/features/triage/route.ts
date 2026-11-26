import type { Tone } from "../../components/Pill";
import type { Zone } from "../../components/ConfidenceLine";
import type { Route, Triage } from "./types";

export type RouteInfo = { readonly label: string; readonly tone: Tone; readonly action: string };

export function routeInfo(route: Route): RouteInfo {
  switch (route) {
    case "AUTO_REPAIR_TICKET":
      return { label: "Auto repair ticket", tone: "good", action: "Very sure of a defect: the ticket opens without a person." };
    case "HUMAN_REVIEW":
      return { label: "Human review", tone: "warn", action: "Unclear: a person reads the message before anything happens." };
    case "NO_ACTION":
      return { label: "No action", tone: "info", action: "Sure there is no defect: nothing to repair." };
    default: {
      const unreachable: never = route;
      return unreachable;
    }
  }
}

export function defectZones(thresholds: Triage["thresholds"]): readonly Zone[] {
  return [
    { from: 0, to: thresholds.review, label: "no action", tone: "good" },
    { from: thresholds.review, to: thresholds.auto, label: "review", tone: "warn" },
    { from: thresholds.auto, to: 1, label: "auto", tone: "bad" },
  ];
}
