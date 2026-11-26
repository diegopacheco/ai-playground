import { asNumber, asString, bool, list, num, numbers, object, oneOf, str } from "../../parse";
import { ROUTES, type Triage } from "./types";

export function parseTriage(raw: unknown): Triage {
  const o = object(raw, "triage");
  const defect = object(o.defect, "defect");
  const topic = object(o.topic, "topic");
  const satisfaction = object(o.satisfaction, "satisfaction");
  const thresholds = object(o.thresholds, "thresholds");
  return {
    model: str(o, "model"),
    latencyMs: num(o, "latencyMs"),
    defect: { yes: bool(defect, "yes"), probability: num(defect, "probability") },
    topic: { value: str(topic, "value"), probabilities: numbers(topic, "probabilities") },
    satisfaction: {
      label: str(satisfaction, "label"),
      nearestLevel: num(satisfaction, "nearestLevel"),
      expectedLevel: num(satisfaction, "expectedLevel"),
      levels: list(satisfaction, "levels", asString),
      probabilities: list(satisfaction, "probabilities", asNumber),
    },
    route: oneOf(o, "route", ROUTES),
    thresholds: { auto: num(thresholds, "auto"), review: num(thresholds, "review") },
  };
}
