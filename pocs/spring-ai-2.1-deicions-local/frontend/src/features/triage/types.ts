export const ROUTES = ["AUTO_REPAIR_TICKET", "HUMAN_REVIEW", "NO_ACTION"] as const;
export type Route = (typeof ROUTES)[number];

export type Triage = {
  readonly model: string;
  readonly latencyMs: number;
  readonly defect: { readonly yes: boolean; readonly probability: number };
  readonly topic: { readonly value: string; readonly probabilities: Readonly<Record<string, number>> };
  readonly satisfaction: {
    readonly label: string;
    readonly nearestLevel: number;
    readonly expectedLevel: number;
    readonly levels: readonly string[];
    readonly probabilities: readonly number[];
  };
  readonly route: Route;
  readonly thresholds: { readonly auto: number; readonly review: number };
};
