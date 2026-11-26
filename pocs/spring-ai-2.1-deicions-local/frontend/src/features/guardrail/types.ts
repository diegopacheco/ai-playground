export const BATTERIES = ["INPUT", "OUTPUT"] as const;
export type Battery = (typeof BATTERIES)[number];

export const OUTCOMES = ["PASS", "REVIEW", "BLOCK", "SUPPORT"] as const;
export type Outcome = (typeof OUTCOMES)[number];

export type Screening = {
  readonly battery: Battery;
  readonly latencyMs: number;
  readonly outcome: Outcome;
  readonly blocked: boolean;
  readonly severity: number;
  readonly scores: Readonly<Record<string, number>>;
  readonly triggered: readonly string[];
  readonly flagged: readonly string[];
  readonly summary: string;
};
