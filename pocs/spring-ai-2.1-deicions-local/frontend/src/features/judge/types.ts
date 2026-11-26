export const FINDING_OUTCOMES = ["PASSED", "FAILED", "INCONCLUSIVE", "ERROR", "NOT_APPLICABLE"] as const;
export type FindingOutcome = (typeof FINDING_OUTCOMES)[number];

export type Finding = {
  readonly name: string;
  readonly outcome: FindingOutcome;
  readonly measure: string;
  readonly value: number;
  readonly threshold: number;
  readonly detail: string;
};

export type Verdict = {
  readonly latencyMs: number;
  readonly passed: boolean;
  readonly findings: readonly Finding[];
  readonly feedback: string;
};
