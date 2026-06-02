export type Scenario = {
  failRate: number;
  latencyMs: number;
  jitterMs: number;
};

export type CallResult = {
  outcome: string;
  latencyMs: number;
};

export type CbMetrics = {
  state: string;
  failureRate: number;
  slowCallRate: number;
  bufferedCalls: number;
  failedCalls: number;
  slowCalls: number;
  successfulCalls: number;
  notPermittedCalls: number;
  ts: number;
};

export type CircuitBreakerSettings = {
  failureRateThreshold: number;
  slowCallRateThreshold: number;
  slowCallDurationThresholdMs: number;
  slidingWindowType: string;
  slidingWindowSize: number;
  minimumNumberOfCalls: number;
  waitDurationInOpenStateSeconds: number;
  permittedNumberOfCallsInHalfOpenState: number;
};

export type FieldClamp = {
  field: string;
  proposed: number;
  applied: number;
  wasClamped: boolean;
};

export type TuneResult = {
  current: CircuitBreakerSettings;
  proposed: CircuitBreakerSettings;
  clamped: CircuitBreakerSettings;
  clamps: FieldClamp[];
  rationale: string;
  model: string;
  metrics: CbMetrics;
};

export type TuneStatus = {
  configured: boolean;
  model: string;
};

export type RunSummary = {
  total: number;
  success: number;
  failure: number;
  shortCircuited: number;
  rateLimited: number;
  rejected: number;
  meanLatencyMs: number;
  p95LatencyMs: number;
};

export type RunResult = RunSummary & {
  id: number;
  label: string;
  failureRatePoints: number[];
};

export const SETTING_KEYS: (keyof CircuitBreakerSettings)[] = [
  "failureRateThreshold",
  "slowCallRateThreshold",
  "slowCallDurationThresholdMs",
  "slidingWindowType",
  "slidingWindowSize",
  "minimumNumberOfCalls",
  "waitDurationInOpenStateSeconds",
  "permittedNumberOfCallsInHalfOpenState",
];
