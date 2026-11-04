export type CheckResult = {
  type: string;
  expected: unknown;
  passed: boolean;
  detail: string;
};

export type EvalCase = {
  name: string;
  input: string;
  output: string;
  latency_ms: number;
  passed: boolean;
  checks: CheckResult[];
};

export type EvalRun = {
  id: number;
  tenant: string;
  suite: string;
  passed: number;
  failed: number;
  score: number;
  created_at: string;
};

export type EvalRunDetail = EvalRun & { cases: EvalCase[] };

export type Evaluator = {
  name: string;
  description: string;
  value: string;
};

export type Session = {
  clientId: string;
  accessToken: string;
  expiresAt: number;
  scopes: string[];
};

export type ClientKind = "skill" | "mcp" | "admin" | "test" | "unknown";

export type ClientInfo = {
  client_id: string;
  kind: ClientKind;
  description: string;
  scopes: string[];
};

export type LocalCredential = ClientInfo & { client_secret: string };

export type StateSummary = {
  tenant: string;
  keys: number;
  last_updated: string;
  recent_keys: string[];
};

export type LogCount = { tenant: string; level: string; count: number };

export type LogLine = {
  tenant: string;
  timestamp_ns: string;
  level: string;
  message: string;
  fields: Record<string, unknown>;
};

export type LogsSummary = { since_minutes: number; counts: LogCount[]; recent: LogLine[] };

export type StateEntry = { tenant: string; key: string; value: unknown; updated_at: string };

export type Me = {
  client_id: string;
  kind: ClientKind;
  description: string;
  scopes: string[];
  iat: number;
  exp: number;
  iss: string;
};
