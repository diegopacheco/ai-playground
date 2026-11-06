export type Access = "read-only" | "read-write" | "full";
export type Enforcement = "enforce" | "audit";
export type Decision = "ALLOWED" | "DENIED";

export type Sandbox = {
  name: string;
  phase: string;
  createdAt: string;
  policyVersion: number;
};

export type NetworkRule = {
  ruleName: string;
  binary: string;
  host: string;
  port: number;
  access: Access;
  enforcement: Enforcement;
};

export type ExecResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
};

export type PolicyEvent = {
  timeMs: number;
  kind: string;
  severity: string;
  decision: Decision;
  target: string;
  policy: string;
  engine: string;
  reason: string;
};

export type Verdict = "BLOCKED" | "ALLOWED";

export type Scenario = {
  id: string;
  title: string;
  intent: "attack" | "legit";
  layer: string;
  command: string;
};

export type ScenarioResult = Scenario & {
  verdict: Verdict;
  secure: boolean;
  exitCode: number;
  evidence: string;
};
