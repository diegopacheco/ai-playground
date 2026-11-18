export type Phase = "planning" | "working" | "summarizing" | "done" | "failed";

export type StepStatus = "pending" | "running" | "done";

export interface Step {
  task: string;
  status: StepStatus;
  output: string;
}

export interface RunState {
  goal: string;
  provider: string;
  model: string;
  phase: Phase;
  steps: Step[];
  summary: string;
  error: string;
}

export interface RunRow {
  runId: string;
  invocationId: string;
  status: string;
  createdAt: string;
}
