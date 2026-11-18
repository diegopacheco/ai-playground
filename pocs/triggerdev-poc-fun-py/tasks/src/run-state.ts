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

export function startRun(goal: string, provider: string, model: string): RunState {
  return { goal, provider, model, phase: "planning", steps: [], summary: "", error: "" };
}

export function withPlan(state: RunState, tasks: string[]): RunState {
  return { ...state, phase: "working", steps: tasks.map(task => ({ task, status: "pending", output: "" })) };
}

export function stepRunning(state: RunState, index: number): RunState {
  return { ...state, steps: state.steps.map((step, i) => (i === index ? { ...step, status: "running" } : step)) };
}

export function stepDone(state: RunState, index: number, output: string): RunState {
  return { ...state, steps: state.steps.map((step, i) => (i === index ? { ...step, status: "done", output } : step)) };
}

export function summarizing(state: RunState): RunState {
  return { ...state, phase: "summarizing" };
}

export function finished(state: RunState, summary: string): RunState {
  return { ...state, phase: "done", summary };
}

export function failed(state: RunState, error: string): RunState {
  return { ...state, phase: "failed", error };
}

export function completedWork(state: RunState): string {
  return state.steps
    .filter(step => step.status === "done")
    .map((step, i) => `Task ${i + 1}: ${step.task}\nResult: ${step.output}`)
    .join("\n\n");
}

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

export function toJson(state: RunState): { [key: string]: Json } {
  return { ...state, steps: state.steps.map(step => ({ ...step })) };
}
