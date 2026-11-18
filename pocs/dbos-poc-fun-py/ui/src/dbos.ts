import type { JournalStep, RunState } from "./types.ts";

const FINAL_WORKFLOW_STATUSES = new Set(["SUCCESS", "ERROR", "CANCELLED", "MAX_RECOVERY_ATTEMPTS_EXCEEDED"]);

export function progress(state: RunState): number {
  if (state.phase === "done") return 100;
  const total = state.steps.length + 2;
  const finished = 1 + state.steps.filter(step => step.status === "done").length;
  return state.phase === "planning" ? 0 : Math.round((finished / total) * 100);
}

export function isFinal(state: RunState | null | undefined): boolean {
  return state?.phase === "done" || state?.phase === "failed";
}

export function isFinalWorkflow(status: string | undefined): boolean {
  return status !== undefined && FINAL_WORKFLOW_STATUSES.has(status);
}

export function pillClass(status: string): string {
  if (status === "PENDING") return "running";
  return status.toLowerCase().replaceAll("_", "-");
}

export function stepDuration(step: JournalStep): string {
  if (step.started_at === null || step.completed_at === null) return "";
  const ms = step.completed_at - step.started_at;
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export function isAgentStep(step: JournalStep): boolean {
  return !step.name.startsWith("DBOS.");
}
