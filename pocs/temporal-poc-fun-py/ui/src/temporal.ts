import type { RunState, RunView } from "./types.ts";

const CLOSED_STATUSES = new Set(["COMPLETED", "FAILED", "CANCELED", "TERMINATED", "CONTINUED_AS_NEW", "TIMED_OUT"]);

export function progress(state: RunState): number {
  if (state.phase === "done") return 100;
  const total = state.steps.length + 2;
  const finished = 1 + state.steps.filter(step => step.status === "done").length;
  return state.phase === "planning" ? 0 : Math.round((finished / total) * 100);
}

export function isFinalRun(view: RunView | undefined): boolean {
  return view !== undefined && CLOSED_STATUSES.has(view.status);
}

export function pillClass(status: string): string {
  if (status === "COMPLETED") return "done";
  if (CLOSED_STATUSES.has(status)) return "failed";
  return "running";
}
