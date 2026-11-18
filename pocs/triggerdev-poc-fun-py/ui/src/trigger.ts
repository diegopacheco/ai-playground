import type { RunState, RunView } from "./types.ts";

const FINAL_RUN_STATUSES = new Set(["COMPLETED", "CANCELED", "FAILED", "CRASHED", "SYSTEM_FAILURE", "EXPIRED", "TIMED_OUT"]);

export function progress(state: RunState): number {
  if (state.phase === "done") return 100;
  const total = state.steps.length + 2;
  const finished = 1 + state.steps.filter(step => step.status === "done").length;
  return state.phase === "planning" ? 0 : Math.round((finished / total) * 100);
}

export function isFinalRun(view: RunView | undefined): boolean {
  if (view === undefined || !FINAL_RUN_STATUSES.has(view.status)) return false;
  return view.status !== "COMPLETED" || view.state === null || view.state.phase === "done";
}

export function pillClass(status: string): string {
  if (status === "COMPLETED") return "done";
  if (FINAL_RUN_STATUSES.has(status)) return "failed";
  return "running";
}
