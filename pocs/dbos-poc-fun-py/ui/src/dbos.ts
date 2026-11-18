import type { RunState } from "./types.ts";

export function progress(state: RunState): number {
  if (state.phase === "done") return 100;
  const total = state.steps.length + 2;
  const finished = 1 + state.steps.filter(step => step.status === "done").length;
  return state.phase === "planning" ? 0 : Math.round((finished / total) * 100);
}

export function isFinal(state: RunState | null | undefined): boolean {
  return state?.phase === "done" || state?.phase === "failed";
}

export function pillClass(status: string): string {
  if (status === "PENDING") return "running";
  return status.toLowerCase().replaceAll("_", "-");
}
