import type { EventRow, StepRow, Stored, WorkflowDetail, WorkflowRow } from "./types.ts";

const FINAL_STATUSES = new Set(["SUCCESS", "ERROR", "CANCELLED", "MAX_RECOVERY_ATTEMPTS_EXCEEDED"]);

export type BarKind = "agent" | "internal" | "error" | "running";

export interface Bar {
  key: string;
  label: string;
  kind: BarKind;
  left: number;
  width: number;
  duration: string;
}

export function isFinal(status: string | undefined): boolean {
  return status !== undefined && FINAL_STATUSES.has(status);
}

export function pillClass(status: string): string {
  if (status === "PENDING") return "running";
  return status.toLowerCase().replaceAll("_", "-");
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.floor((ms % 60_000) / 1000);
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

export function workflowEnd(workflow: WorkflowRow, now: number): number {
  if (workflow.completed_at !== null) return workflow.completed_at;
  return isFinal(workflow.status) ? workflow.updated_at : now;
}

export function isInternal(step: StepRow): boolean {
  return step.name.startsWith("DBOS.");
}

export function stepDuration(step: StepRow): string {
  if (step.started_at === null || step.completed_at === null) return "";
  return formatDuration(step.completed_at - step.started_at);
}

function lastFinish(detail: WorkflowDetail): number {
  return detail.steps.reduce((latest, step) => Math.max(latest, step.completed_at ?? 0), detail.workflow.created_at);
}

export function timeline(detail: WorkflowDetail, now: number, showInternal: boolean): Bar[] {
  const start = detail.workflow.created_at;
  const span = Math.max(1, workflowEnd(detail.workflow, now) - start);
  const place = (from: number, to: number) => ({
    left: ((from - start) / span) * 100,
    width: Math.max(0.6, ((to - from) / span) * 100),
    duration: formatDuration(to - from),
  });
  const bars: Bar[] = detail.steps
    .filter(step => step.started_at !== null && step.completed_at !== null && (showInternal || !isInternal(step)))
    .map(step => ({
      key: String(step.id),
      label: `${step.id} ${step.name}`,
      kind: step.error.value !== null ? "error" : isInternal(step) ? "internal" : "agent",
      ...place(step.started_at ?? start, step.completed_at ?? start),
    }));
  if (!isFinal(detail.workflow.status)) {
    bars.push({ key: "running", label: "next step running", kind: "running", ...place(lastFinish(detail), now) });
  }
  return bars;
}

export function preview(stored: Stored): string {
  if (!stored.readable) return `stored as ${stored.format}, not readable outside Python`;
  if (stored.value === null) return "";
  return typeof stored.value === "string" ? stored.value : JSON.stringify(stored.value, null, 2);
}

export function stepValue(step: StepRow, history: EventRow[]): Stored {
  if (step.name !== "DBOS.setEvent") return step.error.value !== null ? step.error : step.output;
  return history.find(event => event.function_id === step.id)?.value ?? step.output;
}
