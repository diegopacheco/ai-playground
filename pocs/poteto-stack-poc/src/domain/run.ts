import type { Playbook } from "./types";

export type Step =
  | { text: string; state: "todo" }
  | { text: string; state: "done"; evidence: string }
  | { text: string; state: "skipped"; reason: string };

export type Run = { playbook: string; steps: Step[]; principles: string[] };

export function startRun(playbook: Playbook): Run {
  return { playbook: playbook.title, steps: playbook.steps.map((text) => ({ text, state: "todo" })), principles: [] };
}

export function complete(run: Run, index: number, evidence: string): Run {
  return update(run, index, (s) => (evidence.trim() ? { text: s.text, state: "done", evidence: evidence.trim() } : { text: s.text, state: "todo" }));
}

export function skip(run: Run, index: number, reason: string): Run {
  return update(run, index, (s) => (reason.trim() ? { text: s.text, state: "skipped", reason: reason.trim() } : { text: s.text, state: "todo" }));
}

export function reopen(run: Run, index: number): Run {
  return update(run, index, (s) => ({ text: s.text, state: "todo" }));
}

export function togglePrinciple(run: Run, name: string): Run {
  const principles = run.principles.includes(name) ? run.principles.filter((p) => p !== name) : [...run.principles, name];
  return { ...run, principles };
}

export type Gate = { ok: true } | { ok: false; missing: string[] };

export function gate(run: Run): Gate {
  const missing = run.steps.filter((s) => s.state === "todo").map((s) => `no evidence: ${s.text}`);
  if (run.principles.length === 0) missing.push("no principle named");
  return missing.length === 0 ? { ok: true } : { ok: false, missing };
}

export function todoLine(step: Step): string {
  if (step.state === "done") return `[x] ${step.text} | evidence: ${step.evidence}`;
  if (step.state === "skipped") return `skip: ${step.text} | reason: ${step.reason}`;
  return `[ ] ${step.text}`;
}

export function report(run: Run): string {
  const result = gate(run);
  if (!result.ok) throw new Error(`cannot report done: ${result.missing.join("; ")}`);
  return [`playbook: ${run.playbook}`, ...run.steps.map(todoLine), `principles: ${run.principles.join(", ")}`].join("\n");
}

function update(run: Run, index: number, change: (s: Step) => Step): Run {
  return { ...run, steps: run.steps.map((s, i) => (i === index ? change(s) : s)) };
}
