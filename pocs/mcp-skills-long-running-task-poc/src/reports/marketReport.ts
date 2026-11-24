import type { Agent } from "../agents/agent.ts";
import type { Job } from "../tasks/task.ts";
import type { Sleep } from "./sleep.ts";

export const REPORT_STEPS = ["collecting sources", "analyzing trends", "scoring competitors", "drafting report"] as const;

export type MarketReportRequest = { readonly topic: string; readonly depth: "quick" | "deep" };

export type MarketReportDeps = { readonly stepMs: number; readonly sleep: Sleep; readonly agent?: Agent };

export function stepCount(depth: MarketReportRequest["depth"]): number {
  switch (depth) {
    case "quick":
      return 2;
    case "deep":
      return REPORT_STEPS.length;
    default: {
      const unreachable: never = depth;
      return unreachable;
    }
  }
}

export function agentPrompt(topic: string): string {
  return `Write a market research brief about "${topic}" in at most 5 plain text bullets. No preamble.`;
}

export function renderReport(request: MarketReportRequest, steps: readonly string[], brief: string | undefined): string {
  const lines = steps.map((step, i) => `${i + 1}. ${step}: ok`);
  const body = [`# Market report: ${request.topic}`, `depth: ${request.depth}`, "", ...lines];
  return brief ? [...body, "", "## Agent brief", brief].join("\n") : body.join("\n");
}

export function marketReportJob(request: MarketReportRequest, deps: MarketReportDeps): Job {
  const steps = REPORT_STEPS.slice(0, stepCount(request.depth));
  return async ({ signal, report }) => {
    for (const [i, step] of steps.entries()) {
      report(`step ${i + 1}/${steps.length}: ${step}`);
      await deps.sleep(deps.stepMs, signal);
    }
    if (!deps.agent) return { text: renderReport(request, steps, undefined) };
    report("agent writing brief");
    const brief = await deps.agent.run(agentPrompt(request.topic), signal);
    return { text: renderReport(request, steps, brief) };
  };
}
