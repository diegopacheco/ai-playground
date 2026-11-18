import * as restate from "@restatedev/restate-sdk";
import { agentFor } from "./agents.ts";
import { readConfig, type AgentConfig } from "./config.ts";
import { parsePlan } from "./plan.ts";
import { loadPrompt } from "./prompts.ts";
import { completedWork, failed, finished, startRun, stepDone, stepRunning, summarizing, withPlan, type RunState } from "./run-state.ts";

export interface RunRequest {
  goal: string;
}

const STATE = "state";
const RETRY = { maxRetryAttempts: 3, initialRetryInterval: { seconds: 2 } };

function ask(config: AgentConfig, prompt: string): Promise<string> {
  return agentFor(config.provider).call(config.model, prompt).then(output => output.trim());
}

async function runAgent(ctx: restate.WorkflowContext, goal: string): Promise<string> {
  const config = await ctx.run("read config", () => readConfig());
  let state = startRun(goal, config.provider, config.model);
  ctx.set(STATE, state);
  try {
    const tasks = await ctx.run("plan", async () => parsePlan(await ask(config, loadPrompt("plan-goal", { goal }))), RETRY);
    state = withPlan(state, tasks);
    ctx.set(STATE, state);
    for (const [index, task] of tasks.entries()) {
      state = stepRunning(state, index);
      ctx.set(STATE, state);
      const output = await ctx.run(`task ${index + 1}`, () => ask(config, loadPrompt("execute-task", { goal, task, previous_work: completedWork(state) || "none yet" })), RETRY);
      state = stepDone(state, index, output);
      ctx.set(STATE, state);
    }
    state = summarizing(state);
    ctx.set(STATE, state);
    const summary = await ctx.run("summarize", () => ask(config, loadPrompt("summarize-run", { goal, work: completedWork(state) })), RETRY);
    ctx.set(STATE, finished(state, summary));
    return summary;
  } catch (error) {
    if (!(error instanceof restate.TerminalError)) throw error;
    ctx.set(STATE, failed(state, error.message));
    throw error;
  }
}

export const agentWorkflow = restate.workflow({
  name: "AgentWorkflow",
  handlers: {
    run: (ctx: restate.WorkflowContext, request: RunRequest) => {
      if (!request?.goal?.trim()) throw new restate.TerminalError("goal is required", { errorCode: 400 });
      return runAgent(ctx, request.goal.trim());
    },
    status: async (ctx: restate.WorkflowSharedContext): Promise<RunState | null> => (await ctx.get<RunState>(STATE)) ?? null,
  },
});
