import { AbortTaskRunError, idempotencyKeys, metadata, task } from "@trigger.dev/sdk";
import { agentFor } from "../agents.ts";
import { readConfig, type AgentConfig } from "../config.ts";
import { parsePlan } from "../plan.ts";
import { loadPrompt } from "../prompts.ts";
import { completedWork, failed, finished, startRun, stepDone, stepRunning, summarizing, toJson, withPlan, type RunState } from "../run-state.ts";

export interface RunRequest {
  goal: string;
}

interface AgentPayload {
  config: AgentConfig;
  goal: string;
}

const RETRY = { maxAttempts: 3, minTimeoutInMs: 2000, maxTimeoutInMs: 8000 };

function ask(config: AgentConfig, prompt: string): Promise<string> {
  return agentFor(config.provider).call(config.model, prompt).then(output => output.trim());
}

export const planGoal = task({
  id: "plan-goal",
  retry: RETRY,
  run: async ({ config, goal }: AgentPayload) => parsePlan(await ask(config, loadPrompt("plan-goal", { goal }))),
});

export const executeTask = task({
  id: "execute-task",
  retry: RETRY,
  run: ({ config, goal, task, previousWork }: AgentPayload & { task: string; previousWork: string }) =>
    ask(config, loadPrompt("execute-task", { goal, task, previous_work: previousWork })),
});

export const summarizeRun = task({
  id: "summarize-run",
  retry: RETRY,
  run: ({ config, goal, work }: AgentPayload & { work: string }) => ask(config, loadPrompt("summarize-run", { goal, work })),
});

function publish(state: RunState): RunState {
  metadata.set("state", toJson(state));
  return state;
}

export const agentWorkflow = task({
  id: "agent-workflow",
  retry: { maxAttempts: 3 },
  run: async (request: RunRequest, { ctx }) => {
    const goal = request?.goal?.trim();
    if (!goal) throw new AbortTaskRunError("goal is required");
    const config = readConfig();
    const tags = [`wf_${ctx.run.id}`];
    let state = publish(startRun(goal, config.provider, config.model));
    try {
      const tasks = await planGoal.triggerAndWait({ config, goal }, { idempotencyKey: await idempotencyKeys.create("plan"), tags }).unwrap();
      state = publish(withPlan(state, tasks));
      for (const [index, item] of tasks.entries()) {
        state = publish(stepRunning(state, index));
        const key = await idempotencyKeys.create(`task-${index + 1}`);
        const output = await executeTask.triggerAndWait({ config, goal, task: item, previousWork: completedWork(state) || "none yet" }, { idempotencyKey: key, tags }).unwrap();
        state = publish(stepDone(state, index, output));
      }
      state = publish(summarizing(state));
      const summary = await summarizeRun.triggerAndWait({ config, goal, work: completedWork(state) }, { idempotencyKey: await idempotencyKeys.create("summarize"), tags }).unwrap();
      publish(finished(state, summary));
      await metadata.flush();
      return summary;
    } catch (error) {
      publish(failed(state, error instanceof Error ? error.message : String(error)));
      await metadata.flush();
      throw error;
    }
  },
});
