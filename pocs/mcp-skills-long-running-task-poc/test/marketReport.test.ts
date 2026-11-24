import { describe, expect, test } from "bun:test";
import type { Agent } from "../src/agents/agent.ts";
import { agentPrompt, marketReportJob, REPORT_STEPS } from "../src/reports/marketReport.ts";
import { abortableSleep, type Sleep } from "../src/reports/sleep.ts";

const instant: Sleep = async () => {};
const ctx = (messages: string[] = [], signal = new AbortController().signal) => ({ signal, report: (m: string) => messages.push(m) });

describe("marketReportJob", () => {
  test("quick depth runs only the first 2 steps so it finishes faster than deep", async () => {
    const messages: string[] = [];
    const out = await marketReportJob({ topic: "bikes", depth: "quick" }, { stepMs: 0, sleep: instant })(ctx(messages));
    expect(messages).toEqual(["step 1/2: collecting sources", "step 2/2: analyzing trends"]);
    expect(out.text).not.toContain("drafting report");
  });

  test("deep depth runs every step and names the topic in the report", async () => {
    const out = await marketReportJob({ topic: "bikes", depth: "deep" }, { stepMs: 0, sleep: instant })(ctx());
    expect(out.text).toContain("# Market report: bikes");
    for (const step of REPORT_STEPS) expect(out.text).toContain(step);
  });

  test("the agent brief is appended when an agent is configured, and the agent gets the topic", async () => {
    const prompts: string[] = [];
    const agent: Agent = { run: async (prompt) => (prompts.push(prompt), "- demand is growing") };
    const out = await marketReportJob({ topic: "bikes", depth: "quick" }, { stepMs: 0, sleep: instant, agent })(ctx());
    expect(prompts).toEqual([agentPrompt("bikes")]);
    expect(out.text).toContain("## Agent brief\n- demand is growing");
  });

  test("an agent failure fails the job so the task reports failed instead of a partial report", async () => {
    const agent: Agent = { run: async () => { throw new Error("claude exited with 1"); } };
    expect(marketReportJob({ topic: "bikes", depth: "quick" }, { stepMs: 0, sleep: instant, agent })(ctx())).rejects.toThrow("claude exited with 1");
  });

  test("aborting stops the job mid step instead of waiting for the step to finish", async () => {
    const controller = new AbortController();
    const started = Date.now();
    const run = marketReportJob({ topic: "bikes", depth: "deep" }, { stepMs: 10000, sleep: abortableSleep })(ctx([], controller.signal));
    controller.abort();
    expect(run).rejects.toThrow("aborted");
    await run.catch(() => {});
    expect(Date.now() - started).toBeLessThan(1000);
  });
});
