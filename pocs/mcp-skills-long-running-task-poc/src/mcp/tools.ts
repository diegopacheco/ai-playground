import type { McpServer } from "@modelcontextprotocol/server";
import * as z from "zod";
import { marketReportJob, type MarketReportDeps } from "../reports/marketReport.ts";
import type { TaskStore } from "../tasks/store.ts";
import { clientAcceptsTasks, createTaskResult, type TaskTiming } from "./tasksExtension.ts";

export type ToolDeps = {
  readonly store: TaskStore;
  readonly timing: TaskTiming;
  readonly report: MarketReportDeps;
};

const MarketReportInput = {
  topic: z.string().min(2).max(120).describe("Market or product to research"),
  depth: z.enum(["quick", "deep"]).default("deep").describe("quick runs 2 steps, deep runs 4 steps"),
};

export function registerTools(mcp: McpServer, deps: ToolDeps): void {
  mcp.registerTool(
    "market_report",
    {
      title: "Market report",
      description: "Builds a market research report. Long running: returns an MCP task handle when the client supports the Tasks extension.",
      inputSchema: MarketReportInput,
    },
    async (input, ctx) => {
      const job = marketReportJob(input, deps.report);
      if (clientAcceptsTasks(ctx)) return { content: [], ...createTaskResult(deps.store.start(job), deps.timing) };
      const output = await job({ signal: ctx.mcpReq.signal, report: () => {} });
      return { content: [{ type: "text", text: output.text }] };
    },
  );
}
