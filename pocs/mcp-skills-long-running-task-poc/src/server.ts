import { McpServer } from "@modelcontextprotocol/server";
import type { SkillCatalog } from "@modelcontextprotocol/server/ext/skills";
import { loadSkills } from "@modelcontextprotocol/server/ext/skills/node";
import { claudeCliAgent } from "./agents/agent.ts";
import { randomIds, systemClock } from "./clock.ts";
import type { Config } from "./config.ts";
import type { Logger } from "./logger.ts";
import { attachTasks } from "./mcp/tasksExtension.ts";
import { registerTools } from "./mcp/tools.ts";
import { abortableSleep } from "./reports/sleep.ts";
import { TaskStore } from "./tasks/store.ts";

export type ServerParts = { readonly skills: SkillCatalog; readonly store: TaskStore };

export async function loadParts(config: Config, logger: Logger): Promise<ServerParts> {
  const skills = await loadSkills(config.skillsDir, { strict: true });
  logger.info("skills loaded", { count: skills.entries.length });
  const store = new TaskStore({ clock: systemClock, ids: randomIds, logger, ttlMs: config.taskTtlMs });
  return { skills, store };
}

export function buildServer(config: Config, parts: ServerParts): McpServer {
  const mcp = new McpServer({ name: config.serviceName, version: "1.0.0" });
  parts.skills.attach(mcp);
  const timing = { ttlMs: config.taskTtlMs, pollIntervalMs: config.pollIntervalMs };
  attachTasks(mcp, parts.store, timing);
  const agent = config.agentModel ? claudeCliAgent(config.agentModel) : undefined;
  registerTools(mcp, { store: parts.store, timing, report: { stepMs: config.stepMs, sleep: abortableSleep, agent } });
  return mcp;
}
