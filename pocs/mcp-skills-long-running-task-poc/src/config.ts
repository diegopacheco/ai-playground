import { fileURLToPath } from "node:url";

export type Config = {
  readonly serviceName: string;
  readonly skillsDir: string;
  readonly stepMs: number;
  readonly taskTtlMs: number;
  readonly pollIntervalMs: number;
  readonly agentModel: string | undefined;
};

function intFromEnv(name: string, fallback: number): number {
  const raw = Bun.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer, got ${raw}`);
  return value;
}

export function loadConfig(): Config {
  return {
    serviceName: "market-research-mcp",
    skillsDir: fileURLToPath(new URL("../skills", import.meta.url)),
    stepMs: intFromEnv("REPORT_STEP_MS", 1500),
    taskTtlMs: intFromEnv("TASK_TTL_MS", 600000),
    pollIntervalMs: intFromEnv("TASK_POLL_INTERVAL_MS", 500),
    agentModel: Bun.env.REPORT_AGENT_MODEL || undefined,
  };
}
