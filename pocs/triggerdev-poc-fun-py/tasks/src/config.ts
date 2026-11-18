import { isProvider, type Provider } from "./agents.ts";

export interface AgentConfig {
  provider: Provider;
  model: string;
}

export function readConfig(env: Record<string, string | undefined> = process.env): AgentConfig {
  const provider = env.AGENT_PROVIDER ?? "";
  const model = env.AGENT_MODEL ?? "";
  if (!isProvider(provider)) throw new Error(`AGENT_PROVIDER must be claude, codex, agy or ollama, got "${provider}"`);
  if (!model.trim()) throw new Error("AGENT_MODEL is required");
  return { provider, model };
}
