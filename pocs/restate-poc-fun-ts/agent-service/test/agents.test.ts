import { describe, expect, test } from "bun:test";
import { agentFor, type CommandRunner } from "../src/agents.ts";
import { readConfig } from "../src/config.ts";

function recorder(): { runner: CommandRunner; calls: string[][] } {
  const calls: string[][] = [];
  return { calls, runner: async command => (calls.push(command), "ok") };
}

describe("agent providers", () => {
  test("each provider shells out to its own CLI with the model and prompt", async () => {
    const expected = {
      claude: ["claude", "-p", "--model", "m", "hi"],
      codex: ["codex", "exec", "--model", "m", "hi"],
      agy: ["agy", "-p", "--model", "m", "hi"],
      ollama: ["ollama", "run", "m", "hi"],
    } as const;
    for (const [provider, command] of Object.entries(expected)) {
      const { runner, calls } = recorder();
      await agentFor(provider as keyof typeof expected, runner).call("m", "hi");
      expect(calls).toEqual([[...command]]);
    }
  });

  test("an empty model or prompt is rejected before any CLI runs", () => {
    const { runner, calls } = recorder();
    expect(() => agentFor("claude", runner).call(" ", "hi")).toThrow("model");
    expect(() => agentFor("claude", runner).call("m", "")).toThrow("prompt");
    expect(calls).toHaveLength(0);
  });

  test("config refuses an unknown provider instead of failing later inside the workflow", () => {
    expect(() => readConfig({ AGENT_PROVIDER: "gpt", AGENT_MODEL: "m" })).toThrow("AGENT_PROVIDER");
    expect(() => readConfig({ AGENT_PROVIDER: "ollama", AGENT_MODEL: "" })).toThrow("AGENT_MODEL");
    expect(readConfig({ AGENT_PROVIDER: "ollama", AGENT_MODEL: "llama3.2" })).toEqual({ provider: "ollama", model: "llama3.2" });
  });
});
