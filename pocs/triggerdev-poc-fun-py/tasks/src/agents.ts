import { spawn } from "node:child_process";
import { readdirSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join } from "node:path";

export type CommandRunner = (command: string[]) => Promise<string>;

export function commandPath(command: string): string {
  const home = homedir();
  const candidates = [
    ...(command.includes("/") ? [dirname(command)] : []),
    join(home, ".local", "bin"),
    join(home, ".bun", "bin"),
    "/opt/homebrew/bin",
    "/usr/local/bin",
    ...nodeBinaries(home),
    ...(process.env.PATH ?? "").split(delimiter),
  ];
  return [...new Set(candidates.filter(Boolean))].join(delimiter);
}

function nodeBinaries(home: string): string[] {
  const root = join(home, ".nvm", "versions", "node");
  try {
    return readdirSync(root).sort().reverse().map(version => join(root, version, "bin"));
  } catch {
    return [];
  }
}

export const processRunner: CommandRunner = command =>
  new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      env: { ...process.env, PATH: commandPath(command[0]) },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", chunk => (stdout += chunk));
    child.stderr.on("data", chunk => (stderr += chunk));
    child.on("error", error => reject(new Error(`Unable to start ${command[0]}: ${error.message}`)));
    child.on("close", code => {
      if (code === 0) resolve(stdout);
      else reject(new Error(`Command failed with exit code ${code}: ${(stderr || stdout).trim()}`));
    });
  });

export abstract class Agent {
  constructor(protected readonly runner: CommandRunner = processRunner) {}

  call(model: string, prompt: string, args: string[] = []): Promise<string> {
    if (!model.trim()) throw new TypeError("model is required");
    if (!prompt.trim()) throw new TypeError("prompt is required");
    return this.runner(this.command(model, prompt, [...args]));
  }

  protected abstract command(model: string, prompt: string, args: string[]): string[];
}

export class ClaudeCodeAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["claude", "-p", "--model", model, ...args, prompt]; }
}

export class CodexAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["codex", "exec", "--model", model, ...args, prompt]; }
}

export class AgyAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["agy", "-p", "--model", model, ...args, prompt]; }
}

export class OllamaAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["ollama", "run", ...args, model, prompt]; }
}

export const providers = { claude: ClaudeCodeAgent, codex: CodexAgent, agy: AgyAgent, ollama: OllamaAgent } as const;

export type Provider = keyof typeof providers;

export function isProvider(value: string): value is Provider {
  return Object.hasOwn(providers, value);
}

export function agentFor(provider: Provider, runner: CommandRunner = processRunner): Agent {
  return new providers[provider](runner);
}
