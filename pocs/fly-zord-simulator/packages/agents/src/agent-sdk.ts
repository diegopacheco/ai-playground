import { spawnSync } from "node:child_process";
import { closeSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync, openSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";

export type CommandRunner = (command: string[]) => string;

const UNAVAILABLE = "unavailable";
const NEVER = "never";
const USAGE_URL = "https://api.anthropic.com/api/oauth/usage";
const SESSION_WINDOW_MINUTES = 360;

export class ModelQuota {
  constructor(public readonly label: string, public readonly remaining: string, public readonly resetsIn: string) {}
}

export class Usage {
  constructor(
    public readonly weeklyLimit = UNAVAILABLE,
    public readonly currentLimit = UNAVAILABLE,
    public readonly timeToResetWeek = UNAVAILABLE,
    public readonly timeToResetSession = UNAVAILABLE,
    public readonly models: ModelQuota[] = []
  ) {}
}

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

const processRunner: CommandRunner = command => {
  const result = spawnSync(command[0], command.slice(1), { encoding: "utf8", env: { ...process.env, PATH: commandPath(command[0]) } });
  if (result.error) throw new Error(`Unable to start ${command[0]}: ${result.error.message}`);
  if (result.status !== 0) throw new Error(`Command failed with exit code ${result.status}: ${(result.stderr || result.stdout || "").trim()}`);
  return result.stdout ?? "";
};

export function jsonObject(output: string): Record<string, unknown> {
  const start = output.indexOf("{");
  const end = output.lastIndexOf("}");
  if (start < 0 || end < start) return {};
  try {
    const value: unknown = JSON.parse(output.slice(start, end + 1));
    return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export function formatDuration(ms: number): string {
  if (ms <= 0) return "0m";
  const totalMinutes = Math.floor(ms / 60000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

export function formatPercent(value: number): string {
  return `${Math.round(Math.max(0, Math.min(100, value)))}%`;
}

function durationUntil(moment: unknown): string {
  const reset = typeof moment === "number" ? moment * 1000 : typeof moment === "string" && moment ? Date.parse(moment) : Number.NaN;
  return Number.isNaN(reset) ? UNAVAILABLE : formatDuration(reset - Date.now());
}

function windowUsage(windows: [string, unknown][], usedKey: string, resetKey: string, remainingFrom: (used: number) => number): Usage {
  const values = new Map<string, string>();
  for (const [prefix, value] of windows) {
    if (value === null || typeof value !== "object") continue;
    const window = value as Record<string, unknown>;
    const used = window[usedKey];
    if (typeof used !== "number") continue;
    values.set(`${prefix}Limit`, formatPercent(remainingFrom(used)));
    values.set(`${prefix}Reset`, durationUntil(window[resetKey]));
  }
  return new Usage(
    values.get("weekLimit") ?? UNAVAILABLE,
    values.get("sessionLimit") ?? UNAVAILABLE,
    values.get("weekReset") ?? UNAVAILABLE,
    values.get("sessionReset") ?? UNAVAILABLE
  );
}

export abstract class Agent {
  constructor(protected readonly runner: CommandRunner = processRunner) {}

  call(model: string, prompt: string, args: string[] = []): string {
    if (!model.trim()) throw new TypeError("model is required");
    if (!prompt.trim()) throw new TypeError("prompt is required");
    return this.runner(this.command(model, prompt, [...args]));
  }

  usage(): Usage { return new Usage(); }
  protected abstract command(model: string, prompt: string, args: string[]): string[];
}

export function curlConfig(token: string): string {
  const headers = [`Authorization: Bearer ${token}`, "anthropic-beta: oauth-2025-04-20", "Accept: application/json"];
  return [`url = "${USAGE_URL}"`, ...headers.map(header => `header = "${header}"`)].join("\n") + "\n";
}

export function accessToken(credentials: Record<string, unknown>): string {
  const oauth = credentials.claudeAiOauth;
  if (oauth === null || typeof oauth !== "object") return "";
  const token = (oauth as Record<string, unknown>).accessToken;
  return typeof token === "string" ? token : "";
}

export function parseClaudeUsage(data: Record<string, unknown>): Usage {
  return windowUsage([["session", data.five_hour], ["week", data.seven_day]], "utilization", "resets_at", used => 100 - used);
}

export class ClaudeCodeAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["claude", "-p", "--model", model, ...args, prompt]; }

  usage(): Usage {
    const token = this.token();
    if (!token) return new Usage();
    try {
      return parseClaudeUsage(jsonObject(this.readUsage(token)));
    } catch {
      return new Usage();
    }
  }

  private readUsage(token: string): string {
    const directory = mkdtempSync(join(tmpdir(), "agent-sdk-"));
    const path = join(directory, "usage.curl");
    try {
      closeSync(openSync(path, "w", 0o600));
      writeFileSync(path, curlConfig(token), { mode: 0o600 });
      return this.runner(["curl", "-sS", "-K", path]);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }

  private token(): string {
    if (process.env.CLAUDE_CODE_OAUTH_TOKEN) return process.env.CLAUDE_CODE_OAUTH_TOKEN;
    try {
      const token = accessToken(jsonObject(this.runner(["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"])));
      if (token) return token;
    } catch {
    }
    try {
      const home = process.env.CLAUDE_CONFIG_DIR || join(homedir(), ".claude");
      return accessToken(jsonObject(readFileSync(join(home, ".credentials.json"), "utf8")));
    } catch {
      return "";
    }
  }
}

function sessionFiles(root: string): string[] {
  const found: string[] = [];
  const walk = (directory: string) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith(".jsonl")) found.push(path);
    }
  };
  try {
    walk(root);
  } catch {
    return [];
  }
  return found.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs).slice(0, 20);
}

export function latestRateLimits(home: string): Record<string, unknown> {
  for (const path of sessionFiles(join(home, "sessions"))) {
    let lines: string[];
    try {
      lines = readFileSync(path, "utf8").split("\n");
    } catch {
      continue;
    }
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const payload = jsonObject(lines[index]).payload;
      if (payload === null || typeof payload !== "object") continue;
      const limits = (payload as Record<string, unknown>).rate_limits;
      if (limits === null || typeof limits !== "object") continue;
      const typed = limits as Record<string, unknown>;
      if (typeof typed.primary === "object" && typed.primary !== null) return typed;
      if (typeof typed.secondary === "object" && typed.secondary !== null) return typed;
    }
  }
  return {};
}

export function parseCodexUsage(limits: Record<string, unknown>): Usage {
  const windows: [string, unknown][] = [];
  for (const value of [limits.primary, limits.secondary]) {
    if (value === null || typeof value !== "object") continue;
    const minutes = (value as Record<string, unknown>).window_minutes;
    windows.push([typeof minutes === "number" && minutes <= SESSION_WINDOW_MINUTES ? "session" : "week", value]);
  }
  return windowUsage(windows, "used_percent", "resets_at", used => 100 - used);
}

export class CodexAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["codex", "exec", "--model", model, ...args, prompt]; }

  usage(): Usage {
    const home = process.env.CODEX_HOME || join(homedir(), ".codex");
    return existsSync(home) ? parseCodexUsage(latestRateLimits(home)) : new Usage();
  }
}

export function parseAgyUsage(data: Record<string, unknown>): Usage {
  const models = data.models;
  if (!Array.isArray(models)) return new Usage();
  const quotas = new Map<string, { remaining: number; resetMs: number }>();
  for (const value of models) {
    if (value === null || typeof value !== "object") continue;
    const model = value as Record<string, unknown>;
    if (model.isAutocompleteOnly === true) continue;
    const remaining = model.remainingPercentage;
    const label = typeof model.label === "string" && model.label ? model.label : typeof model.modelId === "string" ? model.modelId : "";
    if (typeof remaining !== "number" || !label) continue;
    const resetMs = typeof model.timeUntilResetMs === "number" ? model.timeUntilResetMs : 0;
    const current = quotas.get(label);
    if (!current || remaining < current.remaining) quotas.set(label, { remaining, resetMs });
  }
  if (quotas.size === 0) return new Usage();
  const quotaModels = [...quotas].map(([label, quota]) => new ModelQuota(label, formatPercent(quota.remaining * 100), formatDuration(quota.resetMs)));
  const lowest = [...quotas.values()].reduce((a, b) => (a.remaining <= b.remaining ? a : b));
  return new Usage(UNAVAILABLE, formatPercent(lowest.remaining * 100), UNAVAILABLE, formatDuration(lowest.resetMs), quotaModels);
}

export class AgyAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["agy", "-p", "--model", model, ...args, prompt]; }

  usage(): Usage {
    try {
      return parseAgyUsage(jsonObject(this.runner(["antigravity-usage", "quota", "--json"])));
    } catch {
      return new Usage();
    }
  }
}

export class OllamaAgent extends Agent {
  protected command(model: string, prompt: string, args: string[]): string[] { return ["ollama", "run", ...args, model, prompt]; }

  usage(): Usage { return new Usage("100%", "100%", NEVER, NEVER); }
}
