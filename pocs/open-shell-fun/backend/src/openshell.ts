import { CliError, type CliRunner } from "./cli";
import { parsePolicyEvents } from "./logs";
import { evaluate, SCENARIOS } from "./attacks";
import type { ExecResult, NetworkRule, PolicyEvent, Sandbox, ScenarioResult } from "./types";

type SandboxJson = {
  name: string;
  phase: string;
  created_at: string;
  current_policy_version: number | null;
};

export function endpointSpec(rule: NetworkRule): string {
  return `${rule.host}:${rule.port}:${rule.access}:rest:${rule.enforcement}`;
}

export class OpenShell {
  constructor(private readonly run: CliRunner, private readonly image: string) {}

  private async ok(args: string[]): Promise<string> {
    const result = await this.run(args);
    if (result.code !== 0) {
      throw new CliError(args, result);
    }
    return result.stdout;
  }

  async status(): Promise<string> {
    return this.ok(["status"]);
  }

  async listSandboxes(): Promise<Sandbox[]> {
    const out = await this.ok(["sandbox", "list", "-o", "json"]);
    const parsed = JSON.parse(out) as { sandboxes: SandboxJson[] };
    return parsed.sandboxes.map((s) => ({
      name: s.name,
      phase: s.phase,
      createdAt: s.created_at,
      policyVersion: s.current_policy_version ?? 0,
    }));
  }

  async createSandbox(name: string): Promise<void> {
    await this.ok(["sandbox", "create", "--name", name, "--from", this.image, "--no-auto-providers", "--detach", "--no-tty"]);
  }

  async deleteSandbox(name: string): Promise<void> {
    await this.ok(["sandbox", "delete", name]);
  }

  async exec(name: string, command: string): Promise<ExecResult> {
    const result = await this.run(["sandbox", "exec", name, "--no-tty", "--timeout", "30", "--", "sh", "-c", command]);
    return { exitCode: result.code, stdout: result.stdout, stderr: result.stderr };
  }

  async policy(name: string): Promise<string> {
    const out = await this.ok(["policy", "get", name, "--full"]);
    const start = out.indexOf("---");
    return start === -1 ? out : out.slice(start + 4);
  }

  async addRule(name: string, rule: NetworkRule): Promise<string> {
    return this.ok([
      "policy", "update", name,
      "--rule-name", rule.ruleName,
      "--binary", rule.binary,
      "--add-endpoint", endpointSpec(rule),
      "--wait",
    ]);
  }

  async removeRule(name: string, ruleName: string): Promise<string> {
    return this.ok(["policy", "update", name, "--remove-rule", ruleName, "--wait"]);
  }

  async attack(name: string): Promise<ScenarioResult[]> {
    return Promise.all(SCENARIOS.map(async (scenario) => evaluate(scenario, await this.exec(name, scenario.command))));
  }

  async events(name: string): Promise<PolicyEvent[]> {
    const out = await this.ok(["logs", name, "-n", "300", "--source", "sandbox"]);
    return parsePolicyEvents(out).reverse();
  }
}
