export type Transport = "stdio" | "http" | "sse";

export interface SkillFile {
  readonly path: string;
  readonly source: string | undefined;
  readonly content: string | undefined;
}

export interface Skill {
  readonly name: string;
  readonly description: string | undefined;
  readonly source: string | undefined;
  readonly content: string | undefined;
  readonly files: readonly SkillFile[];
}

export interface Agent {
  readonly name: string;
  readonly description: string | undefined;
  readonly model: string | undefined;
  readonly tools: readonly string[];
  readonly source: string | undefined;
  readonly content: string | undefined;
}

export interface Command {
  readonly name: string;
  readonly description: string | undefined;
  readonly argumentHint: string | undefined;
  readonly model: string | undefined;
  readonly tools: readonly string[];
  readonly source: string | undefined;
  readonly content: string | undefined;
}

export interface Mcp {
  readonly name: string;
  readonly description: string | undefined;
  readonly source: string | undefined;
  readonly transport: Transport | undefined;
  readonly command: string | undefined;
  readonly args: readonly string[];
  readonly env: Readonly<Record<string, string>>;
  readonly url: string | undefined;
  readonly headers: Readonly<Record<string, string>>;
}

export interface Plugin {
  readonly name: string;
  readonly description: string | undefined;
  readonly version: string | undefined;
  readonly author: string | undefined;
  readonly source: string | undefined;
  readonly skills: readonly Skill[];
  readonly agents: readonly Agent[];
  readonly commands: readonly Command[];
  readonly mcps: readonly Mcp[];
}

export interface Profile {
  readonly name: string;
  readonly description: string | undefined;
  readonly skills: readonly Skill[];
  readonly agents: readonly Agent[];
  readonly commands: readonly Command[];
  readonly mcps: readonly Mcp[];
  readonly plugins: readonly Plugin[];
}

export interface Agentsfile {
  readonly version: string;
  readonly profiles: readonly Profile[];
}
