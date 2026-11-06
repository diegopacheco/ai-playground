export type Transport = "stdio" | "http" | "sse";

export interface Skill {
  readonly name: string;
  readonly source: string;
  readonly description: string | undefined;
}

export interface Agent {
  readonly name: string;
  readonly source: string;
  readonly description: string | undefined;
  readonly model: string | undefined;
  readonly tools: readonly string[];
}

export interface Command {
  readonly name: string;
  readonly source: string;
  readonly description: string | undefined;
}

export interface Mcp {
  readonly name: string;
  readonly transport: Transport;
  readonly description: string | undefined;
  readonly command: string | undefined;
  readonly args: readonly string[];
  readonly env: Readonly<Record<string, string>>;
  readonly url: string | undefined;
  readonly headers: Readonly<Record<string, string>>;
}

export interface Profile {
  readonly name: string;
  readonly description: string | undefined;
  readonly skills: readonly Skill[];
  readonly agents: readonly Agent[];
  readonly commands: readonly Command[];
  readonly mcps: readonly Mcp[];
}

export interface Agentsfile {
  readonly version: string;
  readonly profiles: readonly Profile[];
}
