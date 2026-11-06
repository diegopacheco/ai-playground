import { AgentsfileError } from "./errors";
import type { Agent, Agentsfile, Command, Mcp, Profile, Skill, Transport } from "./model";
import type { Attribute, Block, Statement, Value } from "./parser";

type Kind = Value["kind"];

interface Rules {
  readonly ctx: string;
  readonly attributes: Readonly<Record<string, Kind>>;
  readonly blocks: readonly string[];
}

const FILE: Rules = { ctx: "file", attributes: { version: "string" }, blocks: ["profile"] };
const PROFILE: Rules = {
  ctx: "profile",
  attributes: { description: "string" },
  blocks: ["skill", "agent", "command", "mcp"],
};
const SKILL: Rules = { ctx: "skill", attributes: { source: "string", description: "string" }, blocks: [] };
const COMMAND: Rules = { ctx: "command", attributes: { source: "string", description: "string" }, blocks: [] };
const AGENT: Rules = {
  ctx: "agent",
  attributes: { source: "string", description: "string", model: "string", tools: "list" },
  blocks: [],
};
const MCP: Rules = {
  ctx: "mcp",
  attributes: {
    transport: "string",
    description: "string",
    command: "string",
    args: "list",
    env: "map",
    url: "string",
    headers: "map",
  },
  blocks: [],
};

const ARTICLES: Readonly<Record<Kind, string>> = { string: "a string", list: "a list", map: "a map" };
const TRANSPORTS: readonly Transport[] = ["stdio", "http", "sse"];

type Attributes = ReadonlyMap<string, Attribute>;

const isTransport = (value: string): value is Transport => (TRANSPORTS as readonly string[]).includes(value);

function checkValue(rules: Rules, attr: Attribute): void {
  if (attr.value.kind !== "string") {
    return;
  }
  if (rules === FILE && attr.name === "version" && attr.value.value !== "1") {
    throw new AgentsfileError(attr.value.line, attr.value.col, `unsupported version "${attr.value.value}"`);
  }
  if (rules === MCP && attr.name === "transport" && !isTransport(attr.value.value)) {
    throw new AgentsfileError(
      attr.value.line,
      attr.value.col,
      `invalid transport "${attr.value.value}", expected stdio, http or sse`,
    );
  }
}

function scan(statements: readonly Statement[], rules: Rules, onBlock: (block: Block) => void): Attributes {
  const attributes = new Map<string, Attribute>();
  for (const statement of statements) {
    if (statement.type === "block") {
      if (!rules.blocks.includes(statement.keyword)) {
        throw new AgentsfileError(statement.line, statement.col, `unknown block "${statement.keyword}" in ${rules.ctx}`);
      }
      onBlock(statement);
      continue;
    }
    const kind = rules.attributes[statement.name];
    if (kind === undefined) {
      throw new AgentsfileError(statement.line, statement.col, `unknown attribute "${statement.name}" in ${rules.ctx}`);
    }
    if (attributes.has(statement.name)) {
      throw new AgentsfileError(statement.line, statement.col, `duplicate attribute "${statement.name}" in ${rules.ctx}`);
    }
    if (statement.value.kind !== kind) {
      throw new AgentsfileError(statement.line, statement.col, `attribute "${statement.name}" must be ${ARTICLES[kind]}`);
    }
    checkValue(rules, statement);
    attributes.set(statement.name, statement);
  }
  return attributes;
}

function str(attributes: Attributes, name: string): string | undefined {
  const value = attributes.get(name)?.value;
  return value?.kind === "string" ? value.value : undefined;
}

function list(attributes: Attributes, name: string): readonly string[] {
  const value = attributes.get(name)?.value;
  return value?.kind === "list" ? value.items.map((item) => item.value) : [];
}

function map(attributes: Attributes, name: string): Readonly<Record<string, string>> {
  const value = attributes.get(name)?.value;
  return value?.kind === "map" ? Object.fromEntries(value.entries.map((entry) => [entry.key, entry.value.value])) : {};
}

function required(attributes: Attributes, name: string, block: Block): string {
  const value = str(attributes, name);
  if (value === undefined) {
    throw new AgentsfileError(block.line, block.col, `missing required attribute "${name}" in ${block.keyword} "${block.label}"`);
  }
  return value;
}

function leaf(block: Block, rules: Rules): Attributes {
  return scan(block.body, rules, () => undefined);
}

function toSkill(block: Block): Skill {
  const attributes = leaf(block, SKILL);
  return {
    name: block.label,
    source: required(attributes, "source", block),
    description: str(attributes, "description"),
  };
}

function toCommand(block: Block): Command {
  const attributes = leaf(block, COMMAND);
  return {
    name: block.label,
    source: required(attributes, "source", block),
    description: str(attributes, "description"),
  };
}

function toAgent(block: Block): Agent {
  const attributes = leaf(block, AGENT);
  return {
    name: block.label,
    source: required(attributes, "source", block),
    description: str(attributes, "description"),
    model: str(attributes, "model"),
    tools: list(attributes, "tools"),
  };
}

function toMcp(block: Block): Mcp {
  const attributes = leaf(block, MCP);
  const transport = required(attributes, "transport", block);
  if (!isTransport(transport)) {
    throw new AgentsfileError(block.line, block.col, `invalid transport "${transport}", expected stdio, http or sse`);
  }
  const needed = transport === "stdio" ? "command" : "url";
  if (str(attributes, needed) === undefined) {
    throw new AgentsfileError(block.line, block.col, `mcp "${block.label}" with ${transport} transport requires "${needed}"`);
  }
  return {
    name: block.label,
    transport,
    description: str(attributes, "description"),
    command: str(attributes, "command"),
    args: list(attributes, "args"),
    env: map(attributes, "env"),
    url: str(attributes, "url"),
    headers: map(attributes, "headers"),
  };
}

function toProfile(block: Block): Profile {
  const skills: Skill[] = [];
  const agents: Agent[] = [];
  const commands: Command[] = [];
  const mcps: Mcp[] = [];
  const seen = new Set<string>();
  const attributes = scan(block.body, PROFILE, (child) => {
    const id = `${child.keyword}\u0000${child.label}`;
    if (seen.has(id)) {
      throw new AgentsfileError(
        child.line,
        child.col,
        `duplicate ${child.keyword} "${child.label}" in profile "${block.label}"`,
      );
    }
    seen.add(id);
    if (child.keyword === "skill") {
      skills.push(toSkill(child));
    } else if (child.keyword === "agent") {
      agents.push(toAgent(child));
    } else if (child.keyword === "command") {
      commands.push(toCommand(child));
    } else {
      mcps.push(toMcp(child));
    }
  });
  return { name: block.label, description: str(attributes, "description"), skills, agents, commands, mcps };
}

export function validate(statements: readonly Statement[]): Agentsfile {
  const profiles: Profile[] = [];
  const names = new Set<string>();
  const attributes = scan(statements, FILE, (block) => {
    if (names.has(block.label)) {
      throw new AgentsfileError(block.line, block.col, `duplicate profile "${block.label}"`);
    }
    names.add(block.label);
    profiles.push(toProfile(block));
  });
  const version = str(attributes, "version");
  if (version === undefined) {
    throw new AgentsfileError(1, 1, `missing required attribute "version"`);
  }
  return { version, profiles };
}
