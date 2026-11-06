import { AgentsfileError } from "./errors";
import type { Agent, Agentsfile, Command, Mcp, Plugin, Profile, Skill, SkillFile, Transport } from "./model";
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
  blocks: ["skill", "agent", "command", "mcp", "plugin"],
};
const PLUGIN: Rules = {
  ctx: "plugin",
  attributes: { description: "string", version: "string", author: "string", source: "string" },
  blocks: ["skill", "agent", "command", "mcp"],
};
const SKILL: Rules = {
  ctx: "skill",
  attributes: { description: "string", source: "string", content: "string" },
  blocks: ["file"],
};
const SKILL_FILE: Rules = { ctx: "file", attributes: { source: "string", content: "string" }, blocks: [] };
const AGENT: Rules = {
  ctx: "agent",
  attributes: { description: "string", model: "string", tools: "list", source: "string", content: "string" },
  blocks: [],
};
const COMMAND: Rules = {
  ctx: "command",
  attributes: {
    description: "string",
    "argument-hint": "string",
    model: "string",
    tools: "list",
    source: "string",
    content: "string",
  },
  blocks: [],
};
const MCP: Rules = {
  ctx: "mcp",
  attributes: {
    description: "string",
    source: "string",
    transport: "string",
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

interface Items {
  readonly skills: Skill[];
  readonly agents: Agent[];
  readonly commands: Command[];
  readonly mcps: Mcp[];
  readonly plugins: Plugin[];
}

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

function unique(owner: Block): (child: Block) => void {
  const seen = new Set<string>();
  return (child) => {
    const id = `${child.keyword}\u0000${child.label}`;
    if (seen.has(id)) {
      throw new AgentsfileError(
        child.line,
        child.col,
        `duplicate ${child.keyword} "${child.label}" in ${owner.keyword} "${owner.label}"`,
      );
    }
    seen.add(id);
  };
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

function exactlyOne(block: Block, hasSource: boolean, hasInline: boolean, inline: string): void {
  if (!hasSource && !hasInline) {
    throw new AgentsfileError(block.line, block.col, `${block.keyword} "${block.label}" requires "source" or ${inline}`);
  }
  if (hasSource && hasInline) {
    throw new AgentsfileError(
      block.line,
      block.col,
      `${block.keyword} "${block.label}" cannot set both "source" and ${inline}`,
    );
  }
}

function sourceOrContent(block: Block, attributes: Attributes): void {
  exactlyOne(block, attributes.has("source"), attributes.has("content"), `"content"`);
}

function leaf(block: Block, rules: Rules): Attributes {
  return scan(block.body, rules, () => undefined);
}

function toSkillFile(block: Block): SkillFile {
  const attributes = leaf(block, SKILL_FILE);
  sourceOrContent(block, attributes);
  return { path: block.label, source: str(attributes, "source"), content: str(attributes, "content") };
}

function toSkill(block: Block): Skill {
  const files: SkillFile[] = [];
  const check = unique(block);
  const attributes = scan(block.body, SKILL, (child) => {
    check(child);
    files.push(toSkillFile(child));
  });
  sourceOrContent(block, attributes);
  return {
    name: block.label,
    description: str(attributes, "description"),
    source: str(attributes, "source"),
    content: str(attributes, "content"),
    files,
  };
}

function toAgent(block: Block): Agent {
  const attributes = leaf(block, AGENT);
  sourceOrContent(block, attributes);
  return {
    name: block.label,
    description: str(attributes, "description"),
    model: str(attributes, "model"),
    tools: list(attributes, "tools"),
    source: str(attributes, "source"),
    content: str(attributes, "content"),
  };
}

function toCommand(block: Block): Command {
  const attributes = leaf(block, COMMAND);
  sourceOrContent(block, attributes);
  return {
    name: block.label,
    description: str(attributes, "description"),
    argumentHint: str(attributes, "argument-hint"),
    model: str(attributes, "model"),
    tools: list(attributes, "tools"),
    source: str(attributes, "source"),
    content: str(attributes, "content"),
  };
}

function toMcp(block: Block): Mcp {
  const attributes = leaf(block, MCP);
  exactlyOne(block, attributes.has("source"), attributes.has("transport"), `"transport"`);
  const name = str(attributes, "transport");
  const transport = name !== undefined && isTransport(name) ? name : undefined;
  if (transport !== undefined) {
    const needed = transport === "stdio" ? "command" : "url";
    if (!attributes.has(needed)) {
      throw new AgentsfileError(block.line, block.col, `mcp "${block.label}" with ${transport} transport requires "${needed}"`);
    }
  }
  return {
    name: block.label,
    description: str(attributes, "description"),
    source: str(attributes, "source"),
    transport,
    command: str(attributes, "command"),
    args: list(attributes, "args"),
    env: map(attributes, "env"),
    url: str(attributes, "url"),
    headers: map(attributes, "headers"),
  };
}

function container(block: Block, rules: Rules): { attributes: Attributes; items: Items } {
  const items: Items = { skills: [], agents: [], commands: [], mcps: [], plugins: [] };
  const check = unique(block);
  const attributes = scan(block.body, rules, (child) => {
    check(child);
    if (child.keyword === "skill") {
      items.skills.push(toSkill(child));
    } else if (child.keyword === "agent") {
      items.agents.push(toAgent(child));
    } else if (child.keyword === "command") {
      items.commands.push(toCommand(child));
    } else if (child.keyword === "mcp") {
      items.mcps.push(toMcp(child));
    } else {
      items.plugins.push(toPlugin(child));
    }
  });
  return { attributes, items };
}

function toPlugin(block: Block): Plugin {
  const { attributes, items } = container(block, PLUGIN);
  const inline = items.skills.length + items.agents.length + items.commands.length + items.mcps.length > 0;
  exactlyOne(block, attributes.has("source"), inline, "inline items");
  return {
    name: block.label,
    description: str(attributes, "description"),
    version: str(attributes, "version"),
    author: str(attributes, "author"),
    source: str(attributes, "source"),
    skills: items.skills,
    agents: items.agents,
    commands: items.commands,
    mcps: items.mcps,
  };
}

function toProfile(block: Block): Profile {
  const { attributes, items } = container(block, PROFILE);
  return {
    name: block.label,
    description: str(attributes, "description"),
    skills: items.skills,
    agents: items.agents,
    commands: items.commands,
    mcps: items.mcps,
    plugins: items.plugins,
  };
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
