import { FactoryfileError } from "./errors";
import type { Agent, Factory, Factoryfile, Loop, Stage } from "./model";
import type { Attribute, Block, Statement } from "./parser";

type Kind = "string" | "list";

interface Rules {
  readonly ctx: string;
  readonly attributes: Readonly<Record<string, Kind>>;
  readonly blocks: readonly string[];
}

interface Reference {
  readonly target: string;
  readonly names: readonly string[];
  readonly owner: string;
  readonly line: number;
  readonly col: number;
}

const FILE: Rules = { ctx: "file", attributes: { version: "string" }, blocks: ["factory"] };
const FACTORY: Rules = {
  ctx: "factory",
  attributes: { description: "string", entry: "string" },
  blocks: ["agent", "stage", "loop"],
};
const AGENT: Rules = {
  ctx: "agent",
  attributes: {
    description: "string",
    role: "string",
    model: "string",
    tools: "list",
    source: "string",
    content: "string",
  },
  blocks: [],
};
const STAGE: Rules = {
  ctx: "stage",
  attributes: { description: "string", agents: "list", next: "list", artifacts: "list", content: "string" },
  blocks: [],
};
const LOOP: Rules = {
  ctx: "loop",
  attributes: { description: "string", from: "string", to: "string", until: "string", max: "string" },
  blocks: [],
};

const REFERENCES: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  factory: { entry: "stage" },
  stage: { agents: "agent", next: "stage" },
  loop: { from: "stage", to: "stage" },
};

const ARTICLES: Readonly<Record<Kind, string>> = { string: "a string", list: "a list" };
const MAX = /^[1-9][0-9]{0,5}$/;

type Attributes = ReadonlyMap<string, Attribute>;

const key = (kind: string, name: string): string => `${kind}\u0000${name}`;

function optional<K extends string, V>(name: K, value: V | undefined): { [P in K]?: V } {
  return (value === undefined ? {} : { [name]: value }) as { [P in K]?: V };
}

function checkValue(rules: Rules, attr: Attribute): void {
  if (attr.value.kind !== "string") {
    return;
  }
  const { value, line, col } = attr.value;
  if (rules === FILE && attr.name === "version" && value !== "1") {
    throw new FactoryfileError(line, col, `unsupported version "${value}"`);
  }
  if (rules === LOOP && attr.name === "max" && !MAX.test(value)) {
    throw new FactoryfileError(line, col, `invalid max "${value}", expected an integer from 1 to 999999`);
  }
}

function collectReference(rules: Rules, label: string, attr: Attribute, references: Reference[]): void {
  const target = REFERENCES[rules.ctx]?.[attr.name];
  if (target === undefined || attr.value.kind === "map") {
    return;
  }
  const names = attr.value.kind === "list" ? attr.value.items.map((item) => item.value) : [attr.value.value];
  references.push({ target, names, owner: `${rules.ctx} "${label}"`, line: attr.value.line, col: attr.value.col });
}

function scan(
  label: string,
  statements: readonly Statement[],
  rules: Rules,
  references: Reference[],
  onBlock: (block: Block) => void,
): Attributes {
  const attributes = new Map<string, Attribute>();
  for (const statement of statements) {
    if (statement.type === "block") {
      if (!rules.blocks.includes(statement.keyword)) {
        throw new FactoryfileError(statement.line, statement.col, `unknown block "${statement.keyword}" in ${rules.ctx}`);
      }
      onBlock(statement);
      continue;
    }
    const kind = rules.attributes[statement.name];
    if (kind === undefined) {
      throw new FactoryfileError(statement.line, statement.col, `unknown attribute "${statement.name}" in ${rules.ctx}`);
    }
    if (attributes.has(statement.name)) {
      throw new FactoryfileError(statement.line, statement.col, `duplicate attribute "${statement.name}" in ${rules.ctx}`);
    }
    if (statement.value.kind !== kind) {
      throw new FactoryfileError(statement.line, statement.col, `attribute "${statement.name}" must be ${ARTICLES[kind]}`);
    }
    checkValue(rules, statement);
    collectReference(rules, label, statement, references);
    attributes.set(statement.name, statement);
  }
  return attributes;
}

function leaf(block: Block, rules: Rules, references: Reference[]): Attributes {
  return scan(block.label, block.body, rules, references, () => undefined);
}

function str(attributes: Attributes, name: string): string | undefined {
  const value = attributes.get(name)?.value;
  return value?.kind === "string" ? value.value : undefined;
}

function list(attributes: Attributes, name: string): readonly string[] {
  const value = attributes.get(name)?.value;
  return value?.kind === "list" ? value.items.map((item) => item.value) : [];
}

function requires(block: Block, attributes: Attributes, name: string, what: string): void {
  if (!attributes.has(name)) {
    throw new FactoryfileError(block.line, block.col, `${block.keyword} "${block.label}" requires ${what}`);
  }
}

function toAgent(block: Block, references: Reference[]): Agent {
  const attributes = leaf(block, AGENT, references);
  const hasSource = attributes.has("source");
  const hasContent = attributes.has("content");
  if (!hasSource && !hasContent) {
    throw new FactoryfileError(block.line, block.col, `agent "${block.label}" requires "source" or "content"`);
  }
  if (hasSource && hasContent) {
    throw new FactoryfileError(block.line, block.col, `agent "${block.label}" cannot set both "source" and "content"`);
  }
  return {
    name: block.label,
    ...optional("description", str(attributes, "description")),
    ...optional("role", str(attributes, "role")),
    ...optional("model", str(attributes, "model")),
    tools: list(attributes, "tools"),
    ...optional("source", str(attributes, "source")),
    ...optional("content", str(attributes, "content")),
  };
}

function toStage(block: Block, references: Reference[]): Stage {
  const attributes = leaf(block, STAGE, references);
  requires(block, attributes, "agents", `"agents"`);
  return {
    name: block.label,
    ...optional("description", str(attributes, "description")),
    agents: list(attributes, "agents"),
    next: list(attributes, "next"),
    artifacts: list(attributes, "artifacts"),
    ...optional("content", str(attributes, "content")),
  };
}

function toLoop(block: Block, references: Reference[]): Loop {
  const attributes = leaf(block, LOOP, references);
  requires(block, attributes, "from", `"from"`);
  requires(block, attributes, "to", `"to"`);
  const max = str(attributes, "max");
  return {
    name: block.label,
    ...optional("description", str(attributes, "description")),
    from: str(attributes, "from") ?? "",
    to: str(attributes, "to") ?? "",
    ...optional("until", str(attributes, "until")),
    ...optional("max", max === undefined ? undefined : Number(max)),
  };
}

function checkReferences(references: readonly Reference[], defined: ReadonlySet<string>): void {
  for (const reference of references) {
    const unknown = reference.names.find((name) => !defined.has(key(reference.target, name)));
    if (unknown !== undefined) {
      throw new FactoryfileError(
        reference.line,
        reference.col,
        `unknown ${reference.target} "${unknown}" in ${reference.owner}`,
      );
    }
  }
}

function toFactory(block: Block): Factory {
  const references: Reference[] = [];
  const defined = new Set<string>();
  const agents: Agent[] = [];
  const stages: Stage[] = [];
  const loops: Loop[] = [];
  const attributes = scan(block.label, block.body, FACTORY, references, (child) => {
    const id = key(child.keyword, child.label);
    if (defined.has(id)) {
      throw new FactoryfileError(
        child.line,
        child.col,
        `duplicate ${child.keyword} "${child.label}" in factory "${block.label}"`,
      );
    }
    defined.add(id);
    if (child.keyword === "agent") {
      agents.push(toAgent(child, references));
    } else if (child.keyword === "stage") {
      stages.push(toStage(child, references));
    } else {
      loops.push(toLoop(child, references));
    }
  });
  checkReferences(references, defined);
  return {
    name: block.label,
    ...optional("description", str(attributes, "description")),
    ...optional("entry", str(attributes, "entry") ?? stages[0]?.name),
    agents,
    stages,
    loops,
  };
}

export function validate(statements: readonly Statement[]): Factoryfile {
  const factories: Factory[] = [];
  const names = new Set<string>();
  const attributes = scan("", statements, FILE, [], (block) => {
    if (names.has(block.label)) {
      throw new FactoryfileError(block.line, block.col, `duplicate factory "${block.label}"`);
    }
    names.add(block.label);
    factories.push(toFactory(block));
  });
  const version = str(attributes, "version");
  if (version === undefined) {
    throw new FactoryfileError(1, 1, `missing required attribute "version"`);
  }
  return { version, factories };
}
