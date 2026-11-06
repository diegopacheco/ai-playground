import { FactoryfileError } from "./errors";
import type { Agent, Exhausted, Factory, Factoryfile, Loop, Route, Stage } from "./model";
import type { Attribute, Block, ListValue, MapValue, Statement, StringValue } from "./parser";

type Kind = "string" | "list" | "map";

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
  attributes: { title: "string", description: "string", entry: "string", flow: "string", backlog: "string" },
  blocks: ["agent", "stage", "loop"],
};
const AGENT: Rules = {
  ctx: "agent",
  attributes: {
    description: "string",
    role: "string",
    provider: "string",
    model: "string",
    tools: "list",
    source: "string",
    content: "string",
  },
  blocks: [],
};
const STAGE: Rules = {
  ctx: "stage",
  attributes: {
    label: "string",
    description: "string",
    agents: "list",
    run: "string",
    script: "string",
    next: "list",
    artifacts: "list",
    output: "string",
    check: "string",
    verdict: "map",
    source: "string",
    content: "string",
    skills: "list",
    tasks: "string",
    parallel: "string",
    visits: "string",
    end: "string",
    at: "list",
  },
  blocks: ["on"],
};
const ON: Rules = {
  ctx: "on",
  attributes: { to: "string", label: "string", reason: "string", cycle: "string" },
  blocks: ["exhausted"],
};
const EXHAUSTED: Rules = {
  ctx: "exhausted",
  attributes: { label: "string", reason: "string" },
  blocks: ["exhausted"],
};
const LOOP: Rules = {
  ctx: "loop",
  attributes: { description: "string", from: "string", to: "string", until: "string", max: "string" },
  blocks: [],
};

const REFERENCES: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  factory: { entry: "stage" },
  stage: { agents: "agent", next: "stage", tasks: "stage" },
  on: { to: "stage" },
  loop: { from: "stage", to: "stage" },
};

const ARTICLES: Readonly<Record<Kind, string>> = { string: "a string", list: "a list", map: "a map" };
const MAX = /^[1-9][0-9]{0,5}$/;
const AT = /^(0|[1-9][0-9]{0,2})$/;
const VERDICT_KEYS: readonly string[] = ["field", "yes", "no", "list", "notes"];

type Attributes = ReadonlyMap<string, Attribute>;

const key = (kind: string, name: string): string => `${kind}\u0000${name}`;

function optional<K extends string, V>(name: K, value: V | undefined): { [P in K]?: V } {
  return (value === undefined ? {} : { [name]: value }) as { [P in K]?: V };
}

function checkNumber(name: string, { value, line, col }: StringValue): void {
  if (!MAX.test(value)) {
    throw new FactoryfileError(line, col, `invalid ${name} "${value}", expected an integer from 1 to 999999`);
  }
}

function checkAt({ items, line, col }: ListValue): void {
  if (items.length !== 2 || !items.every((item) => AT.test(item.value))) {
    throw new FactoryfileError(line, col, "invalid at, expected two integers from 0 to 999");
  }
}

function checkVerdict({ entries, line, col }: MapValue): void {
  const keys = entries.map((entry) => entry.key).sort();
  const unknown = keys.find((key) => !VERDICT_KEYS.includes(key));
  if (unknown !== undefined) {
    throw new FactoryfileError(line, col, `unknown key "${unknown}" in verdict`);
  }
  if (!keys.includes("field")) {
    throw new FactoryfileError(line, col, `verdict requires "field"`);
  }
}

function checkString(rules: Rules, name: string, attr: StringValue): void {
  const { value, line, col } = attr;
  if (rules === FILE && name === "version" && value !== "1") {
    throw new FactoryfileError(line, col, `unsupported version "${value}"`);
  }
  if ((rules === LOOP && name === "max") || (rules === STAGE && (name === "parallel" || name === "visits"))) {
    checkNumber(name, attr);
  }
  if (rules === ON && name === "cycle" && value !== "true" && value !== "false") {
    throw new FactoryfileError(line, col, `invalid cycle "${value}", expected "true" or "false"`);
  }
}

function checkValue(rules: Rules, attr: Attribute): void {
  const { name, value } = attr;
  if (value.kind === "string") {
    checkString(rules, name, value);
  } else if (value.kind === "list" && rules === STAGE && name === "at") {
    checkAt(value);
  } else if (value.kind === "map" && rules === STAGE && name === "verdict") {
    checkVerdict(value);
  }
}

function collectReference(rules: Rules, owner: string, attr: Attribute, references: Reference[]): void {
  const target = REFERENCES[rules.ctx]?.[attr.name];
  if (target === undefined || attr.value.kind === "map") {
    return;
  }
  const names = attr.value.kind === "list" ? attr.value.items.map((item) => item.value) : [attr.value.value];
  references.push({ target, names, owner, line: attr.value.line, col: attr.value.col });
}

function scan(
  owner: string,
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
    collectReference(rules, owner, statement, references);
    attributes.set(statement.name, statement);
  }
  return attributes;
}

function leaf(block: Block, rules: Rules, references: Reference[]): Attributes {
  return scan(`${rules.ctx} "${block.label}"`, block.body, rules, references, () => undefined);
}

function str(attributes: Attributes, name: string): string | undefined {
  const value = attributes.get(name)?.value;
  return value?.kind === "string" ? value.value : undefined;
}

function list(attributes: Attributes, name: string): readonly string[] {
  const value = attributes.get(name)?.value;
  return value?.kind === "list" ? value.items.map((item) => item.value) : [];
}

function num(attributes: Attributes, name: string): number | undefined {
  const value = str(attributes, name);
  return value === undefined ? undefined : Number(value);
}

function entries(attributes: Attributes, name: string): Record<string, string> | undefined {
  const value = attributes.get(name)?.value;
  return value?.kind === "map" ? Object.fromEntries(value.entries.map((entry) => [entry.key, entry.value.value])) : undefined;
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
    ...optional("provider", str(attributes, "provider")),
    ...optional("model", str(attributes, "model")),
    tools: list(attributes, "tools"),
    ...optional("source", str(attributes, "source")),
    ...optional("content", str(attributes, "content")),
  };
}

function exhaustedBlock(parent: Block, owner: string, references: Reference[], into: Exhausted[]) {
  return (child: Block): void => {
    if (into.length > 0) {
      throw new FactoryfileError(child.line, child.col, `duplicate exhausted in ${parent.keyword} "${parent.label}"`);
    }
    references.push({ target: "stage", names: [child.label], owner, line: child.line, col: child.col });
    into.push(toExhausted(child, owner, references));
  };
}

function toExhausted(block: Block, owner: string, references: Reference[]): Exhausted {
  const next: Exhausted[] = [];
  const attributes = scan(owner, block.body, EXHAUSTED, references, exhaustedBlock(block, owner, references, next));
  return {
    to: block.label,
    ...optional("label", str(attributes, "label")),
    ...optional("reason", str(attributes, "reason")),
    ...optional("exhausted", next[0]),
  };
}

function toRoute(block: Block, owner: string, references: Reference[]): Route {
  const fallback: Exhausted[] = [];
  const attributes = scan(owner, block.body, ON, references, exhaustedBlock(block, owner, references, fallback));
  requires(block, attributes, "to", `"to"`);
  return {
    outcome: block.label,
    to: str(attributes, "to") ?? "",
    ...optional("label", str(attributes, "label")),
    ...optional("reason", str(attributes, "reason")),
    cycle: str(attributes, "cycle") === "true",
    ...optional("exhausted", fallback[0]),
  };
}

function checkRoutes(block: Block, attributes: Attributes, routes: readonly Route[]): void {
  const onNext = routes.some((route) => route.outcome === "next");
  if (attributes.has("next") && onNext) {
    throw new FactoryfileError(block.line, block.col, `stage "${block.label}" cannot set both "next" and on "next"`);
  }
  const other = routes.find((route) => route.outcome !== "next");
  if ((attributes.has("next") || onNext) && other !== undefined) {
    throw new FactoryfileError(
      block.line,
      block.col,
      `stage "${block.label}" cannot mix the next route with on "${other.outcome}"`,
    );
  }
}

function toStage(block: Block, references: Reference[]): Stage {
  const owner = `stage "${block.label}"`;
  const routes: Route[] = [];
  const attributes = scan(owner, block.body, STAGE, references, (child) => {
    if (routes.some((route) => route.outcome === child.label)) {
      throw new FactoryfileError(child.line, child.col, `duplicate on "${child.label}" in stage "${block.label}"`);
    }
    routes.push(toRoute(child, owner, references));
  });
  if (!attributes.has("agents") && !attributes.has("run")) {
    throw new FactoryfileError(block.line, block.col, `stage "${block.label}" requires "agents" or "run"`);
  }
  if (attributes.has("source") && attributes.has("content")) {
    throw new FactoryfileError(block.line, block.col, `stage "${block.label}" cannot set both "source" and "content"`);
  }
  checkRoutes(block, attributes, routes);
  return {
    name: block.label,
    ...optional("label", str(attributes, "label")),
    ...optional("description", str(attributes, "description")),
    agents: list(attributes, "agents"),
    ...optional("run", str(attributes, "run")),
    ...optional("script", str(attributes, "script")),
    next: list(attributes, "next"),
    artifacts: list(attributes, "artifacts"),
    ...optional("output", str(attributes, "output")),
    ...optional("check", str(attributes, "check")),
    ...optional("verdict", entries(attributes, "verdict")),
    ...optional("source", str(attributes, "source")),
    ...optional("content", str(attributes, "content")),
    skills: list(attributes, "skills"),
    ...optional("tasks", str(attributes, "tasks")),
    ...optional("parallel", num(attributes, "parallel")),
    ...optional("visits", num(attributes, "visits")),
    ...optional("end", str(attributes, "end")),
    at: list(attributes, "at").map(Number),
    routes,
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
  const attributes = scan(`factory "${block.label}"`, block.body, FACTORY, references, (child) => {
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
    ...optional("title", str(attributes, "title")),
    ...optional("description", str(attributes, "description")),
    ...optional("entry", str(attributes, "entry") ?? stages[0]?.name),
    ...optional("flow", str(attributes, "flow")),
    ...optional("backlog", str(attributes, "backlog")),
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
